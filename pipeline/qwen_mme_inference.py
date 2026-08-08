"""Open-ended TC visual-lift probe for short-answer VQA.

MCQ visual-lift selects the option whose real-vs-blank evidence gain is largest.
For open-ended rows, there is no option set, so this probe first builds a compact
candidate set from semantic-preserving prompt views, then reranks candidates by
yes/no visual support lift:

    lift(p, c) = support(real image, p, c) - support(blank image, p, c)

where support is log P(yes) - log P(no) for a candidate-answer verification
prompt.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = "/data/shunshungu/models/Qwen2-VL-7B-Instruct"
LMU_ROOT = Path("/home/shunshungu/LMUData")
OUT_DIR = ROOT / "pipeline/data/qwen_mme_inference"

GEN_PROMPTS = {
    "default": "{q}\nAnswer the question directly using a single word or short phrase.",
    "answer_format": (
        "Look at the image and answer the question. Use only the final answer, no explanation.\n"
        "Question: {q}\nAnswer:"
    ),
    "paraphrase": (
        "Based on the image, what is the answer to this question?\n"
        "{q}\nReply with a concise noun phrase or short answer."
    ),
}

VERIFY_PROMPTS = {
    "default": (
        "Question: {q}\n"
        "Candidate answer: {c}\n"
        "Based only on the image, is the candidate answer correct? Answer yes or no."
    ),
    "answer_format": (
        "Does the image support '{c}' as the answer to the question '{q}'? "
        "Answer yes or no."
    ),
}

BINARY_PROMPTS = {
    "default": "{q}\nAnswer yes or no.",
    "answer_format": (
        "Look at the image and answer the question.\n"
        "Question: {q}\nAnswer only Yes or No:"
    ),
    "paraphrase": (
        "Based on the image, determine the correct yes/no answer.\n"
        "{q}\nAnswer:"
    ),
}

BINARY_OPTION_PROMPTS = {
    "default": (
        "{q}\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option letter."
    ),
    "answer_format": (
        "Look at the image and answer the question.\n"
        "Question: {q}\n"
        "Options:\n"
        "A. Yes\n"
        "B. No\n"
        "Answer only A or B."
    ),
    "paraphrase": (
        "Based on the image, choose the correct yes/no answer.\n"
        "{q}\n"
        "A. Yes\n"
        "B. No\n"
        "Final answer:"
    ),
}


def paper_hit(answer: str, prediction: str) -> float:
    ans = str(answer).lower().strip()
    pred = str(prediction).lower().strip()
    if len(pred) < len(ans):
        return 0.0
    if len(pred) == len(ans):
        return float(pred == ans)
    if not bool(re.fullmatch(r"[A-Za-z]", pred[len(ans)])):
        return float(pred[: len(ans)] == ans)
    return 0.0


def load_image(row: dict) -> Image.Image:
    if row.get("image"):
        return Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
    if row.get("image_path"):
        return Image.open(row["image_path"]).convert("RGB")
    raise ValueError("row has no image/image_path")


def clean_answer(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", " ", str(text))
    text = text.replace("\r", "\n").split("\n", 1)[0].strip()
    text = re.sub(r"^(answer|final answer)\s*[:：]\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^(the answer is|it is|this is)\s+", "", text, flags=re.I).strip()
    text = text.strip(" \t\"'`.,;:")
    if len(text) > 80:
        text = text[:80].rsplit(" ", 1)[0].strip()
    return text


def first_token_ids(tokenizer, words: tuple[str, ...]) -> list[int]:
    out = []
    for word in words:
        ids = [idx for idx in tokenizer(word).input_ids if idx != getattr(tokenizer, "bos_token_id", None)]
        if ids:
            out.append(ids[0])
    return sorted(set(out))


def label_token_ids(tokenizer, label: str) -> list[int]:
    variants = (label, " " + label, label + ".", " " + label + ".")
    return first_token_ids(tokenizer, variants)


def apply_chat(processor, image: Image.Image, prompt: str):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    return processor(text=[text], images=images, videos=videos, return_tensors="pt")


@torch.inference_mode()
def generate_answer(model, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    inputs = apply_chat(processor, image, prompt).to("cuda")
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
    new_ids = generated[0, inputs.input_ids.shape[1] :]
    return clean_answer(processor.tokenizer.decode(new_ids, skip_special_tokens=True))


@torch.inference_mode()
def support_score(model, processor, image: Image.Image, prompt: str, yes_ids: list[int], no_ids: list[int]) -> float:
    inputs = apply_chat(processor, image, prompt).to("cuda")
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    yes = max(float(logp[idx].item()) for idx in yes_ids)
    no = max(float(logp[idx].item()) for idx in no_ids)
    return yes - no


@torch.inference_mode()
def next_token_logprob(model, processor, image: Image.Image, prompt: str, token_ids: list[int]) -> float:
    inputs = apply_chat(processor, image, prompt).to("cuda")
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    return max(float(logp[idx].item()) for idx in token_ids)


@torch.inference_mode()
def option_logprobs(model, processor, image: Image.Image, prompt: str, label_ids: dict[str, list[int]]) -> dict[str, float]:
    inputs = apply_chat(processor, image, prompt).to("cuda")
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    return {
        label: max(float(logp[idx].item()) for idx in ids)
        for label, ids in label_ids.items()
        if ids
    }


def candidate_token_ids(tokenizer, candidate: str) -> list[int]:
    variants = {candidate, candidate.lower(), candidate.capitalize(), " " + candidate, " " + candidate.lower()}
    out = []
    for text in variants:
        ids = [idx for idx in tokenizer(text).input_ids if idx != getattr(tokenizer, "bos_token_id", None)]
        if ids:
            out.append(ids[0])
    return sorted(set(out))


def is_binary_candidate_set(answer: str, candidates: list[str]) -> bool:
    values = {str(answer).strip().lower(), *(cand.strip().lower() for cand in candidates)}
    return bool(values) and values.issubset({"yes", "no"})


def is_binary_answer(answer: str) -> bool:
    return str(answer).strip().lower() in {"yes", "no"}


def choose_by(values: dict[str, float]) -> str:
    return max(values, key=values.get) if values else ""


def summarize(records: list[dict]) -> dict:
    keys = [
        "default_hit",
        "answer_format_hit",
        "paraphrase_hit",
        "oracle_hit",
        "real_verify_hit",
        "lift_verify_hit",
        "direct_real_hit",
        "direct_lift_hit",
        "binary_option_real_hit",
        "binary_option_lift_hit",
        "hybrid_lift_hit",
    ]
    def mean_key(key: str) -> float:
        values = [rec[key] for rec in records if rec.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    return {
        "n": len(records),
        "metrics": {key: mean_key(key) for key in keys},
        "avg_candidates": float(np.mean([len(rec["candidates"]) for rec in records])) if records else 0.0,
        "binary_rows": int(sum(bool(rec.get("is_binary")) for rec in records)),
    }


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["key"]] = rec
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ViLP_Qwen2-VL-7B_Biased_Test.tsv")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{Path(args.dataset).stem}_v4_n{args.n}_seed{args.seed}.jsonl"
    done = load_jsonl(out_path)

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=256 * 28 * 28,
        max_pixels=896 * 28 * 28,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    tokenizer = processor.tokenizer
    yes_ids = first_token_ids(tokenizer, ("Yes", " yes", "yes", " Yes"))
    no_ids = first_token_ids(tokenizer, ("No", " no", "no", " No"))
    binary_label_ids = {"A": label_token_ids(tokenizer, "A"), "B": label_token_ids(tokenizer, "B")}

    rows = pd.read_csv(LMU_ROOT / args.dataset, sep="\t").to_dict("records")
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    with out_path.open("a", encoding="utf-8") as writer:
        for pos, row in enumerate(rows, start=1):
            key = str(row["index"])
            if key in done:
                continue
            image = load_image(row)
            blank = Image.new("RGB", image.size, (0, 0, 0))
            q = str(row["question"])
            answer = str(row["answer"])

            generated = {}
            candidates = []
            for name, template in GEN_PROMPTS.items():
                cand = generate_answer(model, processor, image, template.format(q=q), args.max_new_tokens)
                generated[name] = cand
                if cand and cand not in candidates:
                    candidates.append(cand)
            if is_binary_answer(answer):
                candidates = ["Yes", "No"]

            real_scores: dict[str, float] = {}
            lift_scores: dict[str, float] = {}
            direct_real_scores: dict[str, float] = {}
            direct_lift_scores: dict[str, float] = {}
            binary_option_real_scores: dict[str, float] = {}
            binary_option_lift_scores: dict[str, float] = {}
            for cand in candidates:
                real_by_view = []
                lift_by_view = []
                for template in VERIFY_PROMPTS.values():
                    prompt = template.format(q=q, c=cand)
                    real = support_score(model, processor, image, prompt, yes_ids, no_ids)
                    prior = support_score(model, processor, blank, prompt, yes_ids, no_ids)
                    real_by_view.append(real)
                    lift_by_view.append(real - prior)
                real_scores[cand] = max(real_by_view)
                lift_scores[cand] = max(lift_by_view)
                token_ids = candidate_token_ids(tokenizer, cand)
                if token_ids:
                    direct_prompt_templates = BINARY_PROMPTS if is_binary_answer(answer) else GEN_PROMPTS
                    direct_real_by_view = []
                    direct_by_view = []
                    for template in direct_prompt_templates.values():
                        prompt = template.format(q=q)
                        real = next_token_logprob(model, processor, image, prompt, token_ids)
                        prior = next_token_logprob(model, processor, blank, prompt, token_ids)
                        direct_real_by_view.append(real)
                        direct_by_view.append(real - prior)
                    direct_real_scores[cand] = max(direct_real_by_view)
                    direct_lift_scores[cand] = max(direct_by_view)

            if is_binary_answer(answer):
                option_real_by_label = {"A": [], "B": []}
                option_lift_by_label = {"A": [], "B": []}
                for template in BINARY_OPTION_PROMPTS.values():
                    prompt = template.format(q=q)
                    real_scores_by_label = option_logprobs(model, processor, image, prompt, binary_label_ids)
                    blank_scores_by_label = option_logprobs(model, processor, blank, prompt, binary_label_ids)
                    for label in ("A", "B"):
                        real = real_scores_by_label[label]
                        prior = blank_scores_by_label[label]
                        option_real_by_label[label].append(real)
                        option_lift_by_label[label].append(real - prior)
                binary_option_real_scores["Yes"] = max(option_real_by_label["A"])
                binary_option_real_scores["No"] = max(option_real_by_label["B"])
                binary_option_lift_scores["Yes"] = max(option_lift_by_label["A"])
                binary_option_lift_scores["No"] = max(option_lift_by_label["B"])

            real_choice = choose_by(real_scores)
            lift_choice = choose_by(lift_scores)
            direct_real_choice = choose_by(direct_real_scores)
            direct_lift_choice = choose_by(direct_lift_scores)
            binary_option_real_choice = choose_by(binary_option_real_scores)
            binary_option_lift_choice = choose_by(binary_option_lift_scores)
            hybrid_choice = binary_option_lift_choice if is_binary_candidate_set(answer, candidates) else lift_choice
            rec = {
                "key": key,
                "is_binary": is_binary_answer(answer),
                "question": q,
                "answer": answer,
                "generated": generated,
                "candidates": candidates,
                "real_scores": real_scores,
                "lift_scores": lift_scores,
                "direct_real_scores": direct_real_scores,
                "direct_lift_scores": direct_lift_scores,
                "binary_option_real_scores": binary_option_real_scores,
                "binary_option_lift_scores": binary_option_lift_scores,
                "real_choice": real_choice,
                "lift_choice": lift_choice,
                "direct_real_choice": direct_real_choice,
                "direct_lift_choice": direct_lift_choice,
                "binary_option_real_choice": binary_option_real_choice,
                "binary_option_lift_choice": binary_option_lift_choice,
                "hybrid_choice": hybrid_choice,
                "default_hit": paper_hit(answer, generated.get("default", "")),
                "answer_format_hit": paper_hit(answer, generated.get("answer_format", "")),
                "paraphrase_hit": paper_hit(answer, generated.get("paraphrase", "")),
                "oracle_hit": float(any(paper_hit(answer, cand) for cand in candidates)),
                "real_verify_hit": paper_hit(answer, real_choice),
                "lift_verify_hit": paper_hit(answer, lift_choice),
                "direct_real_hit": paper_hit(answer, direct_real_choice),
                "direct_lift_hit": paper_hit(answer, direct_lift_choice),
                "binary_option_real_hit": (
                    paper_hit(answer, binary_option_real_choice) if is_binary_answer(answer) else None
                ),
                "binary_option_lift_hit": (
                    paper_hit(answer, binary_option_lift_choice) if is_binary_answer(answer) else None
                ),
                "hybrid_lift_hit": paper_hit(answer, hybrid_choice),
            }
            writer.write(json.dumps(rec, ensure_ascii=True) + "\n")
            writer.flush()
            done[key] = rec
            if pos % 10 == 0:
                print(f"materialized {pos}/{len(rows)} current={summarize(list(done.values()))}", flush=True)

    records = [done[str(row["index"])] for row in rows if str(row["index"]) in done]
    summary = summarize(records)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
