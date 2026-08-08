"""LLaVA short-answer TC visual-lift probe.

This is the short-answer counterpart of the Qwen open-ended probe.  It keeps
the test narrow:

1. Generate a compact candidate set from semantic-preserving prompt views.
2. Score each candidate's first answer token under real and blank images.
3. Select the candidate with the largest prompt-conditioned visual lift.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import logging as hf_logging


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmeval.vlm.llava.llava import LLaVA_Next  # noqa: E402


MODEL_PATH = "/data/shunshungu/models/llama3-llava-next-8b-hf"
LMU_ROOT = Path("/home/shunshungu/LMUData")
OUT_DIR = ROOT / "pipeline/data/llava_vilp_inference"
IMAGE_DIR = OUT_DIR / "images"

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

PRIOR_SUPPRESSED_PROMPT = (
    "The question may mention a common or default fact that conflicts with the image. "
    "Ignore that general prior and answer only the concrete image-specific question.\n"
    "Question: {q}\n"
    "Answer with a single word or short phrase:"
)

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


def clean_answer(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", " ", str(text))
    text = text.replace("\r", "\n").split("\n", 1)[0].strip()
    text = re.sub(r"^(answer|final answer)\s*[:：]\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^(the answer is|it is|this is)\s+", "", text, flags=re.I).strip()
    text = text.strip(" \t\"'`.,;:")
    if len(text) > 80:
        text = text[:80].rsplit(" ", 1)[0].strip()
    return text


def question_tail(question: str) -> str:
    match = re.search(r"(?<=[.!?])\s+", str(question).strip())
    if not match:
        return str(question)
    tail = str(question)[match.end() :].strip()
    return tail if len(tail) >= 12 else str(question)


def label_token_ids(tokenizer, text: str) -> list[int]:
    variants = {
        text,
        text.lower(),
        text.capitalize(),
        " " + text,
        " " + text.lower(),
        " " + text.capitalize(),
    }
    out = []
    bos = getattr(tokenizer, "bos_token_id", None)
    for variant in variants:
        ids = [idx for idx in tokenizer(variant).input_ids if idx != bos]
        if ids:
            out.append(ids[0])
    return sorted(set(out))


def first_token_ids(tokenizer, words: tuple[str, ...]) -> list[int]:
    out = []
    bos = getattr(tokenizer, "bos_token_id", None)
    for word in words:
        ids = [idx for idx in tokenizer(word).input_ids if idx != bos]
        if ids:
            out.append(ids[0])
    return sorted(set(out))


def build_llava(max_new_tokens: int) -> LLaVA_Next:
    hf_logging.set_verbosity_error()
    model = LLaVA_Next(
        model_path=MODEL_PATH,
        model_name="LLaVA-NeXT-8B-Open-Short-Lift-Probe",
        save_logits=False,
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="default",
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    model.kwargs["pad_token_id"] = model.processor.tokenizer.eos_token_id
    model.kwargs["temperature"] = None
    return model


def row_key(row: dict) -> str:
    return str(row["index"])


def materialize_image(row: dict, dataset_stem: str) -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / f"{dataset_stem}_{row_key(row)}.jpg"
    if path.exists():
        return path
    if row.get("image"):
        image = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
    elif row.get("image_path"):
        image = Image.open(row["image_path"]).convert("RGB")
    else:
        raise ValueError("row has no image/image_path")
    image.save(path, format="JPEG", quality=85)
    return path


def generation_kwargs(model: LLaVA_Next, max_new_tokens: int, num_beams: int = 1) -> dict:
    return {
        **model.kwargs,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
        "return_dict_in_generate": False,
        "output_logits": False,
    }


def logits_kwargs(model: LLaVA_Next) -> dict:
    return {
        **model.kwargs,
        "max_new_tokens": 1,
        "return_dict_in_generate": True,
        "output_logits": True,
    }


@torch.inference_mode()
def generate_answers(
    model: LLaVA_Next,
    image_file: Path,
    prompt: str,
    max_new_tokens: int,
    num_beams: int = 1,
) -> list[str]:
    old_kwargs = model.kwargs
    model.kwargs = generation_kwargs(model, max_new_tokens, num_beams)
    try:
        output_ids = model.generate_ids_or_logits(
            [{"type": "image", "value": str(image_file)}, {"type": "text", "value": prompt}],
            visual_type="default",
            textual_type="default",
            get_logits=False,
        )
    finally:
        model.kwargs = old_kwargs
    decoded = model.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    answers = []
    for text in decoded:
        answer = clean_answer(model.output_process(text))
        if answer and answer not in answers:
            answers.append(answer)
    return answers


def generate_answer(model: LLaVA_Next, image_file: Path, prompt: str, max_new_tokens: int) -> str:
    answers = generate_answers(model, image_file, prompt, max_new_tokens, num_beams=1)
    return answers[0] if answers else ""


@torch.inference_mode()
def token_score(
    model: LLaVA_Next,
    image_file: Path,
    prompt: str,
    visual_type: str,
    token_ids: list[int],
) -> float:
    old_kwargs = model.kwargs
    model.kwargs = logits_kwargs(model)
    try:
        logits = model.generate_ids_or_logits(
            [{"type": "image", "value": str(image_file)}, {"type": "text", "value": prompt}],
            visual_type=visual_type,
            textual_type="default",
            get_logits=True,
        )[0, 0].detach().float()
    finally:
        model.kwargs = old_kwargs
    logp = torch.log_softmax(logits, dim=-1)
    return max(float(logp[idx].item()) for idx in token_ids)


def support_score(
    model: LLaVA_Next,
    image_file: Path,
    prompt: str,
    visual_type: str,
    yes_ids: list[int],
    no_ids: list[int],
) -> float:
    yes = token_score(model, image_file, prompt, visual_type, yes_ids)
    no = token_score(model, image_file, prompt, visual_type, no_ids)
    return yes - no


def choose(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get) if scores else ""


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


def summarize(records: list[dict]) -> dict:
    keys = [
        "default_hit",
        "answer_format_hit",
        "paraphrase_hit",
        "prior_suppressed_hit",
        "question_tail_hit",
        "beam_oracle_hit",
        "oracle_hit",
        "real_verify_hit",
        "lift_verify_hit",
        "direct_real_hit",
        "direct_lift_hit",
    ]
    def mean_key(key: str) -> float:
        values = [rec[key] for rec in records if rec.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    return {
        "n": len(records),
        "metrics": {key: mean_key(key) for key in keys},
        "avg_candidates": float(np.mean([len(rec["candidates"]) for rec in records])) if records else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ViLP_LLaVA-NeXT-8B_Biased_Test.tsv")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--include-prior-suppressed", action="store_true")
    parser.add_argument("--include-question-tail", action="store_true")
    parser.add_argument("--beam-candidates", type=int, default=1)
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--score-prompt-names", default="default,answer_format,paraphrase")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_stem = Path(args.dataset).stem
    suffix_parts = []
    if args.include_prior_suppressed:
        suffix_parts.append("prior")
    if args.include_question_tail:
        suffix_parts.append("tail")
    if args.beam_candidates > 1:
        suffix_parts.append(f"beam{args.beam_candidates}")
    if args.skip_verification:
        suffix_parts.append("noverify")
    score_prompt_names = [name.strip() for name in args.score_prompt_names.split(",") if name.strip()]
    if score_prompt_names != ["default", "answer_format", "paraphrase"]:
        suffix_parts.append("score-" + "-".join(score_prompt_names))
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    out_path = OUT_DIR / f"{dataset_stem}_n{args.n}_seed{args.seed}{suffix}.jsonl"
    done = load_jsonl(out_path)

    rows = pd.read_csv(LMU_ROOT / args.dataset, sep="\t").to_dict("records")
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    model = build_llava(args.max_new_tokens)
    tokenizer = model.processor.tokenizer
    yes_ids = first_token_ids(tokenizer, ("Yes", " yes", "yes", " Yes"))
    no_ids = first_token_ids(tokenizer, ("No", " no", "no", " No"))
    gen_prompts = dict(GEN_PROMPTS)
    if args.include_prior_suppressed:
        gen_prompts["prior_suppressed"] = PRIOR_SUPPRESSED_PROMPT

    with out_path.open("a", encoding="utf-8") as writer:
        for pos, row in enumerate(rows, start=1):
            key = row_key(row)
            if key in done:
                continue
            image_file = materialize_image(row, dataset_stem)
            q = str(row["question"])
            answer = str(row["answer"])
            prompt_texts = {name: template.format(q=q) for name, template in gen_prompts.items()}
            if args.include_question_tail:
                prompt_texts["question_tail"] = GEN_PROMPTS["default"].format(q=question_tail(q))
            missing_score_prompts = [name for name in score_prompt_names if name not in prompt_texts]
            if missing_score_prompts:
                raise ValueError(f"score prompts not generated: {missing_score_prompts}")

            generated = {}
            candidates = []
            for name, prompt in prompt_texts.items():
                cand = generate_answer(model, image_file, prompt, args.max_new_tokens)
                generated[name] = cand
                if cand and cand not in candidates:
                    candidates.append(cand)
            beam_candidates = []
            if args.beam_candidates > 1:
                beam_candidates = generate_answers(
                    model,
                    image_file,
                    prompt_texts["default"],
                    args.max_new_tokens,
                    num_beams=args.beam_candidates,
                )
                generated[f"default_beam{args.beam_candidates}"] = beam_candidates
                for cand in beam_candidates:
                    if cand and cand not in candidates:
                        candidates.append(cand)

            direct_real_scores = {}
            direct_lift_scores = {}
            real_verify_scores = {}
            lift_verify_scores = {}
            for cand in candidates:
                if not args.skip_verification:
                    real_verify_by_view = []
                    lift_verify_by_view = []
                    for template in VERIFY_PROMPTS.values():
                        verify_prompt = template.format(q=q, c=cand)
                        real = support_score(model, image_file, verify_prompt, "default", yes_ids, no_ids)
                        prior = support_score(model, image_file, verify_prompt, "vcf_color0", yes_ids, no_ids)
                        real_verify_by_view.append(real)
                        lift_verify_by_view.append(real - prior)
                    real_verify_scores[cand] = max(real_verify_by_view)
                    lift_verify_scores[cand] = max(lift_verify_by_view)

                token_ids = label_token_ids(tokenizer, cand)
                if not token_ids:
                    continue
                real_by_view = []
                lift_by_view = []
                for name in score_prompt_names:
                    prompt = prompt_texts[name]
                    real = token_score(model, image_file, prompt, "default", token_ids)
                    prior = token_score(model, image_file, prompt, "vcf_color0", token_ids)
                    real_by_view.append(real)
                    lift_by_view.append(real - prior)
                direct_real_scores[cand] = max(real_by_view)
                direct_lift_scores[cand] = max(lift_by_view)

            real_verify_choice = choose(real_verify_scores)
            lift_verify_choice = choose(lift_verify_scores)
            direct_real_choice = choose(direct_real_scores)
            direct_lift_choice = choose(direct_lift_scores)
            rec = {
                "key": key,
                "question": q,
                "answer": answer,
                "generated": generated,
                "candidates": candidates,
                "real_verify_scores": real_verify_scores,
                "lift_verify_scores": lift_verify_scores,
                "direct_real_scores": direct_real_scores,
                "direct_lift_scores": direct_lift_scores,
                "real_verify_choice": real_verify_choice,
                "lift_verify_choice": lift_verify_choice,
                "direct_real_choice": direct_real_choice,
                "direct_lift_choice": direct_lift_choice,
                "default_hit": paper_hit(answer, generated.get("default", "")),
                "answer_format_hit": paper_hit(answer, generated.get("answer_format", "")),
                "paraphrase_hit": paper_hit(answer, generated.get("paraphrase", "")),
                "prior_suppressed_hit": (
                    paper_hit(answer, generated.get("prior_suppressed", ""))
                    if "prior_suppressed" in generated
                    else None
                ),
                "question_tail_hit": (
                    paper_hit(answer, generated.get("question_tail", ""))
                    if "question_tail" in generated
                    else None
                ),
                "beam_oracle_hit": (
                    float(any(paper_hit(answer, cand) for cand in beam_candidates))
                    if args.beam_candidates > 1
                    else None
                ),
                "oracle_hit": float(any(paper_hit(answer, cand) for cand in candidates)),
                "real_verify_hit": None if args.skip_verification else paper_hit(answer, real_verify_choice),
                "lift_verify_hit": None if args.skip_verification else paper_hit(answer, lift_verify_choice),
                "direct_real_hit": paper_hit(answer, direct_real_choice),
                "direct_lift_hit": paper_hit(answer, direct_lift_choice),
            }
            writer.write(json.dumps(rec, ensure_ascii=True) + "\n")
            writer.flush()
            done[key] = rec
            if pos % 10 == 0:
                print(f"materialized {pos}/{len(rows)} current={summarize(list(done.values()))}", flush=True)

    records = [done[row_key(row)] for row in rows if row_key(row) in done]
    summary = summarize(records)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
