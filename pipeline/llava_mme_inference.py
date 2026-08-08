"""LLaVA open-ended yes/no TC visual-lift probe.

This is the LLaVA counterpart of the Qwen MME binary branch.  It casts a
yes/no row into a two-option problem:

    A. Yes
    B. No

and scores the A/B option tokens under real and blank images.  This keeps the
binary open-ended branch aligned with the MCQ visual-lift design.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import logging as hf_logging


ROOT = Path(__file__).resolve().parents[1]

from vlmeval.vlm.llava.llava import LLaVA_Next  # noqa: E402


MODEL_PATH = "/data/shunshungu/models/llama3-llava-next-8b-hf"
LMU_ROOT = Path("/home/shunshungu/LMUData")
OUT_DIR = ROOT / "pipeline/data/llava_mme_inference"
IMAGE_DIR = OUT_DIR / "images"

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


def label_token_ids(tokenizer, label: str) -> list[int]:
    out = []
    for text in (label, " " + label, label + ".", " " + label + "."):
        ids = [idx for idx in tokenizer(text).input_ids if idx != getattr(tokenizer, "bos_token_id", None)]
        if ids:
            out.append(ids[0])
    return sorted(set(out))


def build_llava() -> LLaVA_Next:
    hf_logging.set_verbosity_error()
    model = LLaVA_Next(
        model_path=MODEL_PATH,
        model_name="LLaVA-NeXT-8B-Open-Binary-Lift-Probe",
        save_logits=False,
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type="default",
        textual_type="default",
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_logits=True,
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
        import io

        image = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
    elif row.get("image_path"):
        image = Image.open(row["image_path"]).convert("RGB")
    else:
        raise ValueError("row has no image/image_path")
    image.save(path, format="JPEG", quality=90)
    return path


def logprobs_for_labels(logits: torch.Tensor, label_ids: dict[str, list[int]]) -> dict[str, float]:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return {
        label: max(float(logp[idx].item()) for idx in ids)
        for label, ids in label_ids.items()
    }


def score_prompt(
    model: LLaVA_Next,
    image_file: Path,
    prompt: str,
    visual_type: str,
    label_ids: dict[str, list[int]],
) -> dict[str, float]:
    message = [
        {"type": "image", "value": str(image_file)},
        {"type": "text", "value": prompt},
    ]
    with torch.inference_mode():
        logits = model.generate_ids_or_logits(
            message,
            visual_type=visual_type,
            textual_type="default",
            get_logits=True,
        )[0, 0].detach().cpu()
    return logprobs_for_labels(logits, label_ids)


def choose(scores: dict[str, float]) -> str:
    label = max(scores, key=scores.get)
    return "Yes" if label == "A" else "No"


def hit(answer: str, pred: str) -> float:
    return float(str(answer).strip().lower() == str(pred).strip().lower())


def margin(scores: dict[str, float]) -> float:
    values = sorted(scores.values(), reverse=True)
    return values[0] - values[1] if len(values) > 1 else 0.0


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
        "answer_format_real_hit",
        "max_real_hit",
        "max_lift_hit",
        "default_lift_hit",
        "answer_format_lift_hit",
    ]
    return {
        "n": len(records),
        "metrics": {key: float(np.mean([rec[key] for rec in records])) if records else 0.0 for key in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MME_LLaVA-NeXT-8B_Biased_Test.tsv")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_stem = Path(args.dataset).stem
    out_path = OUT_DIR / f"{dataset_stem}_n{args.n}_seed{args.seed}.jsonl"
    done = load_jsonl(out_path)

    rows = pd.read_csv(LMU_ROOT / args.dataset, sep="\t").to_dict("records")
    rows = rows[: args.n]

    model = build_llava()
    tokenizer = model.processor.tokenizer
    label_ids = {"A": label_token_ids(tokenizer, "A"), "B": label_token_ids(tokenizer, "B")}

    with out_path.open("a", encoding="utf-8") as writer:
        for idx, row in enumerate(rows, start=1):
            key = row_key(row)
            if key in done:
                continue
            image_file = materialize_image(row, dataset_stem)
            q = str(row["question"])
            answer = str(row["answer"]).strip().capitalize()

            real_by_prompt = {}
            blank_by_prompt = {}
            lift_by_prompt = {}
            for name, template in BINARY_OPTION_PROMPTS.items():
                prompt = template.format(q=q)
                real = score_prompt(model, image_file, prompt, "default", label_ids)
                blank = score_prompt(model, image_file, prompt, "vcf_color0", label_ids)
                real_by_prompt[name] = real
                blank_by_prompt[name] = blank
                lift_by_prompt[name] = {label: real[label] - blank[label] for label in real}

            max_real = {
                label: max(scores[label] for scores in real_by_prompt.values())
                for label in ("A", "B")
            }
            max_lift = {
                label: max(scores[label] for scores in lift_by_prompt.values())
                for label in ("A", "B")
            }
            rec = {
                "key": key,
                "answer": answer,
                "question": q,
                "real_scores": real_by_prompt,
                "blank_scores": blank_by_prompt,
                "lift_scores": lift_by_prompt,
                "answer_format_real_choice": choose(real_by_prompt["answer_format"]),
                "max_real_choice": choose(max_real),
                "max_lift_choice": choose(max_lift),
                "default_lift_choice": choose(lift_by_prompt["default"]),
                "answer_format_lift_choice": choose(lift_by_prompt["answer_format"]),
                "max_lift_margin": margin(max_lift),
                "answer_format_real_hit": hit(answer, choose(real_by_prompt["answer_format"])),
                "max_real_hit": hit(answer, choose(max_real)),
                "max_lift_hit": hit(answer, choose(max_lift)),
                "default_lift_hit": hit(answer, choose(lift_by_prompt["default"])),
                "answer_format_lift_hit": hit(answer, choose(lift_by_prompt["answer_format"])),
            }
            writer.write(json.dumps(rec, ensure_ascii=True) + "\n")
            writer.flush()
            done[key] = rec
            if idx % 20 == 0:
                print(f"materialized {idx}/{len(rows)} current={summarize(list(done.values()))}", flush=True)

    records = [done[row_key(row)] for row in rows if row_key(row) in done]
    summary = summarize(records)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
