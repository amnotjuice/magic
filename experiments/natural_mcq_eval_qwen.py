"""Evaluate Qwen CDA-v3 final predictions on natural MCQ rows.

This fills natural final-hit evidence for the conformal act gate.  It can
materialize all Custom_Val rows for strict harm calibration, or only the rows a
frozen gate would touch on Custom_Test.  The operator is the same MCQ
GREC-energy / joint-verifier rule used by ``natural_mcq_eval_qwen.py``.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFilter
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
LMU_ROOT = Path("/data/shunshungu/LMUData")
OUT_DIR = ROOT / "experiments/data/natural_mcq_eval_cache"
MODEL_PATH = "/data/shunshungu/models/Qwen2-VL-7B-Instruct"
DATASETS = (
    "MMStar",
    "MME",
    "ViLP",
    "CCBench",
    "MMBench_DEV_EN_V11",
    "MMBench_DEV_CN_V11",
)
DEFAULT_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def option_labels(row: dict) -> list[str]:
    labels = []
    for label in "ABCD":
        value = row.get(label, "")
        if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan":
            labels.append(label)
    return labels


def build_prompt(row: dict, labels: list[str], instruction: str = DEFAULT_INSTRUCTION) -> str:
    options = "\n".join(f"{label}. {row[label]}" for label in labels)
    return f"{row['question']}\n{options}\n{instruction}"


def prompt_tcf1(row: dict, labels: list[str]) -> str:
    return build_prompt(
        row,
        labels,
        "Think about the question based on details in the given image. "
        "Answer with the option's letter from the given choices directly.",
    )


def prompt_tcf2(row: dict, labels: list[str]) -> str:
    return build_prompt(
        row,
        labels,
        "Please carefully examine the information in the image, then consider the question and choices, "
        "and reply directly with the letter corresponding to the correct answer.",
    )


def prompt_answer_format(row: dict, labels: list[str]) -> str:
    return build_prompt(row, labels, "Return only the final option letter, with no explanation.")


def load_image(row: dict) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
    except Exception:
        return None


def center_mask_image(image: Image.Image, keep_context: float = 0.5) -> Image.Image:
    from PIL import ImageDraw

    image = image.convert("RGB")
    masked = image.copy()
    w, h = image.size
    box_w = round(w * keep_context)
    box_h = round(h * keep_context)
    left = (w - box_w) // 2
    upper = (h - box_h) // 2
    fill = tuple(int(v) for v in image.resize((1, 1)).getpixel((0, 0)))
    ImageDraw.Draw(masked).rectangle((left, upper, left + box_w, upper + box_h), fill=fill)
    return masked


def add_diffusion_noise(image: Image.Image, noise_step: int, seed: int) -> Image.Image:
    from torchvision import transforms

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        image_tensor = transforms.ToTensor()(image)
        betas = torch.linspace(-6, 6, 1000)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        alphas_prod = torch.cumprod(1 - betas, dim=0)
        noise = torch.randn_like(image_tensor)
        noisy = torch.sqrt(alphas_prod[noise_step]) * image_tensor
        noisy += torch.sqrt(1 - alphas_prod[noise_step]) * noise
    return transforms.ToPILImage()(noisy.clamp(0, 1))


def transform_image(image: Image.Image, kind: str, seed: int) -> Image.Image:
    if kind == "default":
        return image
    if kind == "blank":
        return Image.new("RGB", image.size, (0, 0, 0))
    if kind == "noise500":
        return add_diffusion_noise(image, 500, seed)
    if kind == "gray":
        return image.convert("L").convert("RGB")
    if kind == "strongblur":
        return image.filter(ImageFilter.GaussianBlur(radius=100))
    if kind == "centermask":
        return center_mask_image(image)
    raise ValueError(f"unknown image transform: {kind}")


@torch.no_grad()
def first_logits(model, processor, image: Image.Image, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
    return model(**inputs).logits[0, -1, :].float().cpu()


def letter_token_ids(tokenizer, label: str) -> list[int]:
    ids = set()
    for text in (label, label.lower(), f" {label}", f" {label.lower()}"):
        toks = tokenizer(text, add_special_tokens=False).input_ids
        if toks:
            ids.add(int(toks[-1]))
    return sorted(ids)


def option_logprobs(logits: torch.Tensor, labels: list[str], label_ids: dict[str, list[int]]) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    values = []
    for label in labels:
        values.append(max(float(logp[token_id].item()) for token_id in label_ids[label]))
    return torch.tensor(values, dtype=torch.float64)


def centered(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return torch.zeros_like(values)
    fill = values[finite].min() - 20.0
    values = torch.where(finite, values, fill)
    return values - values.mean()


def pred_label(scores: torch.Tensor, labels: list[str]) -> str:
    return labels[int(scores.argmax().item())]


def scores_json(scores: torch.Tensor, labels: list[str]) -> str:
    return json.dumps(
        {label: float(scores[i].item()) for i, label in enumerate(labels)},
        sort_keys=True,
        separators=(",", ":"),
    )


def labels_json(labels: list[str]) -> str:
    return json.dumps(labels, separators=(",", ":"))


def scores_from_json(text: str, labels: list[str]) -> torch.Tensor:
    data = json.loads(text)
    return torch.tensor([float(data[label]) for label in labels], dtype=torch.float64)


def maxp(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    return float(probs.max().item())


def prob_at(logits: torch.Tensor, token_id: int) -> float:
    logden = torch.logsumexp(logits.float(), dim=-1)
    return float(torch.exp(logits.float()[token_id] - logden).item())


def top_retention(orig_logits: torch.Tensor, cf_logits: torch.Tensor) -> float:
    logden = torch.logsumexp(orig_logits.float(), dim=-1)
    top2 = torch.topk(orig_logits.float(), k=2).values
    p0 = float(torch.exp(top2[0] - logden).item())
    y0 = int(orig_logits.argmax().item())
    ratio = min(1.0, prob_at(cf_logits, y0) / max(p0, 1e-12))
    return ratio if int(cf_logits.argmax().item()) == y0 else 0.0


def route_mode(t_on: bool, v_on: bool) -> str:
    if t_on and v_on:
        return "both"
    if t_on:
        return "tcf_only"
    if v_on:
        return "vcf_only"
    return "neither"


def grec_scores(base: torch.Tensor, tc: torch.Tensor, vc: torch.Tensor, route: str) -> tuple[torch.Tensor, str]:
    if route == "neither":
        return base, "stable_preserve"
    if route == "tcf_only":
        return tc, "not_joint"
    if route == "vcf_only":
        return vc, "not_joint"

    labels_n = max(int(base.numel()), 1)
    tc_weights = torch.softmax(tc.float(), dim=0).to(vc.dtype)
    joint = tc + tc_weights * vc
    joint_idx = int(joint.argmax().item())
    if float(tc_weights[joint_idx].item()) >= 1.0 / labels_n and float(vc[joint_idx].item()) > 0.0:
        return joint, "joint_verified"

    tc_idx = int(tc.argmax().item())
    if float(tc_weights[tc_idx].item()) >= 1.0 / labels_n:
        return tc, "fallback_tc"

    vc_idx = int(vc.argmax().item())
    if float(vc[vc_idx].item()) > 0.0:
        return vc, "fallback_vc"

    return base, "fallback_original"


def load_custom_rows(split: str) -> dict[tuple[str, str], dict]:
    rows = {}
    for dataset in DATASETS:
        path = LMU_ROOT / f"{dataset}_Custom_{split}.tsv"
        if not path.exists():
            continue
        for row in pd.read_csv(path, sep="\t").to_dict("records"):
            rows[(dataset, str(int(float(row["index"]))))] = row
    return rows


def load_tau(epsilon: str) -> float:
    cal = json.loads((OUT_DIR / "calibration.json").read_text(encoding="utf-8"))
    tau = cal["qwen"]["epsilons"].get(epsilon)
    if tau is None:
        raise RuntimeError(f"missing Qwen calibration tau for epsilon={epsilon}")
    return float(tau)


def detector_thresholds() -> tuple[float, float]:
    summary = json.loads((ROOT / "experiments/data/retention_detector/summary.json").read_text(encoding="utf-8"))
    block = summary["qwen"]["evaluations"]["retention_top"]["thresholds"]
    return float(block["T"]), float(block["V"])


def msp_rows(split: str) -> list[dict]:
    return [
        row for row in read_csv(OUT_DIR / f"qwen_custom_{split.lower()}_msp.csv")
        if str(row.get("msp_available", "")).strip() == "1" and str(row.get("original_msp", "")).strip()
    ]


def selected_msp_rows(split: str, selection: str, epsilon: str, tau: float | None) -> list[dict]:
    rows = msp_rows(split)
    if selection == "all":
        return rows
    threshold = load_tau(epsilon) if selection == "epsilon" else tau
    if threshold is None:
        raise RuntimeError("--tau is required when --selection=tau")
    return [row for row in rows if 1.0 - float(row["original_msp"]) > float(threshold)]


def default_output_tag(split: str, selection: str, epsilon: str, tau: float | None) -> str:
    if selection == "all":
        return "all"
    if selection == "epsilon":
        return f"e{epsilon.replace('.', '')}"
    if selection == "disagreement":
        assert tau is not None
        return f"d{tau:.6f}".replace(".", "p")
    assert tau is not None
    return f"tau{tau:.6f}".replace(".", "p")


def msp_touched_rows(epsilon: str) -> list[dict]:
    tau = load_tau(epsilon)
    rows = []
    for row in msp_rows("Test"):
        if 1.0 - float(row["original_msp"]) > tau:
            rows.append(row)
    return rows


def existing_done(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["index"]): row for row in read_csv(path)}


def output_path(split: str, mode: str, tag: str) -> Path:
    kind = "detector" if mode == "detector" else "final"
    return OUT_DIR / f"qwen_custom_{split.lower()}_v3_{kind}_{tag}.csv"


def load_detector_rows(split: str, detector_tag: str) -> dict[tuple[str, str], dict]:
    path = output_path(split, "detector", detector_tag)
    if not path.exists():
        raise RuntimeError(f"missing detector cache: {path}")
    return existing_done(path)


def selected_correction_rows(split: str, selection: str, epsilon: str, tau: float | None, detector_rows: dict[tuple[str, str], dict]) -> list[dict]:
    rows = msp_rows(split)
    if selection != "disagreement":
        selected = selected_msp_rows(split, selection, epsilon, tau)
        return [row for row in selected if (row["dataset"], row["index"]) in detector_rows]
    if tau is None:
        raise RuntimeError("--tau stores the calibrated d0 when --selection=disagreement")
    out = []
    for row in rows:
        detector = detector_rows.get((row["dataset"], row["index"]))
        if detector is None:
            continue
        if max(float(detector["T_top"]), float(detector["V_top"])) > float(tau):
            out.append(row)
    return out


def base_detector_row(
    labels: list[str],
    answer: str,
    s0: torch.Tensor,
    blank_scores: torch.Tensor,
    logits: dict[str, torch.Tensor],
    msp_row: dict,
    split: str,
) -> dict:
    base_score = centered(s0)
    base_pred = pred_label(base_score, labels)
    t_top = 1.0 - 0.5 * (top_retention(logits["orig"], logits["tcf1"]) + top_retention(logits["orig"], logits["tcf2"]))
    v_top = 0.5 * (top_retention(logits["orig"], logits["blank"]) + top_retention(logits["orig"], logits["noise"]))
    t_tau, v_tau = detector_thresholds()
    t_on = t_top >= t_tau
    v_on = v_top >= v_tau
    route = route_mode(t_on, v_on)
    return {
        "model": "qwen",
        "dataset": msp_row["dataset"],
        "split": f"Custom_{split}",
        "index": str(int(float(msp_row["index"]))),
        "answer": answer,
        "labels": labels_json(labels),
        "base_prediction": base_pred,
        "msp_prediction": msp_row.get("prediction", ""),
        "base_matches_msp": int(base_pred == msp_row.get("prediction", "")),
        "final_prediction": base_pred,
        "original_hit": float(base_pred == answer),
        "final_hit": float(base_pred == answer),
        "original_msp": msp_row.get("original_msp", ""),
        "orig_vocab_msp": maxp(logits["orig"]),
        "route_mode": route,
        "T_on": int(t_on),
        "V_on": int(v_on),
        "T_top": t_top,
        "V_top": v_top,
        "T_tau": t_tau,
        "V_tau": v_tau,
        "d_max": max(t_top, v_top),
        "vc_source": "",
        "image_info_conf_gray": "",
        "image_info_conf_strongblur": "",
        "image_info_conf_centermask": "",
        "tc_pred": "",
        "vc_pred": "",
        "verify_status": "detector_only",
        "has_detector": 1,
        "has_correction": 0,
        "s0": scores_json(s0, labels),
        "base_score": scores_json(base_score, labels),
        "blank_scores": scores_json(blank_scores, labels),
    }


def add_correction_fields(
    row: dict,
    labels: list[str],
    s0: torch.Tensor,
    blank_scores: torch.Tensor,
    logits: dict[str, torch.Tensor],
    label_ids: dict[str, list[int]],
) -> dict:
    answer_real = option_logprobs(logits["answer_real"], labels, label_ids)
    answer_blank = option_logprobs(logits["answer_blank"], labels, label_ids)
    channel_scores = {
        key: option_logprobs(logits[key], labels, label_ids)
        for key in ("gray", "strongblur", "centermask")
    }
    orig_vocab_msp = float(row["orig_vocab_msp"])
    channel_iic = {key: orig_vocab_msp - maxp(logits[key]) for key in channel_scores}
    vc_source = min(channel_iic, key=channel_iic.get)

    base_score = centered(s0)
    default_delta = s0 - blank_scores
    answer_delta = answer_real - answer_blank
    tc_delta = centered(torch.maximum(default_delta, answer_delta))
    vc_delta = centered(s0 - channel_scores[vc_source])
    final_scores, verify_status = grec_scores(base_score, tc_delta, vc_delta, row["route_mode"])
    base_pred = pred_label(base_score, labels)
    final_pred = pred_label(final_scores, labels) if row["route_mode"] != "neither" else base_pred
    answer = row["answer"]
    joint_delta = tc_delta + torch.softmax(tc_delta.float(), dim=0).to(vc_delta.dtype) * vc_delta

    out = dict(row)
    out.update(
        {
            "final_prediction": final_pred,
            "final_hit": float(final_pred == answer),
            "vc_source": vc_source,
            "image_info_conf_gray": channel_iic["gray"],
            "image_info_conf_strongblur": channel_iic["strongblur"],
            "image_info_conf_centermask": channel_iic["centermask"],
            "tc_pred": pred_label(tc_delta, labels),
            "vc_pred": pred_label(base_score + vc_delta, labels),
            "verify_status": verify_status,
            "has_correction": 1,
            "answer_real_scores": scores_json(answer_real, labels),
            "answer_blank_scores": scores_json(answer_blank, labels),
            "gray_scores": scores_json(channel_scores["gray"], labels),
            "strongblur_scores": scores_json(channel_scores["strongblur"], labels),
            "centermask_scores": scores_json(channel_scores["centermask"], labels),
            "tc_delta": scores_json(tc_delta, labels),
            "vc_delta": scores_json(vc_delta, labels),
            "joint_delta": scores_json(joint_delta, labels),
        }
    )
    return out


def evaluate_detector_row(model, processor, label_ids: dict[str, list[int]], raw: dict, msp_row: dict, split: str) -> dict | None:
    labels = option_labels(raw)
    answer = str(raw.get("answer", "")).strip().upper()[:1]
    if len(labels) < 2 or answer not in labels:
        return None
    image = load_image(raw)
    if image is None:
        return None
    seed = 42 + int(float(raw["index"]))

    prompts = {
        "default": build_prompt(raw, labels),
        "tcf1": prompt_tcf1(raw, labels),
        "tcf2": prompt_tcf2(raw, labels),
        "answer_format": prompt_answer_format(raw, labels),
    }
    logits = {
        "orig": first_logits(model, processor, transform_image(image, "default", seed), prompts["default"]),
        "tcf1": first_logits(model, processor, transform_image(image, "default", seed), prompts["tcf1"]),
        "tcf2": first_logits(model, processor, transform_image(image, "default", seed), prompts["tcf2"]),
        "blank": first_logits(model, processor, transform_image(image, "blank", seed), prompts["default"]),
        "noise": first_logits(model, processor, transform_image(image, "noise500", seed), prompts["default"]),
    }

    s0 = option_logprobs(logits["orig"], labels, label_ids)
    blank_scores = option_logprobs(logits["blank"], labels, label_ids)
    return base_detector_row(labels, answer, s0, blank_scores, logits, msp_row, split)


def evaluate_correction_row(
    model,
    processor,
    label_ids: dict[str, list[int]],
    raw: dict,
    detector_row: dict,
) -> dict | None:
    labels = json.loads(detector_row["labels"])
    image = load_image(raw)
    if image is None:
        return None
    seed = 42 + int(float(raw["index"]))
    prompts = {
        "default": build_prompt(raw, labels),
        "answer_format": prompt_answer_format(raw, labels),
    }
    logits = {
        "gray": first_logits(model, processor, transform_image(image, "gray", seed), prompts["default"]),
        "strongblur": first_logits(model, processor, transform_image(image, "strongblur", seed), prompts["default"]),
        "centermask": first_logits(model, processor, transform_image(image, "centermask", seed), prompts["default"]),
        "answer_real": first_logits(model, processor, transform_image(image, "default", seed), prompts["answer_format"]),
        "answer_blank": first_logits(model, processor, transform_image(image, "blank", seed), prompts["answer_format"]),
    }
    s0 = scores_from_json(detector_row["s0"], labels)
    blank_scores = scores_from_json(detector_row["blank_scores"], labels)
    return add_correction_fields(detector_row, labels, s0, blank_scores, logits, label_ids)


def evaluate_row(model, processor, label_ids: dict[str, list[int]], raw: dict, msp_row: dict, split: str) -> dict | None:
    detector = evaluate_detector_row(model, processor, label_ids, raw, msp_row, split)
    if detector is None:
        return None
    return evaluate_correction_row(model, processor, label_ids, raw, detector)


def summarize(split: str, tag: str, touched_rows: list[dict], final_rows: list[dict]) -> dict:
    source_rows = msp_rows(split)
    final_by_key = {(row["dataset"], row["index"]): row for row in final_rows}
    total_hit = 0.0
    touched = 0
    changed = 0
    clean_touched = 0
    clean_harmed = 0
    wrong_touched = 0
    wrong_fixed = 0
    for row in source_rows:
        key = (row["dataset"], row["index"])
        original_hit = float(row["original_hit"])
        if key not in final_by_key:
            total_hit += original_hit
            continue
        final = final_by_key[key]
        final_hit = float(final["final_hit"])
        total_hit += final_hit
        touched += 1
        changed += int(final["final_prediction"] != final["base_prediction"])
        clean_touched += int(original_hit == 1.0)
        clean_harmed += int(original_hit == 1.0 and final_hit == 0.0)
        wrong_touched += int(original_hit == 0.0)
        wrong_fixed += int(original_hit == 0.0 and final_hit == 1.0)
    n = len(source_rows)
    original_total = sum(float(row["original_hit"]) for row in source_rows)
    return {
        "model": "qwen",
        "split": f"Custom_{split}",
        "tag": tag,
        "n": n,
        "touched_requested": len(touched_rows),
        "touched_evaluated": touched,
        "original_accuracy": round(100.0 * original_total / n, 2),
        "final_accuracy": round(100.0 * total_hit / n, 2),
        "delta_accuracy": round(100.0 * (total_hit - original_total) / n, 2),
        "touched_rate": round(100.0 * touched / n, 2),
        "changed_rate_on_touched": round(100.0 * changed / touched, 2) if touched else 0.0,
        "clean_touched": clean_touched,
        "clean_harmed": clean_harmed,
        "clean_harm_rate_on_clean_touched": round(100.0 * clean_harmed / clean_touched, 2) if clean_touched else 0.0,
        "wrong_touched": wrong_touched,
        "wrong_fixed": wrong_fixed,
        "wrong_fix_rate_on_wrong_touched": round(100.0 * wrong_fixed / wrong_touched, 2) if wrong_touched else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["Val", "Test"], default="Test")
    parser.add_argument("--mode", choices=["full", "detector", "correction"], default="full")
    parser.add_argument("--selection", choices=["all", "epsilon", "tau", "disagreement"], default="epsilon")
    parser.add_argument("--epsilon", default="0.02")
    parser.add_argument("--tau", type=float, default=None, help="MSP tau for --selection=tau; d0 for --selection=disagreement")
    parser.add_argument("--detector-tag", default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    args = parser.parse_args()

    detector_rows = None
    if args.mode == "correction":
        if not args.detector_tag:
            raise RuntimeError("--detector-tag is required in correction mode")
        detector_rows = load_detector_rows(args.split, args.detector_tag)
        selected = selected_correction_rows(args.split, args.selection, args.epsilon, args.tau, detector_rows)
    else:
        if args.selection == "disagreement":
            raise RuntimeError("--selection=disagreement is only valid in correction mode")
        selected = selected_msp_rows(args.split, args.selection, args.epsilon, args.tau)
    if args.limit is not None:
        selected = selected[: args.limit]
    custom = load_custom_rows(args.split)
    tag = args.output_tag or default_output_tag(args.split, args.selection, args.epsilon, args.tau)
    out_path = output_path(args.split, args.mode, tag)
    done = existing_done(out_path)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    label_ids = {label: letter_token_ids(processor.tokenizer, label) for label in "ABCD"}

    rows = dict(done)
    newly = 0
    for pos, msp_row in enumerate(selected, start=1):
        key = (msp_row["dataset"], msp_row["index"])
        if key in rows:
            continue
        raw = custom.get(key)
        if raw is None:
            continue
        if args.mode == "detector":
            result = evaluate_detector_row(model, processor, label_ids, raw, msp_row, args.split)
        elif args.mode == "correction":
            assert detector_rows is not None
            detector = detector_rows.get(key)
            if detector is None:
                continue
            result = evaluate_correction_row(model, processor, label_ids, raw, detector)
        else:
            result = evaluate_row(model, processor, label_ids, raw, msp_row, args.split)
        if result is None:
            continue
        rows[key] = result
        newly += 1
        if newly % args.checkpoint_interval == 0:
            write_csv(out_path, list(rows.values()))
            print(f"{args.split}:{tag}: pos={pos}/{len(selected)} total={len(rows)} newly={newly}", flush=True)

    final_rows = list(rows.values())
    write_csv(out_path, final_rows)
    summary = (
        {
            "model": "qwen",
            "split": f"Custom_{args.split}",
            "tag": tag,
            "mode": args.mode,
            "n_selected": len(selected),
            "n_cached": len(final_rows),
            "newly": newly,
        }
        if args.mode == "detector"
        else summarize(args.split, tag, selected, final_rows)
    )
    summary["mode"] = args.mode
    summary_path = OUT_DIR / f"natural_qwen_{args.split.lower()}_{args.mode}_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
