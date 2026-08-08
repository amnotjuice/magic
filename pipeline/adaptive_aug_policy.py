"""CDA-v2 TC Adaptive-Aug policy summary.

This file keeps the TC-side v2 design separate from older CDA-v1 router
experiments.  It does not run VLM inference; it consumes the cached probe
outputs and reports a clean module-level view:

    Detection-v2 gate: retain_top detector config is recorded, not applied here.
    TC MCQ action: clean2 prompt-conditioned visual-lift.
    TC Oth action: candidate-level visual-lift with optional margin fallback.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
LOCAL_LOG_ROOT = ROOT / "pipeline/data"
ARCHIVE_LOG_ROOT = Path(
    "/data/shunshungu/Self-Critical-Inference-Framework__archive_preupload_20260618T033451Z/research/log"
)
LOG_ROOT = Path(os.environ.get("MAGIC_PIPELINE_DATA_ROOT", LOCAL_LOG_ROOT))
if not (LOG_ROOT / "exp07_clean_main_cached/summary.json").exists() and ARCHIVE_LOG_ROOT.exists():
    LOG_ROOT = ARCHIVE_LOG_ROOT

OUT_DIR = ROOT / "pipeline/data/adaptive_aug_policy"
TC_LOG = ROOT / "pipeline/data"

PROMPT_POOLS = {
    "clean1": ("default",),
    "clean2": ("default", "answer_format"),
    "clean3": ("default", "answer_format", "paraphrase"),
}
MAIN_PROMPT_POOL = "clean2"
MAIN_MCq_VARIANT = "clean2_mcq"

GROUPS = {
    "B": {"vcf_only", "both"},
    "S": {"tcf_only", "both"},
    "BS": {"vcf_only", "tcf_only", "both"},
}

PAPER = {
    "qwen": {
        "SCI5": {
            "B_MCQ": 24.91,
            "B_Oth": 25.69,
            "B_All": 25.09,
            "S_MCQ": 47.22,
            "S_Oth": 42.44,
            "S_All": 44.58,
            "BS_MCQ": 28.00,
            "BS_Oth": 33.14,
            "BS_All": 29.50,
        },
        "SCI7": {
            "B_MCQ": 27.04,
            "B_Oth": 29.66,
            "B_All": 27.65,
            "S_MCQ": 47.22,
            "S_Oth": 45.98,
            "S_All": 46.54,
            "BS_MCQ": 29.61,
            "BS_Oth": 36.84,
            "BS_All": 31.72,
        },
    },
    "llava": {
        "SCI5": {
            "B_MCQ": 23.81,
            "B_Oth": 37.97,
            "B_All": 26.08,
            "S_MCQ": 40.60,
            "S_Oth": 60.65,
            "S_All": 47.95,
            "BS_MCQ": 28.80,
            "BS_Oth": 51.01,
            "BS_All": 34.19,
        },
        "SCI7": {
            "B_MCQ": 24.86,
            "B_Oth": 38.26,
            "B_All": 27.01,
            "S_MCQ": 40.10,
            "S_Oth": 60.65,
            "S_All": 47.64,
            "BS_MCQ": 29.68,
            "BS_Oth": 51.26,
            "BS_All": 34.92,
        },
    },
}

MCQ_SOURCES = {
    "qwen": {
        "path": TC_LOG / "tc_evidence_delta_probe/summary.json",
        "keys": {
            "clean1_mcq": "default1_delta_only_max",
            "clean2_mcq": "clean2_answer_delta_only_max",
            "clean3_mcq": "clean3_delta_only_max",
        },
    },
    "llava": {
        "path": TC_LOG / "tc_lift_llava_probe/summary.json",
        "keys": {
            "clean1_mcq": "default1_delta_max",
            "clean2_mcq": "clean2_answer_delta_max",
            "clean3_mcq": "clean3_delta_max",
        },
    },
}

OPEN_SOURCES = {
    "qwen": {
        "routed": {
            "Val": LOG_ROOT / "exp07_clean_joint_router_fullgen/qwen_val_routed_rows.csv",
            "Test": LOG_ROOT / "exp07_clean_joint_router_fullgen/qwen_test_routed_rows.csv",
        },
        "datasets": {
            "Val": {
                "MME": TC_LOG / "qwen_mme_inference/MME_Qwen2-VL-7B_Biased_Val_v4_n68_seed0.jsonl",
            },
            "Test": {
                "MME": TC_LOG / "qwen_mme_inference/MME_Qwen2-VL-7B_Biased_Test_v4_n222_seed0.jsonl",
                "ViLP": TC_LOG / "qwen_mme_inference/ViLP_Qwen2-VL-7B_Biased_Test_v4_n290_seed0.jsonl",
            },
        },
    },
    "llava": {
        "routed": {
            "Val": LOG_ROOT / "exp07_clean_joint_router_fullgen/llava_val_routed_rows.csv",
            "Test": LOG_ROOT / "exp07_clean_joint_router_fullgen/llava_test_routed_rows.csv",
        },
        "datasets": {
            "Val": {
                "MME": TC_LOG / "llava_mme_inference/MME_LLaVA-NeXT-8B_Biased_Val_n80_seed0.jsonl",
                "ViLP": TC_LOG
                / "llava_vilp_inference/ViLP_LLaVA-NeXT-8B_Biased_Val_n68_seed0_beam5_noverify_score-default.jsonl",
            },
            "Test": {
                "MME": TC_LOG / "llava_mme_inference/MME_LLaVA-NeXT-8B_Biased_Test_n513_seed0.jsonl",
                "ViLP": TC_LOG
                / "llava_vilp_inference/ViLP_LLaVA-NeXT-8B_Biased_Test_n283_seed0_beam5_noverify_score-default.jsonl",
            },
        },
    },
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[str(rec["key"])] = rec
    return out


def cached_summary() -> dict:
    return json.loads((LOG_ROOT / "exp07_clean_main_cached/summary.json").read_text(encoding="utf-8"))


def detection_v2_config() -> dict:
    path = TC_LOG / "retention_detector/summary.json"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    summary = json.loads(path.read_text(encoding="utf-8"))
    out = {
        "status": "recorded_not_applied_to_module_cache",
        "detector": "retention_top",
        "meaning": "gate TC action when predicted mode is tcf_only or both",
        "path": str(path),
        "models": {},
    }
    for model_key, block in summary.items():
        result = block["evaluations"]["retention_top"]
        out["models"][model_key] = {
            "features": result["features"],
            "thresholds": result["thresholds"],
            "test_mode_acc": result["test"]["mode"]["accuracy"],
            "test_macro_f1_3way": result["test"]["mode"]["macro_f1_3way"],
        }
    return out


def mcq_metrics(model_key: str) -> dict[str, dict[str, float]]:
    source = MCQ_SOURCES[model_key]
    block = json.loads(source["path"].read_text(encoding="utf-8"))["Test"]["metrics"]
    out = {}
    for variant, metric_key in source["keys"].items():
        out[variant] = {
            f"{group}_MCQ": float(block[metric_key][group])
            for group in ("B", "S", "BS")
        }
    return out


def flatten_clean2_lifts(rec: dict) -> dict[str, float]:
    scores = rec.get("lift_scores", {})
    if not scores:
        return {}
    if all(isinstance(value, (int, float)) for value in scores.values()):
        return {str(label): float(value) for label, value in scores.items()}
    labels = set()
    for prompt in PROMPT_POOLS[MAIN_PROMPT_POOL]:
        labels.update(scores.get(prompt, {}).keys())
    return {
        label: max(float(scores[prompt][label]) for prompt in PROMPT_POOLS[MAIN_PROMPT_POOL] if label in scores.get(prompt, {}))
        for label in labels
    }


def top_margin(scores: dict[str, float]) -> float:
    values = sorted((float(value) for value in scores.values() if math.isfinite(float(value))), reverse=True)
    if len(values) < 2:
        return 0.0
    return values[0] - values[1]


def hit_from_binary_choice(answer: str, label: str) -> float:
    pred = "Yes" if label == "A" else "No"
    return float(str(answer).strip().lower() == pred.lower())


def choose_label(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get) if scores else ""


def qwen_open_policy(rec: dict, dataset: str) -> dict:
    baseline_key = "answer_format_hit" if dataset == "MME" else "default_hit"
    scores = rec.get("direct_lift_scores", {})
    return {
        "baseline_hit": float(rec.get(baseline_key, 0.0) or 0.0),
        "lift_hit": float(rec.get("direct_lift_hit", 0.0) or 0.0),
        "margin": top_margin(scores),
        "action": "direct_answer_token_visual_lift",
    }


def llava_open_policy(rec: dict, dataset: str) -> dict:
    if dataset == "MME":
        scores = flatten_clean2_lifts(rec)
        label = choose_label(scores)
        lift_hit = hit_from_binary_choice(str(rec.get("answer", "")), label) if label else 0.0
        return {
            "baseline_hit": float(rec.get("answer_format_real_hit", 0.0) or 0.0),
            "lift_hit": lift_hit,
            "margin": top_margin(scores),
            "action": "binary_clean2_visual_lift",
        }
    scores = rec.get("direct_lift_scores", {})
    return {
        "baseline_hit": float(rec.get("default_hit", 0.0) or 0.0),
        "lift_hit": float(rec.get("direct_lift_hit", 0.0) or 0.0),
        "margin": top_margin(scores),
        "action": "beam_candidate_direct_visual_lift",
    }


OPEN_POLICY: dict[str, Callable[[dict, str], dict]] = {
    "qwen": qwen_open_policy,
    "llava": llava_open_policy,
}


def open_rows(model_key: str, split: str, require_complete: bool = True) -> list[dict]:
    source = OPEN_SOURCES[model_key]
    routed = read_csv(source["routed"][split])
    caches = {dataset: read_jsonl(path) for dataset, path in source["datasets"][split].items()}
    out = []
    missing = []
    for row in routed:
        dataset = row["dataset"]
        if dataset not in caches:
            continue
        key = str(int(float(row["index"])))
        rec = caches[dataset].get(key)
        if rec is None:
            missing.append((dataset, key))
            continue
        policy = OPEN_POLICY[model_key](rec, dataset)
        item = dict(row)
        item.update(policy)
        item["dataset"] = dataset
        item["key"] = key
        out.append(item)
    if missing and require_complete:
        raise RuntimeError(f"{model_key} {split} missing {len(missing)} Oth records, e.g. {missing[:5]}")
    return out


def group_values(rows: list[dict], value_fn: Callable[[dict], float]) -> dict[str, dict[str, float]]:
    out = {}
    for group, modes in GROUPS.items():
        sub = [row for row in rows if row["mode"] in modes]
        hit = sum(value_fn(row) for row in sub)
        out[f"{group}_Oth"] = {
            "n": len(sub),
            "hit": hit,
            "accuracy": hit / len(sub) if sub else 0.0,
        }
    return out


def fit_margin_tau(rows: list[dict], target_group: str = "BS") -> dict:
    modes = GROUPS[target_group]
    candidates = sorted({float(row["margin"]) for row in rows if row["mode"] in modes})
    if not candidates:
        return {"tau": float("-inf"), "val_acc": 0.0, "target_group": target_group, "enabled": False}
    candidates = [candidates[0] - 1e-12, *candidates, candidates[-1] + 1e-12]
    best = None
    for tau in candidates:
        selected = [row for row in rows if row["mode"] in modes]
        values = [
            float(row["lift_hit"]) if float(row["margin"]) >= tau else float(row["baseline_hit"])
            for row in selected
        ]
        acc = sum(values) / len(values) if values else 0.0
        cand = (acc, -tau)
        if best is None or cand > best[0]:
            best = (cand, tau, acc)
    assert best is not None
    return {"tau": float(best[1]), "val_acc": float(best[2]), "target_group": target_group, "enabled": True}


def margin_fallback_value(row: dict, tau_by_dataset: dict[str, dict]) -> float:
    fit = tau_by_dataset.get(row["dataset"])
    if not fit or not fit.get("enabled"):
        return float(row["lift_hit"])
    return float(row["lift_hit"]) if float(row["margin"]) >= float(fit["tau"]) else float(row["baseline_hit"])


def open_metrics(model_key: str) -> dict:
    test = open_rows(model_key, "Test")
    val_by_dataset = {}
    try:
        val = open_rows(model_key, "Val", require_complete=False)
    except FileNotFoundError:
        val = []
    for dataset in sorted({row["dataset"] for row in test}):
        val_rows = [row for row in val if row["dataset"] == dataset]
        val_by_dataset[dataset] = fit_margin_tau(val_rows) if val_rows else {
            "tau": None,
            "val_acc": None,
            "target_group": "BS",
            "enabled": False,
            "reason": "no matching Val cache",
        }

    return {
        "counts": {key: value["n"] for key, value in group_values(test, lambda row: float(row["lift_hit"])).items()},
        "baseline": group_values(test, lambda row: float(row["baseline_hit"])),
        "fixed_lift": group_values(test, lambda row: float(row["lift_hit"])),
        "margin_fallback": group_values(test, lambda row: margin_fallback_value(row, val_by_dataset)),
        "margin_fit": val_by_dataset,
        "actions": sorted({row["action"] for row in test}),
    }


def combine(cached: dict, model_key: str, version: str, mcq: dict[str, float], oth: dict[str, dict[str, float]]) -> dict:
    row = {"model": model_key, "version": version}
    for group in ("B", "S", "BS"):
        mcq_key = f"{group}_MCQ"
        oth_key = f"{group}_Oth"
        all_key = f"{group}_All"
        mcq_n = int(cached[model_key]["test"]["paper_metrics"][mcq_key]["n"])
        oth_n = int(oth[oth_key]["n"])
        mcq_acc = mcq[mcq_key]
        oth_acc = oth[oth_key]["accuracy"]
        all_acc = (mcq_acc * mcq_n + oth_acc * oth_n) / (mcq_n + oth_n)
        row[mcq_key] = round(mcq_acc * 100.0, 2)
        row[oth_key] = round(oth_acc * 100.0, 2)
        row[all_key] = round(all_acc * 100.0, 2)
    return row


def paper_row(model_key: str, version: str) -> dict:
    row = {"model": model_key, "version": version}
    row.update(PAPER[model_key][version])
    row["note"] = "paper"
    return row


def pct_block(block: dict[str, dict[str, float]]) -> dict[str, float]:
    return {key: round(value["accuracy"] * 100.0, 2) for key, value in block.items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = cached_summary()
    rows = []
    details = {
        "name": "CDA-v2 / Module 2: TC Adaptive-Aug-v2",
        "status": "module_level_cache_summary",
        "prompt_pools": PROMPT_POOLS,
        "main_prompt_pool": MAIN_PROMPT_POOL,
        "detection_v2_gate": detection_v2_config(),
        "cost_model": {
            "MCQ_clean2": "original real + answer_format real + matched blank/prior scores from cache; final online accounting should reuse Detection-v2 probes where possible",
            "Oth_candidate_lift": "candidate proposal + one real scoring pass + one blank scoring pass when implemented with vectorized candidate scoring",
            "margin_fallback": "no extra model forward; uses top1-top2 visual-lift margin",
        },
        "models": {},
    }

    for model_key in ("qwen", "llava"):
        rows.append(paper_row(model_key, "SCI5"))
        rows.append(paper_row(model_key, "SCI7"))
        mcq_by_variant = mcq_metrics(model_key)
        open_eval = open_metrics(model_key)

        main_mcq = mcq_by_variant[MAIN_MCq_VARIANT]
        fixed = combine(cached, model_key, "tc_aug_v2_clean2_fixed_lift", main_mcq, open_eval["fixed_lift"])
        fixed["note"] = "module-level; Detection-v2 not applied"
        fallback = combine(cached, model_key, "tc_aug_v2_clean2_margin_fallback", main_mcq, open_eval["margin_fallback"])
        fallback["note"] = "optional; Val-fit Oth fallback only"
        baseline = combine(cached, model_key, "tc_aug_v2_clean2_oth_baseline", main_mcq, open_eval["baseline"])
        baseline["note"] = "ablation"
        rows.extend([baseline, fixed, fallback])

        details["models"][model_key] = {
            "mcq_variants": {
                variant: {key: round(value * 100.0, 2) for key, value in metrics.items()}
                for variant, metrics in mcq_by_variant.items()
            },
            "oth_counts": open_eval["counts"],
            "oth_baseline": pct_block(open_eval["baseline"]),
            "oth_fixed_lift": pct_block(open_eval["fixed_lift"]),
            "oth_margin_fallback": pct_block(open_eval["margin_fallback"]),
            "oth_margin_fit": open_eval["margin_fit"],
            "oth_actions": open_eval["actions"],
        }

    write_csv(OUT_DIR / "comparison_9cell.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
