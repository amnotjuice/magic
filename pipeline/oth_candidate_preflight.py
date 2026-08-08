"""Oth-side preflight for the current clean2/admissible-TC candidate.

Pre-experiment gate:

Fixed candidate:
    Oth TC analogue:
        S0(y) = candidate real-image score
        clean2(y) = candidate visual-lift score from the v2 Oth cache
        alpha_T = 1[margin(clean2) + source_agree >= 0.070279]
        S_T = clean2 if alpha_T else S0

    Oth VC analogue target:
        Delta_V(y) = S0(y) - V_c*(y), with c* selected per sample.
        This script does not estimate alpha_V because Val-side route-free Oth
        VC residual scores are not yet materialized.

Elegance check:
    One TC operator, one fixed TC admissibility signal, no route detector, no
    dataset construction rule.  This is a Val-only Oth feasibility preflight,
    not the main method.

Modules:
    TC = Oth candidate-level clean2 visual lift
    VC = coverage audit only in this script
    adaptivity = per-sample alpha_T from clean2 margin + source agreement
    aggregation = choose S_T per sample
    active cost = Oth candidate generation + clean2 real/blank candidate scoring

Calibration:
    The TC threshold is frozen from the current MCQ candidate.  No Test result
    is used for parameter selection here.

Acceptance:
    This run cannot satisfy the full CDA standard by itself because VC,
    natural, All-table, and route-free Oth Test verification remain pending.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT.parent / "Self-Critical-Inference-Framework__archive_preupload_20260618T033451Z"
ARCHIVE_LOG_ROOT = ARCHIVE_ROOT / "pipeline/data"
ARCHIVE_RUNTIME_ROOT = ARCHIVE_ROOT / "ignored_runtime"
if "MAGIC_PIPELINE_DATA_ROOT" not in os.environ and ARCHIVE_LOG_ROOT.exists():
    os.environ["MAGIC_PIPELINE_DATA_ROOT"] = str(ARCHIVE_LOG_ROOT)
if "MAGIC_PIPELINE_RUNTIME_ROOT" not in os.environ and ARCHIVE_RUNTIME_ROOT.exists():
    os.environ["MAGIC_PIPELINE_RUNTIME_ROOT"] = str(ARCHIVE_RUNTIME_ROOT)

import legacy_scoring_framework as F  # noqa: E402
import adaptive_aug_policy as TC_AUG  # noqa: E402
from open_ended_scoring import paper_hit  # noqa: E402


OUT_DIR = ROOT / "pipeline/data/oth_candidate_preflight"
OTH_VC_DIR = ROOT / "pipeline/data/oth_grec_cache"
TC_TAU = 0.070279
GROUPS = {
    "B_Oth": {"vcf_only", "both"},
    "S_Oth": {"tcf_only", "both"},
    "BS_Oth": {"vcf_only", "tcf_only", "both"},
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_key(row: dict) -> tuple[str, str]:
    return str(row["dataset"]), str(int(float(row["index"])))


def load_tc_logs(model_key: str, split: str) -> dict[tuple[str, str], dict]:
    out = {}
    for dataset, path in TC_AUG.OPEN_SOURCES[model_key]["datasets"][split].items():
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["dataset"] = dataset
                out[(dataset, str(rec["key"]))] = rec
    return out


def choose(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get) if scores else ""


def finite_scores(scores: dict) -> dict[str, float]:
    return {
        str(label): float(value)
        for label, value in scores.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def flatten_clean2_lifts(rec: dict) -> dict[str, float]:
    return TC_AUG.flatten_clean2_lifts(rec)


def base_and_tc_scores(model_key: str, dataset: str, rec: dict) -> tuple[dict[str, float], dict[str, float]]:
    if model_key == "llava" and dataset == "MME":
        real = {}
        lift = flatten_clean2_lifts(rec)
        for label in ("A", "B"):
            values = [
                float(scores[label])
                for name, scores in rec.get("real_scores", {}).items()
                if name in TC_AUG.PROMPT_POOLS[TC_AUG.MAIN_PROMPT_POOL] and label in scores
            ]
            if values:
                real["Yes" if label == "A" else "No"] = max(values)
        lift_yn = {("Yes" if label == "A" else "No"): score for label, score in lift.items()}
        return real, lift_yn
    return finite_scores(rec.get("direct_real_scores", {})), finite_scores(rec.get("direct_lift_scores", {}))


def top_margin(scores: dict[str, float]) -> float:
    values = sorted(
        [float(value) for value in scores.values() if math.isfinite(float(value))],
        reverse=True,
    )
    if len(values) < 2:
        return 0.0
    return values[0] - values[1]


def normalize_text(text: object) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,:;!?\"'")


def source_agreement(rec: dict) -> int:
    lift_scores = rec.get("lift_scores", {})
    if isinstance(lift_scores, dict):
        default_scores = lift_scores.get("default")
        answer_scores = lift_scores.get("answer_format")
        if isinstance(default_scores, dict) and isinstance(answer_scores, dict):
            return int(choose(default_scores) == choose(answer_scores))

    generated = rec.get("generated", {})
    if isinstance(generated, dict) and "default" in generated and "answer_format" in generated:
        return int(normalize_text(generated.get("default")) == normalize_text(generated.get("answer_format")))

    return 0


def mode_groups(mode: str) -> list[str]:
    return [group for group, modes in GROUPS.items() if mode in modes]


def metrics(model_key: str, rows: list[dict], variant: str) -> dict:
    out = {"model": model_key, "variant": variant}
    wins = 0
    for group, modes in GROUPS.items():
        selected = [row for row in rows if row["mode"] in modes]
        hits = sum(float(row[f"{variant}_hit"]) for row in selected)
        acc = 100.0 * hits / len(selected) if selected else 0.0
        sci7 = TC_AUG.PAPER[model_key]["SCI7"][group]
        out[group] = round(acc, 2)
        out[f"n_{group}"] = len(selected)
        out[f"hits_{group}"] = round(hits, 6)
        out[f"vs7_{group}"] = round(acc - sci7, 2)
        wins += int(acc > sci7)
    out["wins_vs_sci7_oth"] = wins
    return out


def cache_audit(model_key: str, split: str, open_count: int) -> dict:
    path = OTH_VC_DIR / f"{model_key}_{split.lower()}_oth_vc_scores.jsonl"
    rows = read_jsonl(path)
    datasets = Counter(str(row.get("dataset", "")) for row in rows)
    return {
        "path": str(path),
        "exists": path.exists(),
        "records": len(rows),
        "open_rows": open_count,
        "dataset_counts": dict(sorted(datasets.items())),
        "route_free_ready": path.exists() and len(rows) >= open_count,
        "note": (
            "route-free Oth VC requires candidate-level VC scores for every row "
            "before alpha_V can be selected on Val"
        ),
    }


def build_rows(model_key: str, split: str) -> tuple[list[dict], dict]:
    open_rows = F.load_open_rows(model_key, split)
    tc_logs = load_tc_logs(model_key, split)
    selected = []
    skipped = Counter()
    dataset_counts = Counter()
    source_agree_counts = Counter()
    alpha_counts = Counter()

    for row in open_rows:
        key = row_key(row)
        tc_rec = tc_logs.get(key)
        if tc_rec is None:
            skipped[f"missing_tc_{key[0]}"] += 1
            continue
        base_scores, tc_scores = base_and_tc_scores(model_key, key[0], tc_rec)
        if not base_scores or not tc_scores:
            skipped[f"empty_scores_{key[0]}"] += 1
            continue

        answer = str(tc_rec.get("answer", row.get("answer", "")))
        base_pred = choose(base_scores)
        tc_pred = choose(tc_scores)
        agree = source_agreement(tc_rec)
        tc_conf = top_margin(tc_scores) + float(agree)
        alpha_t = int(tc_conf >= TC_TAU)
        final_pred = tc_pred if alpha_t else base_pred
        mode = row["mode"] if row["mode"] in {"vcf_only", "tcf_only", "both"} else "neither"
        selected.append(
            {
                "model": model_key,
                "split": split,
                "dataset": key[0],
                "index": key[1],
                "mode": mode,
                "answer": answer,
                "base_pred": base_pred,
                "tc_pred": tc_pred,
                "final_pred": final_pred,
                "tc_margin": top_margin(tc_scores),
                "source_agree": agree,
                "tc_confidence": tc_conf,
                "alpha_T": alpha_t,
                "baseline_hit": paper_hit(answer, base_pred),
                "tc_always_hit": paper_hit(answer, tc_pred),
                "tc_admissible_hit": paper_hit(answer, final_pred),
                "groups": ";".join(mode_groups(mode)),
            }
        )
        dataset_counts[key[0]] += 1
        source_agree_counts[agree] += 1
        alpha_counts[alpha_t] += 1

    audit = {
        "open_rows": len(open_rows),
        "rows_with_tc": len(selected),
        "skipped": dict(sorted(skipped.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "source_agree_counts": {str(k): v for k, v in sorted(source_agree_counts.items())},
        "alpha_T_counts": {str(k): v for k, v in sorted(alpha_counts.items())},
        "vc_cache": cache_audit(model_key, split, len(open_rows)),
    }
    return selected, audit


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = []
    summary = {
        "name": "Current clean2/admissible-TC candidate Oth preflight",
        "status": "Val-only TC analogue plus VC cache audit; not a main-method result",
        "standard_check": {
            "allowed_as_main": False,
            "reason": "Oth VC residual, natural, frozen Test protocol, and full SCI table comparison are incomplete.",
        },
        "fixed_protocol": {
            "tc_tau": TC_TAU,
            "tc": "alpha_T=1[margin(clean2 candidate visual-lift)+source_agree >= tc_tau]",
            "vc": "not evaluated; route-free Val residual cache is audited",
            "detector": "none",
        },
        "models": {},
    }

    for model_key in ("qwen", "llava"):
        rows, audit = build_rows(model_key, "Val")
        model_rows = [
            metrics(model_key, rows, "baseline"),
            metrics(model_key, rows, "tc_always"),
            metrics(model_key, rows, "tc_admissible"),
        ]
        for row in model_rows:
            row["split"] = "Val"
        comparison.extend(model_rows)
        write_csv(OUT_DIR / f"{model_key}_val_rows.csv", rows)
        summary["models"][model_key] = {
            "val_audit": audit,
            "comparison": model_rows,
        }

    write_csv(OUT_DIR / "comparison_val_oth.csv", comparison)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
