from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_LOG_ROOT = Path(os.environ.get("MAGIC_PIPELINE_DATA_ROOT", ROOT / "pipeline/data"))

from action_grammar import (  # noqa: E402
    CLEAN_FALLBACK,
    EXECUTABLE_ACTION_ALIASES,
    LEGACY_FALLBACK,
    main_method_slot_actions,
    public_action_name,
    readable_slot_name,
)


CELLS = ["B_MCQ", "B_Oth", "B_All", "S_MCQ", "S_Oth", "S_All", "BS_MCQ", "BS_Oth", "BS_All"]
OUT_DIR = ROOT / "pipeline/data/legacy_cache"

MODEL_CONFIGS = {
    "qwen": {
        "model": "Qwen2-VL-7B",
        "val_table": RESEARCH_LOG_ROOT / "exp07_floor_sci5_ours_aug/val_action_table.csv",
        "test_table": RESEARCH_LOG_ROOT / "exp07_floor_sci5_ours_aug/test_action_table.csv",
        "paper_sci5": {
            "B_MCQ": 0.2491, "B_Oth": 0.2569, "B_All": 0.2509,
            "S_MCQ": 0.4722, "S_Oth": 0.4244, "S_All": 0.4458,
            "BS_MCQ": 0.2800, "BS_Oth": 0.3314, "BS_All": 0.2950,
        },
        "paper_sci7": {
            "B_MCQ": 0.2704, "B_Oth": 0.2966, "B_All": 0.2765,
            "S_MCQ": 0.4722, "S_Oth": 0.4598, "S_All": 0.4654,
            "BS_MCQ": 0.2961, "BS_Oth": 0.3684, "BS_All": 0.3172,
        },
    },
    "llava": {
        "model": "LLaVA-NeXT-8B",
        "val_table": RESEARCH_LOG_ROOT / "exp07_llava_floor_sci5_ours_aug/val_action_table.csv",
        "test_table": RESEARCH_LOG_ROOT / "exp07_llava_floor_sci5_ours_aug/test_action_table.csv",
        "paper_sci5": {
            "B_MCQ": 0.2381, "B_Oth": 0.3797, "B_All": 0.2608,
            "S_MCQ": 0.4060, "S_Oth": 0.6065, "S_All": 0.4795,
            "BS_MCQ": 0.2880, "BS_Oth": 0.5101, "BS_All": 0.3419,
        },
        "paper_sci7": {
            "B_MCQ": 0.2486, "B_Oth": 0.3826, "B_All": 0.2701,
            "S_MCQ": 0.4010, "S_Oth": 0.6065, "S_All": 0.4764,
            "BS_MCQ": 0.2968, "BS_Oth": 0.5126, "BS_All": 0.3492,
        },
    },
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def mean(values) -> float:
    vals = [value for value in values if not math.isnan(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def is_mcq(row: dict) -> bool:
    return bool(int(float(row.get("clean_is_mcq", "0"))))


def fmt(row: dict) -> str:
    return "mcq" if is_mcq(row) else "oth"


def finite_values(rows: list[dict], feature: str) -> list[float]:
    return sorted({as_float(row, feature) for row in rows if not math.isnan(as_float(row, feature))})


def thresholds(rows: list[dict], feature: str, steps: int) -> list[float]:
    values = finite_values(rows, feature)
    if not values:
        return [0.0]
    if len(values) <= steps:
        return values
    return [values[round(i * (len(values) - 1) / steps)] for i in range(1, steps)]


def slot_key(row: dict, slot: str) -> str:
    return f"{slot}:{fmt(row)}"


def route(row: dict, params: dict) -> tuple[str, dict]:
    if is_mcq(row):
        tc_score = as_float(row, "tc_os_semantic_flip_count")
        tc_need = tc_score > float(params["mcq_tc_threshold"])
    else:
        tc_score = as_float(row, "d_tc_js")
        tc_need = tc_score > float(params["oth_tc_threshold"])

    vc_score = as_float(row, "vc_same_count_blur")
    vc_need = vc_score > float(params["vc_threshold"])
    qmask_score = as_float(row, "d_tc_qmask_js")
    qmask_high = qmask_score > float(params["qmask_threshold"])
    uncertainty = as_float(row, "uncertainty_score")
    reliable = uncertainty <= float(params["uncertainty_threshold"])

    # Fixed clean router: unreliable samples use the conservative strategy;
    # simultaneous visual and text evidence routes to Joint before single-axis
    # Text or Visual corrections.
    if not reliable:
        slot = "floor"
    elif tc_need and vc_need:
        slot = "both"
    elif is_mcq(row):
        if tc_need:
            slot = "tc"
        elif vc_need:
            slot = "vc"
        else:
            slot = "floor"
    elif qmask_high:
        slot = "tc_high"
    elif vc_need:
        slot = "vc"
    else:
        slot = "tc_low"

    return slot, {
        "tc_score": tc_score,
        "tc_need": int(tc_need),
        "vc_score": vc_score,
        "vc_need": int(vc_need),
        "qmask_score": qmask_score,
        "qmask_high": int(qmask_high),
        "uncertainty_score": uncertainty,
        "reliable": int(reliable),
    }


def clean_slot_actions(model_key: str) -> dict[str, list[str]]:
    actions = main_method_slot_actions()
    # The cached audit uses legacy executable columns for some clean public
    # actions. Full-generation open-ended Joint is scored separately because
    # LLaVA's cached table does not contain a QuestionMask+VC joint column.
    executable = {
        key: [EXECUTABLE_ACTION_ALIASES.get(action, action) for action in value]
        for key, value in actions.items()
    }
    if model_key == "qwen":
        executable["both:mcq"] = ["b5_joint_strong_os1", "b5_joint_center_os1"]
        executable["both:oth"] = ["tcg_both_strong_qmask_b2_t0.05", "tcg_both_center_qmask_b2_t0.05"]
    elif model_key == "llava":
        executable["both:mcq"] = [
            "joint_os_strong_blur_b0.2_g2.5_t0.2",
            "joint_os_center_mask_b0.2_g2.5_t0.2",
        ]
        executable["both:oth"] = [LEGACY_FALLBACK]
    return executable


def public_action(action: str) -> str:
    if action == LEGACY_FALLBACK:
        return CLEAN_FALLBACK
    return action


def available(rows: list[dict], actions: list[str]) -> list[str]:
    cols = set(rows[0]) if rows else set()
    return [action for action in actions if action in cols]


def action_mean(rows: list[dict], action: str) -> float:
    return mean(as_float(row, action) for row in rows)


def fit_slot_actions(rows: list[dict], params: dict, candidates: dict[str, list[str]]) -> dict[str, str]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        slot, _ = route(row, params)
        groups.setdefault(slot_key(row, slot), []).append(row)

    selected = {}
    for key, actions in candidates.items():
        usable = available(rows, actions)
        if not usable:
            raise KeyError(f"no usable actions for {key}: {actions}")
        if key.startswith("floor:"):
            selected[key] = LEGACY_FALLBACK
            continue
        group = groups.get(key, rows)
        selected[key] = max(usable, key=lambda action: action_mean(group, action))
    return selected


def apply_policy(rows: list[dict], params: dict, slot_actions: dict[str, str]) -> tuple[list[dict], list[float]]:
    selected = []
    hits = []
    for row in rows:
        slot, evidence = route(row, params)
        action = slot_actions[slot_key(row, slot)]
        hit = as_float(row, action)
        out = {
            "dataset": row["dataset"],
            "split_name": row["split_name"],
            "local_idx": row["local_idx"],
            "index": row["index"],
            "mode": row["mode"],
            "format": fmt(row),
            "slot_id": slot,
            "slot": readable_slot_name(slot),
            "action_id": public_action(action),
            "action": public_action_name(action),
            "hit": hit,
            **evidence,
        }
        selected.append(out)
        hits.append(hit)
    return selected, hits


def subset_indices(rows: list[dict], group: str, format_name: str) -> list[int]:
    modes = {
        "B": {"vcf_only", "both"},
        "S": {"tcf_only", "both"},
        "BS": {"vcf_only", "tcf_only", "both"},
    }[group]
    idxs = [idx for idx, row in enumerate(rows) if row["mode"] in modes]
    if format_name == "MCQ":
        idxs = [idx for idx in idxs if is_mcq(rows[idx])]
    elif format_name == "Oth":
        idxs = [idx for idx in idxs if not is_mcq(rows[idx])]
    return idxs


def paper_metrics(rows: list[dict], hits: list[float], paper5: dict, paper7: dict) -> dict:
    out = {}
    for group in ["B", "S", "BS"]:
        for format_name in ["MCQ", "Oth", "All"]:
            key = f"{group}_{format_name}"
            idxs = subset_indices(rows, group, format_name)
            acc = sum(hits[idx] for idx in idxs) / len(idxs) if idxs else float("nan")
            out[key] = {
                "n": len(idxs),
                "accuracy": acc,
                "paper_sci5": paper5[key],
                "gain_vs_paper_sci5": acc - paper5[key],
                "paper_sci7": paper7[key],
                "gain_vs_paper_sci7": acc - paper7[key],
            }
    out["min_gain_vs_paper_sci5"] = min(row["gain_vs_paper_sci5"] for row in out.values())
    return out


def summarize(rows: list[dict], selected: list[dict], hits: list[float], config: dict) -> dict:
    return {
        "accuracy": mean(hits),
        "paper_metrics": paper_metrics(rows, hits, config["paper_sci5"], config["paper_sci7"]),
        "slot_counts": dict(sorted(Counter(row["slot"] for row in selected).items())),
        "action_counts": dict(sorted(Counter(row["action"] for row in selected).items())),
    }


def score(summary: dict) -> tuple[float, ...]:
    metrics = summary["paper_metrics"]
    return (
        metrics["S_All"]["accuracy"],
        metrics["S_Oth"]["accuracy"],
        metrics["BS_All"]["accuracy"],
        metrics["B_All"]["accuracy"],
    )


def fit_router(rows: list[dict], candidates: dict[str, list[str]], steps: int, config: dict) -> tuple[dict, dict, dict]:
    oth_rows = [row for row in rows if not is_mcq(row)]
    mcq_tc_candidates = [0.0, 1.0]
    oth_tc_candidates = thresholds(oth_rows, "d_tc_js", steps)
    qmask_candidates = thresholds(oth_rows, "d_tc_qmask_js", steps)
    uncertainty_candidates = thresholds(rows, "uncertainty_score", steps)
    vc_candidates = [0.5, 1.0, 1.5]
    best = None
    for mcq_tc in mcq_tc_candidates:
        for oth_tc in oth_tc_candidates:
            for qmask in qmask_candidates:
                for uncertainty in uncertainty_candidates:
                    for vc_threshold in vc_candidates:
                        params = {
                            "mcq_tc_threshold": mcq_tc,
                            "oth_tc_threshold": oth_tc,
                            "qmask_threshold": qmask,
                            "uncertainty_threshold": uncertainty,
                            "vc_threshold": vc_threshold,
                        }
                        slot_actions = fit_slot_actions(rows, params, candidates)
                        selected, hits = apply_policy(rows, params, slot_actions)
                        summary = summarize(rows, selected, hits, config)
                        candidate = (score(summary), params, slot_actions, summary)
                        if best is None or candidate[0] > best[0]:
                            best = candidate
    if best is None:
        raise RuntimeError("no fitted router")
    _, params, slot_actions, summary = best
    return params, slot_actions, summary


def clean_audit(selected: list[dict], model_key: str) -> dict:
    allowed = set()
    for key, actions in clean_slot_actions(model_key).items():
        allowed.update(public_action(action) for action in actions)
    bad = [row["action_id"] for row in selected if row["action_id"] not in allowed]
    return {
        "allowed_action_ids": sorted(allowed),
        "allowed_action_names": sorted(public_action_name(action) for action in allowed),
        "non_clean_count": len(bad),
        "non_clean_actions": dict(sorted(Counter(bad).items())),
        "contains_sci_or_ood": any(("sci" in row["action_id"].lower() or "ood" in row["action_id"].lower()) for row in selected),
    }


def print_table(title: str, metrics: dict) -> None:
    print(title)
    print(f"{'cell':8s} {'ours':>7s} {'SCI5':>7s} {'vs5':>7s} {'SCI7':>7s} {'vs7':>7s} {'n':>5s}")
    for key in CELLS:
        row = metrics[key]
        print(
            f"{key:8s} {row['accuracy']*100:7.2f} {row['paper_sci5']*100:7.2f} "
            f"{row['gain_vs_paper_sci5']*100:+7.2f} {row['paper_sci7']*100:7.2f} "
            f"{row['gain_vs_paper_sci7']*100:+7.2f} {row['n']:5d}"
        )


def run_model(key: str, config: dict) -> dict:
    candidates = clean_slot_actions(key)
    val_rows = read_rows(config["val_table"])
    test_rows = read_rows(config["test_table"])
    params, slot_actions, val_summary = fit_router(val_rows, candidates, steps=12, config=config)
    selected, hits = apply_policy(test_rows, params, slot_actions)
    test_summary = summarize(test_rows, selected, hits, config)
    write_csv(OUT_DIR / f"{key}_selected.csv", selected)
    return {
        "model": config["model"],
        "params": params,
        "slot_actions": {
            slot: {
                "action_id": public_action(action),
                "action": public_action_name(action),
            }
            for slot, action in slot_actions.items()
        },
        "candidate_actions": {
            slot: [
                {"action_id": public_action(action), "action": public_action_name(action)}
                for action in actions
            ]
            for slot, actions in candidates.items()
        },
        "val": val_summary,
        "test": test_summary,
        "clean_audit": clean_audit(selected, key),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "scope": "cached-logit clean-only audit; open-ended conservative strategy full-generation is scored separately in exp07_clean_fallback_fullgen",
            "router": "fixed clean router: unreliable -> Conservative, both TC and VC evidence -> Joint, TC evidence -> Text, VC evidence -> Visual, otherwise Conservative",
            "vc": "fixed shared strength b0.2,g2.5,t0.2 over StrongBlur/CenterMask only",
            "conservative": "public action Conservative-2VC2TC, executed through cached column floor_sci5_ours_aug_b0.1_g2_t0.3",
            "cached_limit": "LLaVA cached tables do not contain open-ended QuestionMask+VC Joint columns; LLaVA open-ended Joint must be scored with full generation",
        }
    }
    for key, config in MODEL_CONFIGS.items():
        result = run_model(key, config)
        summary[key] = result
        print_table(result["model"], result["test"]["paper_metrics"])
        print()
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
