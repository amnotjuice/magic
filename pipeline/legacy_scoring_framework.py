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

from open_ended_scoring import (  # noqa: E402
    MODEL_CONFIGS as OPEN_CONFIGS,
    add_candidate_hits,
    find_xlsx,
    paper_hit,
    read_csv,
    read_xlsx,
)
from legacy_cached_score import MODEL_CONFIGS as CACHED_CONFIGS  # noqa: E402
from action_grammar import public_action_name  # noqa: E402


OUT_DIR = ROOT / "pipeline/data/legacy_scoring_framework"
OPEN_ROUTED_DIR = RESEARCH_LOG_ROOT / "exp07_clean_joint_router_fullgen"
JOINT_CENTER_QMASK_DIR = Path("/dev/shm/clean_joint_center_qmask_fullgen")
DEFAULT_VC_STRONG_TAU = 0.06462359428405762
LLAVA_B5_FAMILY_DIR = RESEARCH_LOG_ROOT / "exp07_llava_b5_family_controller"
CURRENT_VARIANT = "tc_early_js_full_gray3"

GROUPS = {
    "B": {"vcf_only", "both"},
    "S": {"tcf_only", "both"},
    "BS": {"vcf_only", "tcf_only", "both"},
}

MCQ_ACTIONS = {
    "qwen": {
        "text": "b5_tc_os1_direct",
        "visual_strong": "both_max_strong_blur_b0.2_g2.5_t0.2",
        "visual_center": "both_max_center_mask_b0.2_g2.5_t0.2",
        "visual_gray": "both_max_grayscale_b0.2_g2.5_t0.2",
        "joint_strong": "b5_joint_strong_os1",
        "joint_center": "b5_joint_center_os1",
        "conservative": "floor_sci5_ours_aug_b0.1_g2_t0.3",
    },
    "llava": {
        "text": "b5_tc_os1_direct",
        "visual_strong": "both_max_strong_blur_b0.2_g2.5_t0.2",
        "visual_center": "both_max_center_mask_b0.2_g2.5_t0.2",
        "visual_gray": "both_max_grayscale_b0.2_g2.5_t0.2",
        "joint_strong": "joint_os_strong_blur_b0.2_g2.5_t0.2",
        "joint_center": "joint_os_center_mask_b0.2_g2.5_t0.2",
        "conservative": "floor_sci5_ours_aug_b0.1_g2_t0.3",
    },
}

OPEN_ACTIONS = {
    "text": "Text-QuestionMask",
    "visual_strong": "Visual-StrongBlur",
    "visual_center": "Visual-CenterMask",
    "visual_gray": "Visual-Grayscale",
    "joint_strong": "Joint-StrongBlur-QuestionMask",
    "joint_center": "Joint-CenterMask-QuestionMask",
    "conservative": "Conservative-2VC2TC",
}

# Pure-visual route 3-way source selector: pick among {strong, center, gray} by their
# confidence probes. Cascade@TAU=0: run sources in a Val-fit order, stop at the first whose
# ablation does not cost confidence (conf <= 0 = clean counterfactual); else argmin-conf.
# Joint route keeps the 2-way strong/center selector (gray joint columns are not materialized
# for Qwen MCQ, so 3-way is scoped to the pure-visual route only).
VC_SOURCE_CONF = {
    "strong": "image_info_conf_strong_blur",
    "center": "image_info_conf_center_mask",
    "gray": "image_info_conf_grayscale",
}
VC_CASCADE_TAU = 0.0
TC_OS1_JS_FEATURE = "d_tc_optionshuffle1_js"


def vc_cascade_pick(row: dict, order: list[str]) -> tuple[str, int]:
    """Return (chosen_source, n_sources_run) using cascade@TAU=0 over the Val-fit order."""
    for depth, source in enumerate(order, start=1):
        if as_float(row, VC_SOURCE_CONF[source]) <= VC_CASCADE_TAU:
            return source, depth
    # No clean-enough ablation found: all sources run -> pick the smallest confidence drop.
    chosen = min(order, key=lambda s: as_float(row, VC_SOURCE_CONF[s]))
    return chosen, len(order)


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


def is_mcq(row: dict) -> bool:
    return bool(int(float(row.get("clean_is_mcq", "0"))))


def source_state(
    row: dict,
    params: dict,
    tc_any_flip: bool,
    vc_strong_tau: float,
    tc_early_model: dict | None = None,
) -> tuple[str, str, dict]:
    reliable = as_float(row, "uncertainty_score") <= float(params["uncertainty_threshold"])
    if not reliable:
        return "none", "none", {"reliable": 0, "tc_source": "none", "vc_source": "none"}

    if is_mcq(row):
        tc_score = as_float(row, "tc_os_semantic_flip_count")
        tc_probe_count = 2
        tc_stage = "tc_full_two_probe"
        if tc_early_model is not None:
            os1_js = as_float(row, TC_OS1_JS_FEATURE)
            low = float(tc_early_model["low"])
            high = float(tc_early_model["high"])
            if os1_js <= low:
                tc_on = False
                tc_probe_count = 1
                tc_stage = "tc_os1_low"
            elif os1_js >= high:
                tc_on = True
                tc_probe_count = 1
                tc_stage = "tc_os1_high"
            else:
                tc_on = tc_score > (0.0 if tc_any_flip else 1.0)
                tc_stage = "tc_os2_fallback"
        else:
            tc_on = tc_score > (0.0 if tc_any_flip else 1.0)
    else:
        tc_score = as_float(row, "d_tc_qmask_js")
        tc_on = tc_score > float(params["qmask_threshold"])
        tc_probe_count = 1
        tc_stage = "tc_qmask"

    vc_count = as_float(row, "vc_same_count_blur")
    vc_on = vc_count > float(params["vc_threshold"])
    if not vc_on:
        vc_source = "none"
    elif as_float(row, "image_info_conf_strong_blur") <= vc_strong_tau:
        vc_source = "strong"
    else:
        vc_source = "center"

    return (
        "qmask" if tc_on and not is_mcq(row) else "option_shuffle" if tc_on else "none",
        vc_source,
        {
            "reliable": 1,
            "tc_score": tc_score,
            "tc_probe_count": tc_probe_count,
            "tc_stage": tc_stage,
            "vc_score": vc_count,
            "tc_source": "qmask" if tc_on and not is_mcq(row) else "option_shuffle" if tc_on else "none",
            "vc_source": vc_source,
        },
    )


def _visual_route(model_key: str, row: dict, vc_source: str, evidence: dict, vc_cascade_order):
    """Resolve the pure-visual route. With a cascade order, re-pick the source 3-way."""
    actions = MCQ_ACTIONS[model_key] if is_mcq(row) else OPEN_ACTIONS
    if vc_cascade_order is not None:
        source, depth = vc_cascade_pick(row, vc_cascade_order)
        evidence["vc_source"] = source
        evidence["vc_cascade_depth"] = depth
        state = {"strong": "visual_strong", "center": "visual_center", "gray": "visual_gray"}[source]
        return actions[state], state, evidence
    state = "visual_strong" if vc_source == "strong" else "visual_center"
    return actions[state], state, evidence


def choose_action(
    model_key: str,
    row: dict,
    params: dict,
    tc_any_flip: bool,
    vc_strong_tau: float,
    vc_cascade_order=None,
    tc_early_model: dict | None = None,
) -> tuple[str, str, dict]:
    tc_source, vc_source, evidence = source_state(row, params, tc_any_flip, vc_strong_tau, tc_early_model)
    actions = MCQ_ACTIONS[model_key] if is_mcq(row) else OPEN_ACTIONS
    # Joint route keeps the 2-way strong/center source.
    if tc_source != "none" and vc_source == "strong":
        return actions["joint_strong"], "joint_strong", evidence
    if tc_source != "none" and vc_source == "center":
        return actions["joint_center"], "joint_center", evidence
    if tc_source != "none":
        return actions["text"], "text", evidence
    # Pure-visual route: optional 3-way cascade source selection.
    if vc_source in ("strong", "center"):
        return _visual_route(model_key, row, vc_source, evidence, vc_cascade_order)
    return actions["conservative"], "conservative", evidence


def choose_action_with_policy(
    model_key: str,
    row: dict,
    params: dict,
    tc_any_flip: bool,
    conservative_policy: str,
    vc_only_policy: str,
    vc_strong_tau: float,
    vc_cascade_order=None,
    tc_early_model: dict | None = None,
) -> tuple[str, str, dict]:
    action, state, evidence = choose_action(
        model_key, row, params, tc_any_flip, vc_strong_tau, vc_cascade_order, tc_early_model
    )
    if state == "conservative":
        return conservative_action(model_key, row, conservative_policy), f"conservative_{conservative_policy}", evidence
    if state in {"visual_strong", "visual_center", "visual_gray"} and vc_only_policy != "visual":
        return conservative_action(model_key, row, vc_only_policy), f"vc_only_{vc_only_policy}", evidence
    return action, state, evidence


def conservative_action(model_key: str, row: dict, policy: str) -> str:
    if policy == "full":
        return MCQ_ACTIONS[model_key]["conservative"] if is_mcq(row) else OPEN_ACTIONS["conservative"]
    if policy == "light_strong":
        return MCQ_ACTIONS[model_key]["joint_strong"] if is_mcq(row) else OPEN_ACTIONS["joint_strong"]
    if policy == "light_center":
        return MCQ_ACTIONS[model_key]["joint_center"] if is_mcq(row) else OPEN_ACTIONS["joint_center"]
    raise ValueError(policy)


def load_open_rows(model_key: str, split: str) -> list[dict]:
    rows = read_csv(OPEN_ROUTED_DIR / f"{model_key}_{split.lower()}_routed_rows.csv")
    out = add_candidate_hits(OPEN_CONFIGS[model_key], rows, split)
    center_hits = load_joint_center_qmask_hits(model_key, split)
    gray_hits = load_grayscale_visual_hits(model_key, split)
    for row in out:
        row["clean_is_mcq"] = "0"
        key = (row["dataset"], str(int(float(row["index"]))))
        row["Joint-CenterMask-QuestionMask"] = center_hits[key]
        # Open-ended grayscale must be full-generation scored, NOT the first-token cached column.
        row["Visual-Grayscale"] = gray_hits[key]
    return out


def load_mcq_rows(model_key: str, split: str) -> list[dict]:
    config = CACHED_CONFIGS[model_key]
    table = config["val_table"] if split == "Val" else config["test_table"]
    rows = [row for row in read_rows(table) if is_mcq(row)]
    if model_key != "llava":
        return rows

    # The clean LLaVA floor table omitted first-probe JS, but the matching B5
    # family table materialized it for the same MCQ rows.
    b5_table = LLAVA_B5_FAMILY_DIR / f"{split.lower()}_action_table.csv"
    b5_rows = {
        (row["dataset"], row["split_name"], row["index"]): row
        for row in read_rows(b5_table)
    }
    for row in rows:
        match = b5_rows[(row["dataset"], row["split_name"], row["index"])]
        for key in (TC_OS1_JS_FEATURE, "d_tc_optionshuffle2_js", "d_tc_optionshuffle_js"):
            row[key] = match[key]
    return rows


def load_grayscale_visual_hits(model_key: str, split: str) -> dict[tuple[str, str], float]:
    config = OPEN_CONFIGS[model_key]
    model_name = {
        "qwen": "Qwen2-VL-7B-VCF-Grayscale",
        "llava": "LLaVA-NeXT-8B-VCF-Grayscale",
    }[model_key]
    hits = {}
    for dataset in ("MME", "ViLP"):
        for row in read_xlsx(find_xlsx(model_name, dataset, config["split_name"], split)):
            key = (dataset, str(int(float(row["index"]))))
            hits[key] = paper_hit(row["answer"], row["prediction"])
    return hits


def load_joint_center_qmask_hits(model_key: str, split: str) -> dict[tuple[str, str], float]:
    config = OPEN_CONFIGS[model_key]
    model_name = {
        "qwen": "Qwen2-VL-7B-Joint-CenterMask-QuestionMask-b02a1g25t02",
        "llava": "LLaVA-NeXT-8B-Joint-CenterMask-QuestionMask-b02a1g25t02",
    }[model_key]
    hits = {}
    for dataset in ("MME", "ViLP"):
        try:
            path = find_xlsx(model_name, dataset, config["split_name"], split)
        except FileNotFoundError:
            pattern = f"**/{model_name}_{dataset}_{config['split_name']}_{split}.xlsx"
            matches = sorted(JOINT_CENTER_QMASK_DIR.glob(pattern))
            matches.extend(sorted((RESEARCH_LOG_ROOT / "exp08_clean_framework_materialized").glob(pattern)))
            if not matches:
                raise
            path = matches[-1]
        for row in read_xlsx(path):
            key = (dataset, str(int(float(row["index"]))))
            hits[key] = paper_hit(row["answer"], row["prediction"])
    return hits


def source_threshold_candidates(rows: list[dict]) -> list[float]:
    values = sorted({
        as_float(row, "image_info_conf_strong_blur")
        for row in rows
        if not math.isnan(as_float(row, "image_info_conf_strong_blur"))
    })
    if not values:
        return [DEFAULT_VC_STRONG_TAU]
    return [values[0] - 1e-12, *values]


def fit_vc_source_tau(model_key: str, rows: list[dict], params: dict, tc_any_flip: bool) -> dict:
    """Fit only the StrongBlur-vs-CenterMask split on Val source-routed rows."""
    best = None
    for tau in source_threshold_candidates(rows):
        hits = []
        for row in rows:
            action, state, _ = choose_action(model_key, row, params, tc_any_flip, tau)
            if state in {"visual_strong", "visual_center", "joint_strong", "joint_center"}:
                hits.append(as_float(row, action))
        if not hits:
            continue
        mean_hit = sum(hits) / len(hits)
        candidate = (mean_hit, len(hits), tau)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return {
            "tau": DEFAULT_VC_STRONG_TAU,
            "val_source_acc": float("nan"),
            "val_source_n": 0,
        }
    mean_hit, n, tau = best
    return {
        "tau": tau,
        "val_source_acc": mean_hit,
        "val_source_n": n,
    }


def apply_rows(
    model_key: str,
    rows: list[dict],
    params: dict,
    tc_any_flip: bool,
    conservative_policy: str,
    vc_only_policy: str,
    vc_strong_tau: float,
    vc_cascade_order=None,
    tc_early_model: dict | None = None,
) -> tuple[list[dict], list[float]]:
    selected = []
    hits = []
    for row in rows:
        action, state, evidence = choose_action_with_policy(
            model_key, row, params, tc_any_flip, conservative_policy, vc_only_policy,
            vc_strong_tau, vc_cascade_order, tc_early_model,
        )
        hit = as_float(row, action)
        selected.append(
            {
                "dataset": row["dataset"],
                "split_name": row["split_name"],
                "index": row["index"],
                "mode": row["mode"],
                "format": "MCQ" if is_mcq(row) else "Oth",
                "action": action,
                "action_name": public_action_name(action),
                "state": state,
                "hit": hit,
                "passes": pass_count(
                    row,
                    state,
                    tc_any_flip,
                    evidence.get("vc_cascade_depth"),
                    evidence.get("tc_probe_count") if tc_early_model is not None else None,
                ),
                **evidence,
            }
        )
        hits.append(hit)
    return selected, hits


def pass_count(row: dict, state: str, tc_any_flip: bool, vc_cascade_depth=None, tc_probe_count=None) -> int:
    # True online cost counts routing probes as well as the selected action.
    # The router first runs Original; MCQ routing needs OS1+OS2 and both visual
    # probes, while open-ended routing needs QuestionMask and both visual probes.
    # With the 3-way cascade, the pure-visual route's VC cost = sources run (1..3),
    # so it is itemized as TC routing (3 MCQ / 2 open) + cascade depth.
    if tc_probe_count is not None and is_mcq(row):
        tc_cost = 1 + int(tc_probe_count)
        if state in {"visual_strong", "visual_center", "visual_gray"} and vc_cascade_depth is not None:
            return tc_cost + int(vc_cascade_depth)
        if state == "text":
            return tc_cost
        if state == "joint_strong":
            return tc_cost + 1
        if state == "joint_center":
            return tc_cost + 2
        if state in {"conservative_full", "conservative"}:
            return 5
    if vc_cascade_depth is not None and state in {"visual_strong", "visual_center", "visual_gray"}:
        return (3 if is_mcq(row) else 2) + int(vc_cascade_depth)
    if is_mcq(row):
        if state in {"conservative_light_strong", "conservative_light_center"}:
            return 3
        return 5
    if state in {"conservative_light_strong", "conservative_light_center"}:
        return 3
    if state.startswith("vc_only_"):
        return 4
    if state in {"text", "visual_strong", "visual_center", "visual_gray", "joint_strong", "joint_center"}:
        return 4
    if state in {"conservative_full", "conservative"}:
        return 5
    return 5


def subset(rows: list[dict], hits: list[float], group: str, fmt: str) -> tuple[int, float]:
    idxs = [idx for idx, row in enumerate(rows) if row["mode"] in GROUPS[group]]
    if fmt == "MCQ":
        idxs = [idx for idx in idxs if row_is_mcq(rows[idx])]
    elif fmt == "Oth":
        idxs = [idx for idx in idxs if not row_is_mcq(rows[idx])]
    return len(idxs), sum(hits[idx] for idx in idxs)


def row_is_mcq(row: dict) -> bool:
    return row.get("format") == "MCQ" or is_mcq(row)


def metrics(selected: list[dict], hits: list[float], config: dict) -> dict:
    out = {}
    for group in ["B", "S", "BS"]:
        all_n = 0
        all_hit = 0.0
        for fmt in ["MCQ", "Oth"]:
            n, hit = subset(selected, hits, group, fmt)
            all_n += n
            all_hit += hit
            key = f"{group}_{fmt}"
            acc = hit / n if n else 0.0
            out[key] = {
                "n": n,
                "accuracy": acc,
                "paper_sci5": config["paper_sci5"][key],
                "gain_vs_paper_sci5": acc - config["paper_sci5"][key],
                "paper_sci7": config["paper_sci7"][key],
                "gain_vs_paper_sci7": acc - config["paper_sci7"][key],
            }
        key = f"{group}_All"
        acc = all_hit / all_n if all_n else 0.0
        out[key] = {
            "n": all_n,
            "accuracy": acc,
            "paper_sci5": config["paper_sci5"][key],
            "gain_vs_paper_sci5": acc - config["paper_sci5"][key],
            "paper_sci7": config["paper_sci7"][key],
            "gain_vs_paper_sci7": acc - config["paper_sci7"][key],
        }
    return out


def avg_passes(selected: list[dict]) -> float:
    return sum(as_float(row, "passes") for row in selected) / len(selected)


def threshold_candidates(rows: list[dict], feature: str, max_thresholds: int = 17) -> list[float]:
    values = sorted(as_float(row, feature) for row in rows if not math.isnan(as_float(row, feature)))
    if not values:
        return [float("inf")]
    if len(values) <= max_thresholds:
        return sorted(set(values))
    idxs = {round(i * (len(values) - 1) / (max_thresholds - 1)) for i in range(max_thresholds)}
    return sorted({values[idx] for idx in idxs})


def fit_tc_early_model(
    model_key: str,
    rows: list[dict],
    params: dict,
    tc_any_flip: bool,
    conservative_policy: str,
    vc_only_policy: str,
    vc_strong_tau: float,
    vc_cascade_order,
) -> dict:
    mcq_rows = [row for row in rows if is_mcq(row)]
    thresholds = threshold_candidates(mcq_rows, TC_OS1_JS_FEATURE)
    best = None
    for low in thresholds:
        for high in thresholds:
            if low > high:
                continue
            model = {"low": low, "high": high}
            selected, hits = apply_rows(
                model_key,
                rows,
                params,
                tc_any_flip,
                conservative_policy,
                vc_only_policy,
                vc_strong_tau,
                vc_cascade_order,
                model,
            )
            metric = metrics(selected, hits, CACHED_CONFIGS[model_key])["BS_All"]["accuracy"]
            avg_cost = avg_passes(selected)
            candidate = {
                "low": low,
                "high": high,
                "val_bs_all": metric,
                "val_avg_passes": avg_cost,
            }
            key = (metric, -avg_cost)
            if best is None or key > (best["val_bs_all"], -best["val_avg_passes"]):
                best = candidate
    assert best is not None
    return best


def fit_vc_cascade_order(model_key: str, rows: list[dict], params: dict, tc_any_flip: bool, vc_strong_tau: float) -> list[str]:
    """Val-fit the cascade order = how often each source is the argmin-conf winner on the
    pure-visual subset (most-often-best source first, to maximise early stops)."""
    wins = Counter()
    for row in rows:
        _, state, _ = choose_action(model_key, row, params, tc_any_flip, vc_strong_tau)
        if state not in {"visual_strong", "visual_center"}:
            continue
        winner = min(VC_SOURCE_CONF, key=lambda s: as_float(row, VC_SOURCE_CONF[s]))
        wins[winner] += 1
    order = [s for s, _ in wins.most_common()]
    return order + [s for s in VC_SOURCE_CONF if s not in order]


def run_model(
    model_key: str,
    variant: str,
    tc_any_flip: bool,
    conservative_policy: str = "full",
    vc_only_policy: str = "visual",
    vc_three_way: bool = False,
    tc_early: bool = False,
) -> dict:
    config = CACHED_CONFIGS[model_key]
    cached_summary = json.loads((RESEARCH_LOG_ROOT / "exp07_clean_main_cached/summary.json").read_text(encoding="utf-8"))
    params = cached_summary[model_key]["params"]
    val_rows = load_mcq_rows(model_key, "Val")
    val_rows += load_open_rows(model_key, "Val")
    vc_source_fit = fit_vc_source_tau(model_key, val_rows, params, tc_any_flip)
    vc_strong_tau = float(vc_source_fit["tau"])
    cascade_order = (
        fit_vc_cascade_order(model_key, val_rows, params, tc_any_flip, vc_strong_tau)
        if vc_three_way else None
    )
    tc_early_model = (
        fit_tc_early_model(
            model_key,
            val_rows,
            params,
            tc_any_flip,
            conservative_policy,
            vc_only_policy,
            vc_strong_tau,
            cascade_order,
        )
        if tc_early else None
    )
    test_rows = load_mcq_rows(model_key, "Test")
    test_rows += load_open_rows(model_key, "Test")
    selected, hits = apply_rows(
        model_key,
        test_rows,
        params,
        tc_any_flip,
        conservative_policy,
        vc_only_policy,
        vc_strong_tau,
        cascade_order,
        tc_early_model,
    )
    write_csv(OUT_DIR / f"{model_key}_{variant}_selected.csv", selected)
    vc_rule = (
        f"pure-visual: cascade@TAU=0 over {cascade_order} (stop at first image_info_conf<=0, else argmin-conf); "
        f"joint: image_info_conf_strong_blur <= {vc_strong_tau} ? StrongBlur : CenterMask"
        if vc_three_way else
        f"if vc_same_count_blur > vc_threshold and image_info_conf_strong_blur <= {vc_strong_tau} then StrongBlur else CenterMask"
    )
    return {
        "model": config["model"],
        "variant": variant,
        "params": {
            **params,
            "mcq_tc_rule": "tc_os_semantic_flip_count > 0" if tc_any_flip else "tc_os_semantic_flip_count > 1",
            "vc_source_rule": vc_rule,
            "vc_source_tau": vc_strong_tau,
            "vc_source_tau_fit": vc_source_fit,
            "vc_three_way": vc_three_way,
            "vc_cascade_order": cascade_order,
            "tc_early_model": tc_early_model,
            "conservative_policy": conservative_policy,
            "vc_only_policy": vc_only_policy,
        },
        "metrics": metrics(selected, hits, config),
        "avg_passes": avg_passes(selected),
        "state_counts": dict(sorted(Counter(row["state"] for row in selected).items())),
        "action_counts": dict(sorted(Counter(row["action_name"] for row in selected).items())),
        "raw_action_counts": dict(sorted(Counter(row["action"] for row in selected).items())),
    }


def print_summary(model_key: str, result: dict) -> None:
    m = result["metrics"]
    print(f"\n{model_key} {result['variant']}")
    print(f"B/S/BS All: {m['B_All']['accuracy']*100:.2f}/{m['S_All']['accuracy']*100:.2f}/{m['BS_All']['accuracy']*100:.2f}")
    print(f"vs SCI7: {m['B_All']['gain_vs_paper_sci7']*100:+.2f}/{m['S_All']['gain_vs_paper_sci7']*100:+.2f}/{m['BS_All']['gain_vs_paper_sci7']*100:+.2f}")
    print(f"avg_passes: {result['avg_passes']:.2f}")
    print("state_counts", result["state_counts"])
    print("action_counts", result["action_counts"])


def compact_row(variant: str, model_key: str, result: dict) -> dict:
    metrics_dict = result["metrics"]
    return {
        "variant": variant,
        "model": model_key,
        "avg_pass": round(result["avg_passes"], 2),
        "B_All": round(metrics_dict["B_All"]["accuracy"] * 100, 2),
        "S_All": round(metrics_dict["S_All"]["accuracy"] * 100, 2),
        "BS_All": round(metrics_dict["BS_All"]["accuracy"] * 100, 2),
        "Avg": round(
            (
                metrics_dict["B_All"]["accuracy"]
                + metrics_dict["S_All"]["accuracy"]
                + metrics_dict["BS_All"]["accuracy"]
            ) * 100 / 3,
            2,
        ),
        "vs7_B": round(metrics_dict["B_All"]["gain_vs_paper_sci7"] * 100, 2),
        "vs7_S": round(metrics_dict["S_All"]["gain_vs_paper_sci7"] * 100, 2),
        "vs7_BS": round(metrics_dict["BS_All"]["gain_vs_paper_sci7"] * 100, 2),
    }


def write_compact_tables(summary: dict) -> None:
    rows = []
    current_rows = []
    for variant, by_model in summary["models"].items():
        for model_key, result in by_model.items():
            row = compact_row(variant, model_key, result)
            rows.append(row)
            if variant == CURRENT_VARIANT:
                current_rows.append(row)
    write_csv(OUT_DIR / "variant_compact_table.csv", rows)
    write_csv(OUT_DIR / "current_method_table.csv", current_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "name": "materialized_clean_framework_v1",
            "current_variant": CURRENT_VARIANT,
            "reliability_guard": "uncertainty_score > cached Val threshold -> Conservative-2VC2TC",
            "tc_probe": "MCQ uses OptionShuffle semantic flip; Other uses QuestionMask because LLaVA OptionsOnly signal is not materialized.",
            "vc_probe": "Use vc_same_count_blur as visual-evidence trigger, then fit StrongBlur-vs-CenterMask source by image_info_conf_strong_blur on each model's Val split.",
            "avg_passes": "Counts true online passes, including routing probes reused by the selected action.",
            "fullgen": "Joint-CenterMask-QuestionMask is materialized under /dev/shm/clean_joint_center_qmask_fullgen and used for Other center+qmask routes.",
        },
        "models": {},
    }
    # (variant, tc_any_flip, conservative_policy, vc_only_policy, vc_three_way, tc_early)
    variants = [
        ("tc_any_flip_full", True, "full", "visual", False, False),
        ("tc_two_flip_full", False, "full", "visual", False, False),
        ("tc_two_flip_light_strong", False, "light_strong", "visual", False, False),
        ("tc_two_flip_light_center", False, "light_center", "visual", False, False),
        ("tc_two_flip_light_strong_vconly_light", False, "light_strong", "light_strong", False, False),
        ("tc_two_flip_light_center_vconly_light", False, "light_center", "light_center", False, False),
        ("tc_two_flip_full_gray3", False, "full", "visual", True, False),
        ("tc_early_js_full_gray3", False, "full", "visual", True, True),
    ]
    for variant, tc_any_flip, conservative_policy, vc_only_policy, vc_three_way, tc_early in variants:
        summary["models"][variant] = {}
        for model_key in ["qwen", "llava"]:
            result = run_model(
                model_key,
                variant,
                tc_any_flip,
                conservative_policy,
                vc_only_policy,
                vc_three_way,
                tc_early,
            )
            summary["models"][variant][model_key] = result
            print_summary(model_key, result)
    write_compact_tables(summary)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
