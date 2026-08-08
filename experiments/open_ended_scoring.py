from __future__ import annotations

import csv
import json
import math
import os
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_LOG_ROOT = Path(os.environ.get("MAGIC_DATA_ROOT", ROOT / "experiments/data"))
RUNTIME_ROOTS = [
    ROOT,
    *[Path(p) for p in os.environ.get("MAGIC_RUNTIME_ROOT", "").split(":") if p],
]
ROUTED_DIR = RESEARCH_LOG_ROOT / "exp07_clean_joint_router_fullgen"
CACHED_SUMMARY = RESEARCH_LOG_ROOT / "exp07_clean_main_cached/summary.json"
OUT_DIR = ROOT / "experiments/data/open_ended_scoring_cache"

OPEN_DATASETS = ("MME", "ViLP")

GROUPS = {
    "B": {"vcf_only", "both"},
    "S": {"tcf_only", "both"},
    "BS": {"vcf_only", "tcf_only", "both"},
}

MODEL_CONFIGS = {
    "qwen": {
        "split_name": "Qwen2-VL-7B_Biased",
        "candidate_models": {
            "Text-QuestionMask": "Qwen2-VL-7B-TCF-QuestionMask",
            "Text-OptionsOnly": "Qwen2-VL-7B-TCF-OptionsOnly",
            "Visual-StrongBlur": "Qwen2-VL-7B-VCF-StrongBlur",
            "Visual-CenterMask": "Qwen2-VL-7B-VCF-CenterMask",
        },
    },
    "llava": {
        "split_name": "LLaVA-NeXT-8B_Biased",
        "candidate_models": {
            "Text-QuestionMask": "LLaVA-NeXT-8B-TCF-QuestionMask",
            "Text-OptionsOnly": "LLaVA-NeXT-8B-TCF-OptionsOnly",
            "Visual-StrongBlur": "LLaVA-NeXT-8B-VCF-StrongBlur",
            "Visual-CenterMask": "LLaVA-NeXT-8B-VCF-CenterMask",
        },
    },
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def cell_name_to_col(name: str) -> str:
    return "".join(ch for ch in name if ch.isalpha())


def xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    out = []
    for si in root:
        text = "".join(t.text or "" for t in si.iter() if t.tag.endswith("}t"))
        out.append(text)
    return out


def xlsx_cell_text(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iter() if t.tag.endswith("}t"))
    value = None
    for child in cell:
        if child.tag.endswith("}v"):
            value = child.text
            break
    if value is None:
        return ""
    if cell_type == "s":
        return shared[int(value)]
    return value


def read_xlsx(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        shared = xlsx_shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter():
        if not row.tag.endswith("}row"):
            continue
        values = {}
        for cell in row:
            if cell.tag.endswith("}c"):
                values[cell_name_to_col(cell.attrib["r"])] = xlsx_cell_text(cell, shared)
        rows.append(values)
    if not rows:
        return []
    header = rows[0]
    columns = {col: name for col, name in header.items()}
    out = []
    for raw in rows[1:]:
        item = {}
        for col, value in raw.items():
            if col in columns:
                item[columns[col]] = value
        out.append(item)
    return out


def paper_hit(answer: str, prediction: str) -> float:
    answer = str(answer).lower()
    prediction = str(prediction).lower()
    if len(prediction) < len(answer):
        return 0.0
    if len(prediction) == len(answer):
        return float(prediction == answer)
    if not bool(re.fullmatch(r"[A-Za-z]", prediction[len(answer)])):
        return float(prediction[: len(answer)] == answer)
    return 0.0


def find_xlsx(model_name: str, dataset: str, split_name: str, split: str) -> Path:
    pattern = f"**/{model_name}/T*/{model_name}_{dataset}_{split_name}_{split}.xlsx"
    matches = []
    for root in RUNTIME_ROOTS:
        matches.extend(
            path for path in root.glob(pattern)
            if path.is_file() and path.relative_to(root).parts[0].startswith("outputs")
        )
    matches = sorted(matches)
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def load_candidate_hits(config: dict, action: str, split: str) -> dict[tuple[str, str], float]:
    model_name = config["candidate_models"][action]
    hits = {}
    for dataset in OPEN_DATASETS:
        for row in read_xlsx(find_xlsx(model_name, dataset, config["split_name"], split)):
            key = (dataset, str(int(float(row["index"]))))
            hits[key] = paper_hit(row["answer"], row["prediction"])
    return hits


def add_candidate_hits(config: dict, rows: list[dict], split: str) -> list[dict]:
    hits_by_action = {action: load_candidate_hits(config, action, split) for action in config["candidate_models"]}
    out = []
    for row in rows:
        item = dict(row)
        key = (row["dataset"], str(int(float(row["index"]))))
        item["Conservative-2VC2TC"] = as_float(row, "floor_hit")
        item["Joint-StrongBlur-QuestionMask"] = as_float(row, "joint_hit")
        for action, hits in hits_by_action.items():
            if key not in hits:
                raise KeyError((split, action, key))
            item[action] = hits[key]
        out.append(item)
    return out


def metrics(rows: list[dict], hit_key: str) -> dict[str, dict]:
    out = {}
    for group, modes in GROUPS.items():
        sub = [row for row in rows if row["mode"] in modes]
        hit = sum(as_float(row, hit_key) for row in sub)
        out[f"{group}_Oth"] = {"acc": 100.0 * hit / len(sub), "n": len(sub), "hit": int(hit)}
    return out


def combine_all(model_key: str, oth_metrics: dict, cached: dict) -> dict:
    cached_metrics = cached[model_key]["test"]["paper_metrics"]
    out = {}
    for group in GROUPS:
        mcq = cached_metrics[f"{group}_MCQ"]
        oth = oth_metrics[f"{group}_Oth"]
        n = int(mcq["n"]) + int(oth["n"])
        acc = (float(mcq["accuracy"]) * int(mcq["n"]) + (float(oth["acc"]) / 100.0) * int(oth["n"])) / n
        sci7 = float(cached_metrics[f"{group}_All"]["paper_sci7"])
        out[f"{group}_All"] = {
            "acc": 100.0 * acc,
            "n": n,
            "paper_sci7": 100.0 * sci7,
            "gain_vs_sci7": 100.0 * (acc - sci7),
        }
    return out


def thresholds(rows: list[dict], feature: str, limit: int = 60) -> list[float]:
    values = sorted({as_float(row, feature) for row in rows if not math.isnan(as_float(row, feature))})
    if len(values) <= limit:
        return values
    return [values[round(i * (len(values) - 1) / limit)] for i in range(limit + 1)]


def route_by_threshold(rows: list[dict], feature: str, op: str, tau: float, action: str) -> list[dict]:
    out = []
    for row in rows:
        value = as_float(row, feature)
        use_action = value <= tau if op == "<=" else value >= tau
        item = dict(row)
        item["selected_action"] = action if use_action else "Conservative-2VC2TC"
        item["route_hit"] = as_float(row, item["selected_action"])
        out.append(item)
    return out


def objective(metric: dict) -> tuple[float, float, float, float]:
    return (
        metric["S_Oth"]["acc"],
        metric["BS_Oth"]["acc"],
        metric["B_Oth"]["acc"],
        min(metric["B_Oth"]["acc"], metric["S_Oth"]["acc"], metric["BS_Oth"]["acc"]),
    )


def sweep_one_action(model_key: str, val_rows: list[dict], test_rows: list[dict], action: str, cached: dict) -> list[dict]:
    features = [
        "uncertainty_score",
        "orig_margin",
        "d_tc_js",
        "d_tc_qmask_js",
        "image_info_conf_strong_blur",
        "d_vc_strong_blur_js",
        "image_info_conf_center_mask",
        "d_vc_center_mask_js",
        "vc_same_count_blur",
    ]
    out = []
    for feature in features:
        for op in ("<=", ">="):
            best = None
            for tau in thresholds(val_rows, feature):
                val_selected = route_by_threshold(val_rows, feature, op, tau, action)
                val_metric = metrics(val_selected, "route_hit")
                candidate = (objective(val_metric), tau, val_metric)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is None:
                continue
            _, tau, val_metric = best
            test_selected = route_by_threshold(test_rows, feature, op, tau, action)
            test_metric = metrics(test_selected, "route_hit")
            out.append(
                {
                    "action": action,
                    "feature": feature,
                    "op": op,
                    "tau": tau,
                    "joint_or_action_count": sum(row["selected_action"] == action for row in test_selected),
                    "val_oth": val_metric,
                    "test_oth": test_metric,
                    "test_combined": combine_all(model_key, test_metric, cached),
                }
            )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = json.loads(CACHED_SUMMARY.read_text(encoding="utf-8"))
    summary = {
        "protocol": {
            "scope": "Open-ended clean candidate diagnostic only.",
            "eval": "Full-generation predictions are scored with tools/evaluate_dataset.py prefix matching.",
            "route": "Each sweep chooses one clean candidate action versus Conservative-2VC2TC using one reliability feature fitted on Val.",
        },
        "models": {},
    }
    for model_key, config in MODEL_CONFIGS.items():
        val_rows = add_candidate_hits(config, read_csv(ROUTED_DIR / f"{model_key}_val_routed_rows.csv"), "Val")
        test_rows = add_candidate_hits(config, read_csv(ROUTED_DIR / f"{model_key}_test_routed_rows.csv"), "Test")

        fixed = {}
        for action in ["Conservative-2VC2TC", "Joint-StrongBlur-QuestionMask", *config["candidate_models"]]:
            oth = metrics(test_rows, action)
            fixed[action] = {"test_oth": oth, "test_combined": combine_all(model_key, oth, cached)}

        sweeps = []
        for action in [
            "Joint-StrongBlur-QuestionMask",
            "Text-QuestionMask",
            "Text-OptionsOnly",
            "Visual-StrongBlur",
            "Visual-CenterMask",
        ]:
            sweeps.extend(sweep_one_action(model_key, val_rows, test_rows, action, cached))
        sweeps.sort(
            key=lambda row: (
                row["test_combined"]["S_All"]["acc"],
                row["test_combined"]["BS_All"]["acc"],
                row["test_combined"]["B_All"]["acc"],
            ),
            reverse=True,
        )
        summary["models"][model_key] = {
            "fixed_actions": fixed,
            "top_sweeps": sweeps[:30],
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for model_key, model in summary["models"].items():
        print(model_key, "fixed actions")
        for action, row in model["fixed_actions"].items():
            b = row["test_combined"]["B_All"]["acc"]
            s = row["test_combined"]["S_All"]["acc"]
            bs = row["test_combined"]["BS_All"]["acc"]
            print(f"  {action:34s} B/S/BS {b:.2f}/{s:.2f}/{bs:.2f}")
        print(model_key, "top routed sweeps")
        for row in model["top_sweeps"][:8]:
            comb = row["test_combined"]
            print(
                f"  {row['action']:34s} {row['feature']} {row['op']} {row['tau']:.6g} "
                f"n={row['joint_or_action_count']:3d} "
                f"B/S/BS {comb['B_All']['acc']:.2f}/{comb['S_All']['acc']:.2f}/{comb['BS_All']['acc']:.2f}"
            )
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
