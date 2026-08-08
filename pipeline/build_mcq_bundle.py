"""Rebuild the multiple-choice rows with all seven views, from the step-0 dumps.

The frozen bundle stores family C already routed, as one `selected_vc` vector,
so the branch screen cannot re-run the argmin-iic choice over the three sources
the way the newer pair does.  The dumps hold every view, so the screen's inputs
can be rebuilt without touching a GPU.

Views follow the deployed ladder: the real image, the blank contrast, the same
pair under the answer-format wording, and the three corrupted sources family C
routes over.  Scores are the option letters' log-softmax, the same convention
the frozen bundle uses.

Usage:  <myenv>/bin/python research/magic/mcq_channels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DUMP = Path("/data2/shunshungu/dump_tensors")
LMU = Path.home() / "LMUData"
OUT = ROOT / "pipeline/data/mcq_channels_cache"
DATASETS = ["MMStar", "CCBench", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11"]
BACKBONE = {"qwen": "Qwen2-VL-7B", "llava": "LLaVA-NeXT-8B"}
TOK = {"qwen": "/data/shunshungu/models/Qwen2-VL-7B-Instruct",
       "llava": "/data/shunshungu/models/llama3-llava-next-8b-hf"}
VIEWS = {"real": "Original", "blank": "VCF-Color0",
         "gray": "VCF-Grayscale", "strongblur": "VCF-StrongBlur",
         "centermask": "VCF-CenterMask"}
# Family B needs only the difference of these two.  Qwen's Test dumps for them
# were never written, so the difference is taken from the frozen bundle's
# `answer_delta` and rows are matched on `s0`, which is unique per row.
OPTIONAL = {"ans_real": "TCF-AnswerFormat", "ans_blank": "TCF-AnswerFormat-Blank"}


def letter_ids(tok):
    out = {}
    for c in "ABCDE":
        s = {tok.encode(f, add_special_tokens=False)[0] for f in (c, " " + c)
             if len(tok.encode(f, add_special_tokens=False)) == 1}
        out[c] = sorted(s)
    return out


def build(mk, split):
    from transformers import AutoTokenizer
    ids = letter_ids(AutoTokenizer.from_pretrained(TOK[mk]))
    enc = BACKBONE[mk]
    rows, missing = [], 0
    for ds in DATASETS:
        name = f"{ds}_{enc}_Biased_{split}"
        tsv = LMU / f"{name}.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t")
        for i, r in df.iterrows():
            labels = [c for c in "ABCDE" if c in df.columns and not pd.isna(r.get(c))]
            if len(labels) < 2:
                continue
            sc = {}
            try:
                for k, v in {**VIEWS, **OPTIONAL}.items():
                    lg = torch.load(DUMP / f"{enc}-{v}" / name / f"logits_{i}.pt",
                                    map_location="cpu", weights_only=False)[0].float()
                    z = torch.log_softmax(lg, 0)
                    sc[k] = [round(max(float(z[t]) for t in ids[c]), 5) for c in labels]
            except FileNotFoundError:
                if any(k not in sc for k in VIEWS):
                    missing += 1
                    continue
            rows.append({"ds": ds, "row_i": int(i), "labels": labels,
                         "answer": str(r["answer"]).strip().upper(),
                         "mode": str(r.get("mode", "")).strip(), **sc})
    return rows, missing


def attach_answer_delta(rows, mk):
    """Rows whose answer-format dumps are absent take the contrast from file.

    Matching is on the real view, which is unique per row, within a tolerance:
    the rebuilt scores are rounded before storage and the frozen ones are not,
    so an exact key drops rows whose rounding falls on a boundary.
    """
    for r in rows:
        if "ans_real" in r:
            r["answer_delta"] = [round(r["ans_real"][i] - r["ans_blank"][i], 5)
                                 for i in range(len(r["labels"]))]
    need = [r for r in rows if "answer_delta" not in r]
    if not need:
        return 0, 0
    import gzip
    fz = json.load(gzip.open(ROOT / "magic_mcq_bundle.json.gz", "rt"))["mcq"][mk]
    buckets = {}
    for f in fz:
        buckets.setdefault((len(f["s0"]), round(f["s0"][0], 1)), []).append(f)
    hit = 0
    for r in need:
        v = r["real"]
        for f in buckets.get((len(v), round(v[0], 1)), ()):
            if max(abs(a - b) for a, b in zip(f["s0"], v)) < 1e-3:
                r["answer_delta"] = f["answer_delta"]
                hit += 1
                break
    return hit, len(need)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mk in ("qwen", "llava"):
        for split in ("Val", "Test"):
            rows, missing = build(mk, split)
            hit, need = attach_answer_delta(rows, mk)
            rows = [r for r in rows if "answer_delta" in r]
            if need:
                print(f"  {mk} {split}: {hit}/{need} rows took answer_delta from the bundle",
                      flush=True)
            path = OUT / f"{mk}_{split.lower()}.json"
            path.write_text(json.dumps(rows))
            from collections import Counter
            print(f"{mk} {split}: {len(rows)} rows ({missing} skipped for missing dumps)  "
                  f"{Counter(r['ds'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
