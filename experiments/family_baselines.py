"""Answer-level adaptations of two gated/adaptive-contrast baselines.

Pre-registration: experiments/data/family_baselines/PREREG.md.
Zero GPU: reads the frozen step-0 logit dumps only.

Phases
  --extract   build a compact option-score cache from the raw dumps
  --check     alignment check: the Original arm must reproduce the
              known base cells before any baseline number is read
  --fit       fit (alpha[, tau]) on the Biased_Val cache, per backbone
  --test      ONE Test read per arm per backbone, report as-is
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/data/family_baselines"
OUT.mkdir(exist_ok=True)
DUMP = Path("/data2/shunshungu/dump_tensors")
LMU = Path.home() / "LMUData"

MODELS = {"qwen": "Qwen2-VL-7B", "llava": "LLaVA-NeXT-8B"}
TOK_PATH = {"qwen": None,  # resolved from the project config below
            "llava": "/data/shunshungu/models/llama3-llava-next-8b-hf"}
# Amended 2026-07-22 before any readout: Noise400 and HighPass dumps do
# not cover Biased_Test on both backbones; the pool is the five corruptions
# with complete Val+Test coverage.
CORRUPTIONS = ["VCF-Color0", "VCF-Noise500", "VCF-Grayscale",
               "VCF-StrongBlur", "VCF-CenterMask"]
MCQ = ["MMStar", "CCBench", "MMBench_DEV_CN_V11", "MMBench_DEV_EN_V11"]
BIN = ["MME"]
SPLITS = ["Biased_Val", "Biased_Test"]

# Amended 2026-07-22 after a reviewer noted both fits landed on the grid
# edge: grids extended in the baseline-favouring direction until the
# optima are interior. This re-fit uses Val only; the consequent second
# Test read is disclosed in the PREREG amendment.
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0]
TAUS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]


def tokenizer_for(mk):
    from transformers import AutoTokenizer
    path = TOK_PATH[mk]
    if path is None:
        import natural_mcq_eval_qwen as NAT
        path = NAT.MODEL_PATH
    return AutoTokenizer.from_pretrained(path)


def letter_ids(tok, word):
    ids = set()
    for v in (word, " " + word, word.lower(), " " + word.lower(),
              word.upper(), " " + word.upper()):
        ids.add(tok(v, add_special_tokens=False).input_ids[-1])
    return sorted(ids)


def row_meta(model_name, ds, split):
    """(labels per row, gold per row, mode per row) from the LMUData TSVs."""
    df = pd.read_csv(LMU / f"{ds}_{model_name}_{split}.tsv", sep="\t")
    vcf = tcf = None
    if split == "Biased_Test":
        vcf = set(pd.read_csv(LMU / f"{ds}_{model_name}_VCF_Test.tsv",
                              sep="\t")["index"].astype(str))
        tcf = set(pd.read_csv(LMU / f"{ds}_{model_name}_TCF_Test.tsv",
                              sep="\t")["index"].astype(str))
    else:
        vcf = set(pd.read_csv(LMU / f"{ds}_{model_name}_VCF_Val.tsv",
                              sep="\t")["index"].astype(str))
        tcf = set(pd.read_csv(LMU / f"{ds}_{model_name}_TCF_Val.tsv",
                              sep="\t")["index"].astype(str))
    rows = []
    for _, r in df.iterrows():
        idx = str(r["index"])
        if ds in MCQ:
            labels = [c for c in ("A", "B", "C", "D") if c in df.columns
                      and not pd.isna(r.get(c))]
        else:
            labels = ["Yes", "No"]
        in_v, in_t = idx in vcf, idx in tcf
        mode = ("both" if in_v and in_t else
                "vcf_only" if in_v else
                "tcf_only" if in_t else "both")  # union member fallback
        rows.append({"labels": labels,
                     "answer": str(r["answer"]).strip(),
                     "mode": mode})
    return rows


def extract():
    """Raw 152k-dim dumps -> per-row option scores, one json per cell."""
    for mk, model_name in MODELS.items():
        tok = tokenizer_for(mk)
        idcache = {}
        def ids_of(word):
            if word not in idcache:
                idcache[word] = letter_ids(tok, word)
            return idcache[word]
        for ds in MCQ + BIN:
            for split in SPLITS:
                out_f = OUT / f"cache_{mk}_{ds}_{split}.json"
                if out_f.exists():
                    print("skip", out_f.name)
                    continue
                meta = row_meta(model_name, ds, split)
                variants = ["Original"] + CORRUPTIONS
                data = {v: [] for v in variants}
                n = len(meta)
                for v in variants:
                    d = DUMP / f"{model_name}-{v}" / f"{ds}_{model_name}_{split}"
                    if not d.exists():
                        raise SystemExit(f"missing dump dir {d}")
                    m = len(list(d.glob("logits_*.pt")))
                    if m != n:
                        raise SystemExit(f"row mismatch {d}: {m} vs tsv {n}")
                    for i in range(n):
                        lg = torch.load(d / f"logits_{i}.pt",
                                        map_location="cpu",
                                        weights_only=False)[0].float()
                        scores = [max(float(lg[t]) for t in ids_of(w))
                                  for w in meta[i]["labels"]]
                        data[v].append([round(s, 4) for s in scores])
                out_f.write_text(json.dumps(
                    {"meta": meta, "scores": data}))
                print("wrote", out_f.name, n, "rows", flush=True)


# ------------------------------------------------------------ analysis ----
def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def entropy(p):
    return -sum(x * math.log(max(x, 1e-12)) for x in p)


def load_cells(mk, split, datasets):
    rows = []
    for ds in datasets:
        c = json.loads((OUT / f"cache_{mk}_{ds}_{split}.json").read_text())
        for i, m in enumerate(c["meta"]):
            r = dict(m)
            r["orig"] = c["scores"]["Original"][i]
            r["corr"] = {v: c["scores"][v][i] for v in CORRUPTIONS}
            rows.append(r)
    return rows


GROUPS = {"B": {"vcf_only", "both"}, "S": {"tcf_only", "both"},
          "BS": {"vcf_only", "tcf_only", "both"}}


def cells(rows, pred):
    per = {g: [] for g in GROUPS}
    for r in rows:
        ok = r["labels"][pred(r)] == r["answer"] or \
             r["labels"][pred(r)].upper() == r["answer"].upper()
        for g, modes in GROUPS.items():
            if r["mode"] in modes:
                per[g].append(ok)
    return {g: round(100 * sum(v) / len(v), 2) for g, v in per.items() if v}


def arm_base(r):
    return max(range(len(r["orig"])), key=lambda i: r["orig"][i])


def arm_vacode(r, alpha, sense="max"):
    """VACoDe selects the augmentation of HIGHEST contrast (sense='max',
    faithful to the source). A reviewer asked for the opposite sense too,
    since MAGIC's own router takes an argmin; both are reported."""
    p0 = softmax(r["orig"])
    best, dist = None, (-1.0 if sense == "max" else 1e9)
    for v, sc in r["corr"].items():
        p = softmax(sc)
        d = sum((a - b) ** 2 for a, b in zip(p0, p)) ** 0.5
        if (d > dist) if sense == "max" else (d < dist):
            dist, best = d, sc
    f = [(1 + alpha) * o - alpha * c for o, c in zip(r["orig"], best)]
    return max(range(len(f)), key=lambda i: f[i])


def arm_chasd(r, alpha, tau):
    if entropy(softmax(r["orig"])) <= tau:
        return arm_base(r)
    c = r["corr"]["VCF-Noise500"]
    f = [(1 + alpha) * o - alpha * cc for o, cc in zip(r["orig"], c)]
    return max(range(len(f)), key=lambda i: f[i])


def check():
    for mk in MODELS:
        rows = load_cells(mk, "Biased_Test", MCQ)
        print(mk, "MCQ base cells from dumps:", cells(rows, arm_base))


def fit():
    sel = {}
    for mk in MODELS:
        rows = load_cells(mk, "Biased_Val", MCQ)
        best_v = max(ALPHAS, key=lambda a: cells(
            rows, lambda r: arm_vacode(r, a))["BS"])
        best_vmin = max(ALPHAS, key=lambda a: cells(
            rows, lambda r: arm_vacode(r, a, "min"))["BS"])
        best_c = max(((a, t) for a in ALPHAS for t in TAUS),
                     key=lambda at: cells(
                         rows, lambda r: arm_chasd(r, at[0], at[1]))["BS"])
        sel[mk] = {"vacode_alpha": best_v, "vacode_min_alpha": best_vmin,
                   "chasd_alpha": best_c[0], "chasd_tau": best_c[1],
                   "val_vacode": cells(rows, lambda r: arm_vacode(r, best_v)),
                   "val_chasd": cells(rows, lambda r: arm_chasd(r, *best_c))}
        print(mk, sel[mk])
    (OUT / "fit.json").write_text(json.dumps(sel, indent=2))


def test():
    sel = json.loads((OUT / "fit.json").read_text())
    rep = {}
    for mk in MODELS:
        s = sel[mk]
        rows = load_cells(mk, "Biased_Test", MCQ)
        fire = sum(entropy(softmax(r["orig"])) > s["chasd_tau"]
                   for r in rows) / len(rows)
        rep[mk] = {
            "vacode": {"alpha": s["vacode_alpha"], "passes": 6,
                       "cells": cells(rows, lambda r: arm_vacode(
                           r, s["vacode_alpha"]))},
            "vacode_min": {"alpha": s["vacode_min_alpha"], "passes": 6,
                           "cells": cells(rows, lambda r: arm_vacode(
                               r, s["vacode_min_alpha"], "min"))},
            "chasd": {"alpha": s["chasd_alpha"], "tau": s["chasd_tau"],
                      "fire_rate": round(fire, 3),
                      "passes": round(1 + fire, 2),
                      "cells": cells(rows, lambda r: arm_chasd(
                          r, s["chasd_alpha"], s["chasd_tau"]))},
            "binary_MME": {
                "vacode": cells(load_cells(mk, "Biased_Test", BIN),
                                lambda r: arm_vacode(r, s["vacode_alpha"])),
                "chasd": cells(load_cells(mk, "Biased_Test", BIN),
                               lambda r: arm_chasd(r, s["chasd_alpha"],
                                                   s["chasd_tau"]))}}
        print(mk, json.dumps(rep[mk], indent=2))
    (OUT / "test_readout.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("extract", "check", "fit", "test"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.check:
        check()
    if a.fit:
        fit()
    if a.test:
        test()
