"""The calibration screen of Sec. 3.4, run on the OLDER pair's Biased_Val.

`magic_newgen_replay.py` runs this screen on Qwen3-VL / LLaVA-OneVision-1.5.
On Qwen2-VL / LLaVA-NeXT the deployed branch set was fixed by hand and the
ablation was reported afterwards, so the screen itself was never executed
there.  This script executes it, from the frozen step-0 logit dumps, so the
same procedure stands behind all four backbones.

Scope: multiple-choice only.  The answer-format views (TCF-AnswerFormat,
TCF-AnswerFormat-Blank) were dumped for the four MCQ datasets and not for
MME / ViLP, so the Others formats cannot be screened from these caches.

Read-only and zero GPU.  Nothing here re-selects a deployed constant: t2 and
tauV are magic.py's frozen MCQ values, and t1 is recomputed from its stated
label-free rule (95th percentile of the base margin on Biased_Val) as a check
that the rule reproduces the deployed number.

Usage:  python experiments/oldgen_screen.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

DUMP = Path("/data2/shunshungu/dump_tensors")
LMU = Path(os.path.expanduser("~/LMUData"))
SPLIT = "Biased_Val"
MCQ_DS = ["MMStar", "CCBench", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11"]

# view name -> dump directory suffix
VIEWS = {"s0": "Original",
         "blank": "VCF-Color0",
         "ans_real": "TCF-AnswerFormat",
         "ans_blank": "TCF-AnswerFormat-Blank",
         "gray": "VCF-Grayscale",
         "strongblur": "VCF-StrongBlur",
         "centermask": "VCF-CenterMask"}

# magic.py LADDER_MCQ: the deployed restoration constants (not re-selected here)
DEPLOYED = {"Qwen2-VL-7B": {"t1": 3.25, "t2": 0.4, "tauV": 0.625},
            "LLaVA-NeXT-8B": {"t1": 5.03125, "t2": 0.3, "tauV": 0.625}}
COST = {"A": 1, "B": 2, "C": 3}
FAMILIES = ("A", "B", "C")


def f32(x):
    return np.asarray(x, dtype=np.float32)


def margin(v):
    v = np.sort(f32(v))[::-1]
    return float(v[0] - v[1]) if len(v) > 1 else 0.0


def center(v):
    v = f32(v)
    return v - v.mean()


def opt_ids(enc):
    if enc == "Qwen2-VL-7B":
        return {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}
    from transformers import AutoTokenizer
    for mp in ["/data/shunshungu/models/llama3-llava-next-8b-hf",
               "meta-llama/Meta-Llama-3-8B-Instruct"]:
        try:
            tk = AutoTokenizer.from_pretrained(mp)
            return {L: tk.encode(L, add_special_tokens=False)[-1] for L in "ABCDE"}
        except Exception:
            continue
    raise RuntimeError("no LLaVA tokenizer")


def load_rows(enc):
    """Biased_Val MCQ rows carrying every view, as option log-probs + maxp."""
    ids_map = opt_ids(enc)
    rows = []
    for ds in MCQ_DS:
        tsv = LMU / f"{ds}_{enc}_{SPLIT}.tsv"
        dirs = {k: DUMP / f"{enc}-{sfx}" / f"{ds}_{enc}_{SPLIT}"
                for k, sfx in VIEWS.items()}
        if not tsv.exists() or not all(d.exists() for d in dirs.values()):
            print(f"  skip {ds}: missing tsv or view dir")
            continue
        df = pd.read_csv(tsv, sep="\t")
        kept = 0
        for i in range(len(df)):
            opts = [L for L in "ABCDE"
                    if L in df.columns and pd.notna(df.iloc[i].get(L))]
            ans = str(df.iloc[i]["answer"]).strip().upper()
            if len(opts) < 2 or ans not in opts:
                continue
            ids = [ids_map[L] for L in opts]
            try:
                rec = {}
                for k, d in dirs.items():
                    lg = torch.load(d / f"logits_{i}.pt", map_location="cpu")[0]
                    lp = torch.log_softmax(lg.float(), -1)
                    rec[k] = lp[ids].numpy()
                    if k in ("s0", "gray", "strongblur", "centermask"):
                        rec["maxp_" + k] = float(lp.max().exp())
            except Exception:
                continue
            rec["ans"] = opts.index(ans)
            rows.append(rec)
            kept += 1
        print(f"  {ds}: {kept} rows")
    return rows


def predict(r, keep, t1, t2, tauV):
    """Sec. 3 operator restricted to the kept families -> (hit, passes)."""
    state, f1, f2 = center(r["s0"]), 0, 0
    contrasts = []
    if "A" in keep:
        contrasts.append(f32(r["s0"]) - f32(r["blank"]))
    if "B" in keep:
        contrasts.append(f32(r["ans_real"]) - f32(r["ans_blank"]))
    if contrasts and margin(state) <= t1:
        f1 = 1
        lift = contrasts[0]
        for c in contrasts[1:]:
            lift = np.maximum(lift, c)
        state = center(lift)
    if "C" in keep and margin(state) <= t2:
        f2 = 1
        src = min(("gray", "strongblur", "centermask"),
                  key=lambda k: r["maxp_s0"] - r["maxp_" + k])   # argmin-iic
        cand = state + center(f32(r["s0"]) - f32(r[src]))
        if margin(cand) - margin(state) >= tauV:
            state = cand
    step1 = sum(COST[k] for k in keep if k in ("A", "B"))
    return int(np.argmax(state) == r["ans"]), 1 + step1 * f1 + COST["C"] * f2


def main():
    for enc, dep in DEPLOYED.items():
        print(f"\n===== {enc} | {SPLIT} MCQ =====")
        rows = load_rows(enc)
        if not rows:
            print("  no rows"); continue
        t1_rule = float(np.percentile([margin(center(r["s0"])) for r in rows], 95))
        print(f"  n={len(rows)}   t1 by rule (95th pct of base margin) = "
              f"{t1_rule:.4f}   deployed = {dep['t1']}")
        screen = []
        for bits in range(8):
            keep = tuple(f for i, f in enumerate(FAMILIES) if bits >> i & 1)
            out = [predict(r, keep, dep["t1"], dep["t2"], dep["tauV"]) for r in rows]
            acc = 100.0 * sum(h for h, _ in out) / len(out)
            screen.append((round(acc, 2), sum(p for _, p in out) / len(out), keep))
        screen.sort(key=lambda x: (-x[0], x[1]))
        best = screen[0][2]
        for acc, ps, keep in screen:
            star = " <-- screen selects" if keep == best else ""
            print(f"    keep={('+'.join(keep) or 'none'):8} "
                  f"Val MCQ acc={acc:6.2f} passes={ps:.2f}{star}")
        print(f"  => screen keeps {'+'.join(best) if best else 'nothing'}; "
              f"deployed set is A+B+C -> "
              f"{'MATCH' if best == ('A', 'B', 'C') else 'DIFFERS'}")


if __name__ == "__main__":
    main()
