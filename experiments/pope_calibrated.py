"""POPE under the method's OWN contract: thresholds re-fitted on a POPE
validation split, one test read. Zero GPU.

Pre-registration: experiments/data/pope_calibrated/PREREG.md

The paper's Limitations state that the three thresholds are fitted per
target distribution on a validation split. The currently reported POPE
number instead freezes the DRBench constants, which is a stress test and
not the specified usage. This script reads the specified usage, on both
backbones, with base / SCI5 / frozen-MAGIC as comparators on the same rows.

Usage: python3 experiments/pope_calibrated.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import magic as M  # noqa: E402

POPE = ROOT / "experiments/data/pope_external_probe"
OUT = ROOT / "experiments/data/pope_calibrated"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["Yes", "No"]
GRID_T1 = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
GRID_T2 = [0.0, 0.2, 0.3, 0.4, 0.6, 1.0]
GRID_TV = [0.0, 0.25, 0.5, 0.625, 1.0]

# DRBench-frozen constants (what the paper currently reports on POPE)
FROZEN = {"llava": (5.03125, 0.5, 0.375), "qwen2": (3.25, 0.5, -0.220826)}
SCI5 = dict(beta=0.2, gamma=2.0, theta=0.3)


def gold(r):
    return 0 if str(r["answer"]).strip().lower() == "yes" else 1


def magic_pred(r, t1, t2, tv):
    s0, bl, ar, ab, gr = r["s0"], r["blank"], r["ar"], r["ab"], r["gray"]
    lift = [max(s0[i] - bl[i], ar[i] - ab[i]) for i in range(2)]
    env = [max(s0[i], ar[i]) for i in range(2)]
    resid = [env[i] - gr[i] for i in range(2)]
    state = list(s0)
    if M.margin(state) <= t1:
        state = lift
    if M.margin(state) <= t2:
        cand = M._add(state, resid)
        if M.margin(cand) - M.margin(state) >= tv:
            state = cand
    return M.argmax_key(state)


def sci5_pred(r):
    """SCI's published fusion on the cached raw logits."""
    o, t1_, t2_ = r["s0_raw"], r["ar_raw"], r["tcf2_raw"]
    v1, v2 = r["blank_raw"], r["noise500_raw"]
    cons = [max(o[i], t1_[i], t2_[i]) / SCI5["gamma"] for i in range(2)]
    unb = [(o[i] - 0.5 * (v1[i] + v2[i])) / SCI5["beta"] for i in range(2)]
    fused = [cons[i] + unb[i] for i in range(2)]
    cut = math.log(SCI5["theta"]) + max(cons)
    fused = [f if cons[i] >= cut else -math.inf for i, f in enumerate(fused)]
    return M.argmax_key(fused)


def acc(rows, pred):
    return 100.0 * sum(pred(r) == gold(r) for r in rows) / len(rows)


def mcnemar(rows, pa, pb):
    """exact two-sided McNemar on discordant pairs (b = a-wrong->b-right)."""
    b = c = 0
    for r in rows:
        g = gold(r)
        ha, hb = pa(r) == g, pb(r) == g
        if hb and not ha:
            b += 1
        elif ha and not hb:
            c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return b, c, min(1.0, p)


def split(rows, seed=0):
    """stratified 50/50 by (category, answer), deterministic."""
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[(r["category"], r["answer"])].append(i)
    val, test = [], []
    for key in sorted(buckets):
        idx = sorted(buckets[key])
        # deterministic interleave = exact stratification, no RNG dependence
        val += idx[0::2]
        test += idx[1::2]
    return [rows[i] for i in sorted(val)], [rows[i] for i in sorted(test)]


def main():
    report = {}
    for mk, fname in (("qwen2", "pope_qwen2.jsonl"), ("llava", "pope_llava.jsonl")):
        rows = [json.loads(l) for l in (POPE / fname).open() if l.strip()]
        val, test = split(rows)
        print(f"== {mk}: n={len(rows)} -> val {len(val)} / test {len(test)}", flush=True)

        # ---- fit on VAL only -------------------------------------------
        best, best_acc = None, -1.0
        for t1 in GRID_T1:
            for t2 in GRID_T2:
                for tv in GRID_TV:
                    a = acc(val, lambda r, p=(t1, t2, tv): magic_pred(r, *p))
                    if a > best_acc + 1e-9 or (abs(a - best_acc) < 1e-9 and best and t1 < best[0]):
                        best, best_acc = (t1, t2, tv), a
        print(f"   fitted on val: t1={best[0]} t2={best[1]} tauV={best[2]}  "
              f"(val acc {best_acc:.2f})", flush=True)

        # ---- ONE test read ---------------------------------------------
        f = FROZEN[mk]
        base_p = lambda r: M.argmax_key(r["s0"])
        cal_p = lambda r: magic_pred(r, *best)
        frz_p = lambda r: magic_pred(r, *f)
        res = {
            "n_val": len(val), "n_test": len(test), "fitted": best,
            "val_acc": round(best_acc, 2),
            "base": round(acc(test, base_p), 2),
            "SCI5": round(acc(test, sci5_pred), 2),
            "MAGIC_calibrated": round(acc(test, cal_p), 2),
            "MAGIC_frozen": round(acc(test, frz_p), 2),
        }
        for name, p in (("vs_base", base_p), ("vs_SCI5", sci5_pred)):
            b, c, pv = mcnemar(test, p, cal_p)
            res[f"cal_{name}"] = {"delta": round(res["MAGIC_calibrated"] -
                                                 (res["base"] if name == "vs_base" else res["SCI5"]), 2),
                                  "flips": f"{b}:{c}", "p": round(pv, 4)}
        # per-category breakdown of the calibrated arm
        res["by_category"] = {}
        for cat in sorted({r["category"] for r in test}):
            sub = [r for r in test if r["category"] == cat]
            res["by_category"][cat] = {
                "base": round(acc(sub, base_p), 2),
                "MAGIC": round(acc(sub, cal_p), 2),
                "n": len(sub)}
        # mean scoring passes of the calibrated arm
        f1 = f2 = 0
        for r in test:
            s0, bl, ar, ab = r["s0"], r["blank"], r["ar"], r["ab"]
            lift = [max(s0[i] - bl[i], ar[i] - ab[i]) for i in range(2)]
            st = list(s0)
            if M.margin(st) <= best[0]:
                st = lift; f1 += 1
            if M.margin(st) <= best[1]:
                f2 += 1
        res["mean_passes"] = round(1 + 3 * f1 / len(test) + 3 * f2 / len(test), 2)
        report[mk] = res
        print(json.dumps(res, indent=2), flush=True)

    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print("\nwrote", OUT / "results.json")


if __name__ == "__main__":
    main()
