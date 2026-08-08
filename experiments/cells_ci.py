"""Bootstrap CIs for every reported MAGIC cell, plus paired tests against
the ablation arms. Zero GPU: frozen bundles only.

Reviewers asked for intervals on the S column and on the ablation
deltas, which previously carried point estimates alone.
"""
from __future__ import annotations
import gzip, json, math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import magic as M

OUT = ROOT / "experiments/data/paper_reconciliation"
B = 100_000
SEED = 20260716


def views(row):
    s0, dd, ad, vc = (row["s0"], row["default_delta"],
                      row["answer_delta"], row["selected_vc"])
    return s0, [max(dd[i], ad[i]) for i in range(len(s0))], \
        [s0[i] - vc[i] for i in range(len(s0))]


def hit(row, st):
    return row["labels"][M.argmax_key(st)] == str(row["answer"]).strip().upper()


def arms(row, cfg):
    s0, lift, resid = views(row)
    full = M.run_ladder(s0, lift, resid, cfg)
    r1 = M.run_ladder(s0, lift, resid, M.Ladder(cfg.theta1, M.Rung(-1e9, 0.0)))
    nogate = M.run_ladder(s0, lift, resid, M.Ladder(1e9, M.Rung(1e9, cfg.rung2.tau)))
    return {"magic": hit(row, full), "r1": hit(row, r1),
            "always": hit(row, nogate)}


def boot_ci(vals, seed):
    import numpy as np
    a = np.asarray(vals, dtype=np.float64)
    n = a.size
    rng = np.random.default_rng(seed)
    reps = a[rng.integers(0, n, size=(B, n))].mean(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return round(100 * a.mean(), 2), round(100 * (hi - lo) / 2, 2)


def mcnemar(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    with gzip.open(ROOT / "magic_mcq_bundle.json.gz", "rt") as f:
        bundle = json.load(f)
    rep = {}
    for mk in ("qwen", "llava"):
        cfg = M.LADDER_MCQ[mk]
        rows = bundle["mcq"][mk]
        per = {g: {a: [] for a in ("magic", "r1", "always")} for g in M.GROUPS}
        for row in rows:
            a = arms(row, cfg)
            for g, modes in M.GROUPS.items():
                if row["mode"] in modes:
                    for k, v in a.items():
                        per[g][k].append(v)
        cell = {}
        for g in ("B", "S", "BS"):
            acc, hw = boot_ci(per[g]["magic"], SEED)
            cell[g] = {"magic": acc, "ci95_halfwidth": hw,
                       "n": len(per[g]["magic"])}
        # paired tests on the BS column
        pt = {}
        for arm in ("r1", "always"):
            w2r = sum((not x) and y for x, y in
                      zip(per["BS"][arm], per["BS"]["magic"]))
            r2w = sum(x and (not y) for x, y in
                      zip(per["BS"][arm], per["BS"]["magic"]))
            pt[f"magic_vs_{arm}"] = {"w2r": w2r, "r2w": r2w,
                                    "p": round(mcnemar(w2r, r2w), 4)}
        rep[mk] = {"cells": cell, "paired": pt}
        print(mk, json.dumps(rep[mk], indent=1))
    (OUT / "cell_cis.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
