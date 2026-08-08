"""Frozen replay of Table 4's MAGIC rows on the two newer backbones.

The counterpart of `magic.py` (which does this for Qwen2-VL / LLaVA-NeXT).
Zero GPU: reads the frozen phase-B caches and reproduces the deployed cells
bit-exactly, including the two decisions that define the deployed config.

Decision path, in the order it runs -- nothing here is hand-set:

  1. t1  = the 95th percentile of the base decision margin on the BIASED
           validation split.  Label-free (base scores only, no labels, no
           branch set), and the same rule tab:main uses.  This is what SCI's
           own tools/validation.py does too: fit on Biased_Val, read on
           Biased_Test.
  2. branch set = argmax on the BIASED validation split over the 2^3 = 8
           configurations the three perturbation families generate, each kept
           or dropped independently, ties broken toward fewer passes.  This is
           the calibration screen of Sec. 3.4, whose object is the family:

             A  original-wording contrast  D_q  = Z(v,q) - Z(v_blank,q)    1 pass
             B  rephrased contrast         D_q' = Z(v,q') - Z(v_blank,q')  2 passes
             C  visual-source residual     D_2  (Eq. 5-6)                  3 passes

           Dropping both A and B leaves the debiasing step with nothing to
           build, so Eq. 4 never fires -- the "w/o debiasing" arm of the
           ablation.  Dropping all three is the base model (Sec. 3.1).  Both
           are legitimate members of the space and are enumerated.
  3. Test  = one read of the selected configuration.

t2 and tauV are the frozen per-format constants in
experiments/data/solver_calibration/solved_{model}.json; they are only consulted when the
screen keeps the restoration step.

The operator itself is `newbackbone_eval_read.evidence6` plus the two-gate ladder
of Sec. 3.2-3.3 -- no admission gate, no reweighted single-branch variant.

Usage:  python experiments/newgen_replay.py
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "experiments/data/newbackbone_generation"
SOLVER = ROOT / "experiments/data/solver_calibration"

# Frozen Test cells of the deployed configuration (paper Table 4).
FROZEN = {
    "qwen3": {"t1": 11.71, "keep": ("B",),
              "cells": {"B": 28.98, "S": 49.13, "BS": 31.84}, "passes": 2.90},
    "ov15": {"t1": 19.55, "keep": ("A", "C"),
             "cells": {"B": 27.49, "S": 56.47, "BS": 42.16}, "passes": 3.95},
}
# extra forward passes each perturbation family costs when its step fires
COST = {"A": 1, "B": 2, "C": 3}
FAMILIES = ("A", "B", "C")


def _load(model):
    sys.argv = ["replay", model]
    import newbackbone_calibration_eval as EV
    import newbackbone_symmetric_eval as V2
    import newbackbone_eval_read as V6
    EV, V2, V6 = (importlib.reload(x) for x in (EV, V2, V6))
    EV.MODEL = V2.MODEL = V6.MODEL = model
    return EV, V6


def _get(rec, key):
    v = rec[key]
    return v if isinstance(v, dict) else torch.tensor(v, dtype=torch.float64)


def predict(rec, keep, t1, C, V6):
    """The Sec. 3 operator restricted to the kept perturbation families.

    `keep` is a subset of ("A","B","C").  Returns (answer, forward passes).
    """
    m, pick, add, sub, cent, maxi = V6.m, V6.pick, V6.add, V6.sub, V6.cent, V6.maxi
    s0 = _get(rec, "s0")
    state, f1, f2 = cent(s0), 0, 0

    contrasts = []
    if "A" in keep:
        contrasts.append(sub(s0, _get(rec, "blank")))                  # D_q
    if "B" in keep:
        contrasts.append(sub(_get(rec, "ans_real"),
                             _get(rec, "ans_blank")))                  # D_q'
    if contrasts and m(state) <= t1:                            # allocate step 1
        f1 = 1
        lift = contrasts[0]
        for c in contrasts[1:]:
            lift = maxi(lift, c)                                # elementwise max
        state = cent(lift)

    if "C" in keep and m(state) <= C["t2"]:                     # allocate step 2
        f2 = 1
        srcs = {k: _get(rec, k) for k in ("gray", "strongblur", "centermask")}
        v6 = V6.v6of(rec)
        if v6 and "maxp_orig" in v6:                            # argmin-iic route
            src = min(srcs, key=lambda k: v6["maxp_orig"] - v6["maxp_" + k])
        else:
            src = min(srcs, key=lambda k: m(srcs[k]))
        cand = add(state, cent(sub(s0, srcs[src])))
        if m(cand) - m(state) >= C["tauV"]:                     # commit
            state = cand

    step1_cost = sum(COST[k] for k in keep if k in ("A", "B"))
    return pick(state, rec), 1 + step1_cost * f1 + COST["C"] * f2


def run(model):
    EV, V6 = _load(model)
    C = json.loads((SOLVER / f"solved_{model}.json").read_text())["C"]
    rows = [json.loads(l) for l in (CACHE / f"b3prep_{model}.jsonl").open()
            if l.strip()]
    val = [r for r in rows if r.get("surface") == "biased"
           and r.get("split") == "val"]

    # 1. label-free gate from the biased validation split
    t1 = float(np.percentile([V6.m(V6.cent(_get(r, "s0"))) for r in val], 95))

    # 2. calibration screen: every subset of the three families, Val only
    screen = []
    for bits in range(8):
        keep = tuple(f for i, f in enumerate(FAMILIES) if bits >> i & 1)
        out = [predict(r, keep, t1, C, V6) for r in val]
        acc = 100.0 * sum(EV.hit(r, p) for r, (p, _) in zip(val, out)) / len(val)
        screen.append((round(acc, 2), sum(x[1] for x in out) / len(out), keep))
    screen.sort(key=lambda x: (-x[0], x[1]))          # accuracy, then cost
    best = screen[0][2]

    # 3. one Test read of the selected configuration
    test = [r for r in EV.load("biased") if r.get("split") == "test"]
    cells = EV.cells(test, lambda r: EV.hit(r, predict(r, best, t1, C, V6)[0]))
    passes = sum(predict(r, best, t1, C, V6)[1] for r in test) / len(test)
    got = {"B": cells["B_All"], "S": cells["S_All"], "BS": cells["BS_All"]}

    want = FROZEN[model]
    ok = (round(t1, 2) == want["t1"] and best == want["keep"]
          and got == want["cells"] and round(passes, 2) == want["passes"])
    print(f"{model:6} t1={t1:6.2f} (biased-Val p95)   screen keeps "
          f"{'+'.join(best) if best else 'nothing (base)'}")
    for acc, ps, keep in screen:
        star = " <-- selected" if keep == best else ""
        print(f"         Val keep={('+'.join(keep) or 'none'):8} "
              f"BS={acc:6.2f} passes={ps:.2f}{star}")
    print(f"       Test n={len(test)}  B={got['B']:.2f} S={got['S']:.2f} "
          f"BS={got['BS']:.2f} passes={passes:.2f}   "
          f"parity: {'OK' if ok else 'FAIL want ' + str(want)}")
    return ok


def main():
    ok = all(run(mk) for mk in ("qwen3", "ov15"))
    print("ALL PARITY OK" if ok else "PARITY MISMATCH")


if __name__ == "__main__":
    main()
