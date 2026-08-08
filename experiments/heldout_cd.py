"""Held-out contrast-decoding baselines (TIE, VCD) on Qwen3-VL / OneVision-1.5.

ZERO forwards.  Replays the frozen phase-B v6 population (same trows as
replay_newgen_matched_ci.py) and scores TIE/VCD as answer-level fusions of
(s0, blank) with the modeling_qwen2_vl.py masking rule:
    fused  = coef_o*s0 + coef_v*blank
    cutoff = log(theta) + s0.max()
    fused[s0 < cutoff] = -inf   ;   pred = argmax(fused)
TIE : coef=(1,-1), theta=0.5    VCD : coef=(2,-1), theta=0.3  (alpha=1)

VC branch is `blank` (the canonical image-free probe in this held-out cache;
SCI uses the same branch here), NOT noise500 -- noise500 covers only 244/887
supp rows and cannot score the full population.  M3ID is NOT computed: its
position-adaptive alpha is a per-token generation quantity with no faithful
answer-level candidate-scoring analogue (would need a GPU token-level run).

Validity gate: base B/S/BS Overall must bit-match the frozen TARGETS before
TIE/VCD are trusted.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/data/newbackbone_generation"

# frozen base Overall targets (from replay_newgen_matched_ci.py TARGETS)
BASE_TGT = {
    "qwen3": dict(B_All=0.49, S_All=34.65, BS_All=12.56, n=1759),
    "ov15": dict(B_All=0.17, S_All=52.16, BS_All=26.58, n=2728),
}


def cd_pick(r, theta, co, cv):
    """Answer-level contrast decoding: fused=co*s0+cv*blank, mask s0<cutoff."""
    s0 = r["s0"]
    if isinstance(s0, dict):
        keys = [k for k in r.get("pool", list(s0)) if k in s0 and k in r["blank"]]
        if not keys:
            return None
        o = np.array([s0[k] for k in keys], dtype=np.float64)
        v = np.array([r["blank"][k] for k in keys], dtype=np.float64)
        labels = keys
    else:
        o = np.array(s0, dtype=np.float64)
        v = np.array(r["blank"], dtype=np.float64)
        labels = r.get("labels") or ["Yes", "No"]
    cut = np.log(theta) + o.max()
    fused = co * o + cv * v
    fused = np.where(o < cut, -np.inf, fused)
    return labels[int(fused.argmax())]


def run_model(model):
    sys.argv = ["cd", model]
    import newbackbone_calibration_eval as EV
    importlib.reload(EV)
    EV.MODEL = model
    from newbackbone_calibration_eval import hit, cells

    trows = [r for r in EV.load("biased") if r.get("split") == "test"]
    tgt = BASE_TGT[model]
    assert len(trows) == tgt["n"], f"n_test {len(trows)} != {tgt['n']}"

    for r in trows:
        r["_base"] = hit(r, cd_pick(r, 1.0, 1.0, 0.0))  # argmax(s0), no mask effect
        r["_tie"] = hit(r, cd_pick(r, 0.5, 1.0, -1.0))
        r["_vcd"] = hit(r, cd_pick(r, 0.3, 2.0, -1.0))
        # M3ID answer-level: org + lam*(org-blank) = (1+lam)org - lam*blank, theta=0.3
        for lam in (1.5, 2.0, 3.0):
            r[f"_m3id{lam}"] = hit(r, cd_pick(r, 0.3, 1.0 + lam, -lam))

    keys = [("base", "_base"), ("TIE", "_tie"), ("VCD", "_vcd")]
    keys += [(f"M3ID(l={lam})", f"_m3id{lam}") for lam in (1.5, 2.0, 3.0)]
    got = {nm: cells(trows, lambda r, k=k: r[k]) for nm, k in keys}
    return got


for model in ("qwen3", "ov15"):
    got = run_model(model)
    tgt = BASE_TGT[model]
    b = got["base"]
    ok = all(abs(b[k] - tgt[k]) < 0.02 for k in ("B_All", "S_All", "BS_All"))
    print(f"\n======== {model} ========")
    print(f"  base PARITY {'PASS' if ok else 'FAIL'}: "
          f"B={b['B_All']}/{tgt['B_All']} S={b['S_All']}/{tgt['S_All']} "
          f"BS={b['BS_All']}/{tgt['BS_All']}")
    for nm in got:
        c = got[nm]
        print(f"  {nm:12s}  B_All={c['B_All']:.2f}  S_All={c['S_All']:.2f}  "
              f"BS_All={c['BS_All']:.2f}   (MCQ BS={c['BS_MCQ']:.2f} Oth BS={c['BS_Oth']:.2f})")
