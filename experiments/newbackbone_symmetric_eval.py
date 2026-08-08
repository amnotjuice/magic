"""B3-v2: SYMMETRIC-contract calibration + authorized Test read.
Both methods: (i) must pass natural-Val no-harm (tie band), (ii) among
passing configs, selected by biased-Val accuracy.  SCI5 raw-space on
mcq/binary (natural raws from b3nat_raw + b3nat_sci), logprob-space on
open (both splits, consistent).  Usage: --model qwen3|ov15 --run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import newbackbone_calibration_eval as EV  # noqa: E402
from newbackbone_calibration_eval import (  # noqa: E402
    load, ladder_run, derive_constants, hit, cells, gated_m3id, MODEL, OUT)

BAND_KEY = None
SCI_GRID = [dict(beta=b, alpha=1.0, gamma=g, theta=t)
            for b in (0.1, 0.2) for g in (1.5, 2.0, 2.5)
            for t in (0.2, 0.3, 0.5, 0.7)]


def merge_nat(nat):
    sci, raw = {}, {}
    for fn, d in ((f"b3nat_sci_{MODEL}.jsonl", sci), (f"b3nat_raw_{MODEL}.jsonl", raw)):
        p = OUT / fn
        if p.exists():
            with p.open() as f:
                for line in f:
                    if line.strip():
                        s = json.loads(line)
                        d[(s["ds"], s["index"])] = s
    out = []
    for r in nat:
        k = (r["ds"], r["index"])
        if k in sci:
            r = {**r, **{kk: vv for kk, vv in sci[k].items() if kk not in ("ds", "index")}}
        if k in raw:
            r = {**r, **{kk: vv for kk, vv in raw[k].items() if kk not in ("ds", "index")}}
        out.append(r)
    return out


def sci5_any(rec, P, supp=None):
    """SCI5 on any row: raw-space for mcq/binary, logprob dicts for open."""
    if rec["typ"] == "open":
        if rec.get("split") is not None:  # biased open: use b3supp via EV.sci5
            return EV.sci5(rec, P, supp)
        if "noise500" not in rec or "tcf1v" not in rec:
            return None
        pool = [c for c in rec["pool"] if c in rec["noise500"] and c in rec["tcf1v"] and c in rec["tcf2v"]]
        if len(pool) < 1:
            return None  # 31/3315 rows had pool/supp key drift (dup-window); disclosed
        t = lambda d: torch.tensor([d[c] for c in pool], dtype=torch.float64)  # noqa
        o, v1, v2 = t(rec["s0"]), t(rec["blank"]), t(rec["noise500"])
        t1, t2 = t(rec["tcf1v"]), t(rec["tcf2v"])
        labels = pool
    else:
        need = ("s0_raw", "blank_raw", "noise500_raw")
        if not all(k in rec for k in need):
            return None
        g = lambda k: torch.tensor(rec[k], dtype=torch.float64)  # noqa
        o, v1, v2 = g("s0_raw"), g("blank_raw"), g("noise500_raw")
        t1 = g("tcf1_raw") if "tcf1_raw" in rec else g("tcf1v_raw")
        t2 = g("tcf2_raw") if "tcf2_raw" in rec else g("tcf2v_raw")
        labels = rec.get("labels") or ["Yes", "No"]
    consistent = torch.stack([o, t1, t2]).max(0).values / P["gamma"]
    unbiased = (o - (v1 + v2) / 2.0) / P["beta"]
    final = consistent + unbiased
    cutoff = float(np.log(P["theta"])) + float(consistent.max())
    final = final.masked_fill(consistent < cutoff, -float("inf"))
    return labels[int(final.argmax())]


def natural_delta(nat, fn):
    helped = harmed = n = 0
    for r in nat:
        c = fn(r)
        if c is None:
            continue
        n += 1
        b = r["labels"][int(torch.tensor(r["s0"]).argmax())] if not isinstance(r["s0"], dict) \
            else max(r["s0"], key=r["s0"].get)
        hb, hf = hit(r, b), hit(r, c)
        helped += (hf and not hb); harmed += (hb and not hf)
    return (100.0 * (helped - harmed) / max(n, 1)), helped, harmed, n


def main():
    nat = merge_nat(load("natural_val"))
    band = 100.0 / len(nat)
    vrows = [r for r in load("biased") if r.get("split") == "val"]
    trows = [r for r in load("biased") if r.get("split") == "test"]
    supp = {}
    sp = OUT / f"b3supp_{MODEL}.jsonl"
    if sp.exists():
        with sp.open() as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    supp[s["index"]] = s
    print(f"{MODEL}: nat={len(nat)} val={len(vrows)} test={len(trows)} band=±{band:.3f}")

    # --- SCI5 symmetric calibration ---
    best_sci, best_acc = None, -1
    for P in SCI_GRID:
        nd, h, hm, n = natural_delta(nat, lambda r, P=P: sci5_any(r, P))
        if nd < -band:
            continue
        va = np.mean([hit(r, c) for r in vrows
                      for c in [sci5_any(r, P, supp)] if c is not None])
        print(f"  SCI5 {P}: natΔ={nd:+.3f}({h}/{hm},n={n}) valAcc={100*va:.2f}  OK")
        if va > best_acc:
            best_acc, best_sci = va, P
    if best_sci is None:
        print("  SCI5: NO config passes natural no-harm — its unconstrained Test read stands as natural-unconstrained (disclosed)")
    else:
        print(f"  SCI5 selected: {best_sci} (valAcc {100*best_acc:.2f})")

    # --- ladder symmetric calibration (power among no-harm) ---
    best_lad, best_lacc = None, -1
    for tp in (5.0, 10.0, 20.0, 30.0, 40.0, 50.0):
        for pct in (95.0, 85.0, 70.0):
            C = derive_constants(nat, pct, tp)
            hl = hb = helped = harmed = 0
            for r in nat:
                p, bp = ladder_run(r, C)
                a, b = hit(r, p), hit(r, bp)
                hl += a; hb += b
                helped += (a and not b); harmed += (b and not a)
            nd = 100.0 * (hl - hb) / len(nat)
            if nd < -band:
                continue
            va = np.mean([hit(r, ladder_run(r, C)[0]) for r in vrows])
            if va > best_lacc:
                best_lacc, best_lad = va, dict(C, tauT_pct=tp)
    print(f"  LADDER selected: {best_lad} (valAcc {100*best_lacc:.2f})")

    # --- authorized Test read ---
    if "--run" not in sys.argv:
        print("(calibration only; add --run for the authorized Test read)")
        return
    # gm3 symmetric: lambda under natural no-harm, then Val power
    def gm3_pick(r, lam, C):
        import torch as T
        s0 = r["s0"]
        if isinstance(s0, dict):
            mvals = sorted(s0.values(), reverse=True)
            mg = mvals[0]-mvals[1] if len(mvals) > 1 else 0
            if mg <= C["t1"]:
                st = {k: s0[k]+lam*(s0[k]-r["blank"][k]) for k in s0}
                return max(st, key=st.get)
            return max(s0, key=s0.get)
        t = T.tensor(s0, dtype=T.float64); b = T.tensor(r["blank"], dtype=T.float64)
        tc = t - t.mean()
        mg = float(T.topk(tc, 2).values.diff().abs()) if tc.numel() > 1 else 0
        labels = r.get("labels") or ["Yes", "No"]
        return labels[int((t+lam*(t-b)).argmax())] if mg <= C["t1"] else labels[int(t.argmax())]
    best_gl, best_ga = None, -1
    for lam in (0.5, 1.0, 2.0):
        nd, h, hm, n = natural_delta(nat, lambda r, lam=lam: gm3_pick(r, lam, best_lad))
        if nd < -band:
            continue
        va = np.mean([hit(r, gm3_pick(r, lam, best_lad)) for r in vrows])
        if va > best_ga:
            best_ga, best_gl = va, lam
    print(f"  gm3 lambda selected (natural-gated): {best_gl}")
    for r in trows:
        p, bp = ladder_run(r, best_lad)
        r["_lad"], r["_base"] = hit(r, p), hit(r, bp)
        r["_gm3"] = hit(r, gm3_pick(r, best_gl if best_gl else 1.0, best_lad))
        c = sci5_any(r, best_sci or SCI_GRID[0], supp)
        r["_sci"] = hit(r, c) if c is not None else r["_base"]
    for nm, k in (("base", "_base"), ("SCI5-v2", "_sci"),
                  ("gated-M3ID", "_gm3"), ("LADDER-v2", "_lad")):
        print(f"  {nm:<11} " + " ".join(f"{kk}={vv}" for kk, vv in cells(trows, lambda r, k=k: r[k]).items()))
    (OUT / f"b3v2_test_readout_{MODEL}.json").write_text(json.dumps(
        {"ladder": best_lad, "sci5": best_sci,
         "cells": {nm: cells(trows, lambda r, k=k: r[k])
                   for nm, k in (("base", "_base"), ("sci5", "_sci"),
                                  ("gm3", "_gm3"), ("ladder", "_lad"))}}))
    print("wrote b3v2_test_readout")


if __name__ == "__main__":
    main()
