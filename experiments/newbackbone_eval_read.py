"""V6 read: FULL-STRENGTH ladder (true iic) vs base/SCI3/SCI5/SCI7/gm3,
all symmetrically natural-gated.  SCI7 covers mcq+binary only (open rows
lack noise400/tcf3 views — its Oth/All cells are 'partial, no ViLP',
printed separately).  Usage: --model qwen3|ov15 --run"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import newbackbone_calibration_eval as EV  # noqa: E402
import newbackbone_symmetric_eval as V2  # noqa: E402
from newbackbone_calibration_eval import (  # noqa: E402
    load, hit, cells, m, cent, add, sub, maxi, pick, OUT)
from newbackbone_symmetric_eval import merge_nat, sci5_any, natural_delta, SCI_GRID  # noqa: E402

MODEL = "ov15" if "ov15" in sys.argv else ("onevision" if "onevision" in sys.argv else "qwen3")
EV.MODEL = MODEL
V2.MODEL = MODEL

V6 = {}
p6 = OUT / f"b3v6_{MODEL}.jsonl"
with p6.open() as f:
    for line in f:
        if line.strip():
            s = json.loads(line)
            V6[(s["surface"], s["ds"], s["index"])] = s


def v6of(rec):
    return V6.get((rec.get("surface") or ("biased" if rec.get("split") else "?"),
                   rec["ds"], rec["index"]))


def evidence6(rec):
    """Full-strength evidence: true iic (maxp) source selection."""
    g = lambda k: (rec[k] if isinstance(rec[k], dict)  # noqa: E731
                   else torch.tensor(rec[k], dtype=torch.float64))
    s0, blank = g("s0"), g("blank")
    ar, ab = g("ans_real"), g("ans_blank")
    base = cent(s0)
    clean2 = cent(maxi(sub(s0, blank), sub(ar, ab)))
    agree = float(pick(sub(s0, blank), rec) == pick(sub(ar, ab), rec))
    srcs = {k: g(k) for k in ("gray", "strongblur", "centermask")}
    v6 = v6of(rec)
    if v6 and "maxp_orig" in v6:
        iic = {k: v6["maxp_orig"] - v6["maxp_" + k] for k in srcs}
        src = min(iic, key=iic.get)  # true argmin-iic
    else:
        src = min(srcs, key=lambda k: m(srcs[k]))  # proxy fallback (open rows)
    resid = cent(sub(s0, srcs[src]))
    return base, clean2, agree, resid


def ladder6(rec, C):
    base, clean2, agree, resid = evidence6(rec)
    state = base
    if m(state) <= C["t1"]:
        state = clean2 if (m(clean2) + agree) >= C["tauT"] else base
    if m(state) <= C["t2"]:
        cand = add(state, resid)
        if m(cand) - m(state) >= C["tauV"]:
            state = cand
    return pick(state, rec), pick(base, rec)


def derive6(nat, t1p, tTp):
    ev = [evidence6(r) for r in nat]
    m0 = [m(e[0]) for e in ev]
    t1 = float(np.percentile(m0, t1p))
    adm = [m(e[1]) + e[2] for e in ev if m(e[0]) <= t1]
    tauT = float(np.percentile(adm, tTp)) if adm else 0
    post = []
    for e in ev:
        if m(e[0]) <= t1:
            st = e[1] if m(e[1]) + e[2] >= tauT else e[0]
            post.append(m(st))
    t2 = float(np.percentile(post, 30.0)) if post else 0
    gains = []
    for e in ev:
        if m(e[0]) <= t1:
            st = e[1] if m(e[1]) + e[2] >= tauT else e[0]
            if m(st) <= t2:
                gains.append(m(add(st, e[3])) - m(st))
    tauV = float(np.percentile(gains, 70.0)) if gains else 0.625
    return dict(t1=t1, tauT=tauT, t2=t2, tauV=tauV)


def sci_n(rec, P, branches):
    """Generic SCI on raw views. branches=('sci3'|'sci7')."""
    if rec["typ"] == "open":
        return None if branches == "sci7" else V2.sci5_any(rec, dict(P), None) if False else _sci3_open(rec, P)
    v6 = v6of(rec)
    need = ["s0_raw", "blank_raw"]
    if branches == "sci7" and (not v6 or "noise400_raw" not in v6):
        return None
    g = lambda k: torch.tensor(rec[k], dtype=torch.float64)  # noqa: E731
    o, v1 = g("s0_raw"), g("blank_raw")
    t1 = g("tcf1_raw") if "tcf1_raw" in rec else g("tcf1v_raw")
    labels = rec.get("labels") or ["Yes", "No"]
    if branches == "sci3":
        consistent = torch.stack([o, t1]).max(0).values / P["gamma"]
        unbiased = (o - v1) / P["beta"]
    else:
        v2_ = g("noise500_raw")
        v3 = torch.tensor(v6["noise400_raw"], dtype=torch.float64)
        t2_ = g("tcf2_raw") if "tcf2_raw" in rec else g("tcf2v_raw")
        t3 = torch.tensor(v6["tcf3_raw"], dtype=torch.float64)
        consistent = torch.stack([o, t1, t2_, t3]).max(0).values / P["gamma"]
        unbiased = (o - (v1 + v2_ + v3) / 3.0) / P["beta"]
    final = consistent + unbiased
    cutoff = float(np.log(P["theta"])) + float(consistent.max())
    final = final.masked_fill(consistent < cutoff, -float("inf"))
    return labels[int(final.argmax())]


def _sci3_open(rec, P):
    pool = [c for c in rec["pool"] if c in rec["s0"] and c in rec["blank"]
            and c in rec["ans_real"]]
    if not pool:
        return None
    t = lambda d: torch.tensor([d[c] for c in pool], dtype=torch.float64)  # noqa
    o, v1, t1 = t(rec["s0"]), t(rec["blank"]), t(rec["ans_real"])
    consistent = torch.stack([o, t1]).max(0).values / P["gamma"]
    unbiased = (o - v1) / P["beta"]
    final = consistent + unbiased
    cutoff = float(np.log(P["theta"])) + float(consistent.max())
    final = final.masked_fill(consistent < cutoff, -float("inf"))
    return pool[int(final.argmax())]


def main():
    nat = merge_nat(load("natural_val"))
    band = 100.0 / len(nat)
    vrows = [r for r in load("biased") if r.get("split") == "val"]
    trows = [r for r in load("biased") if r.get("split") == "test"]
    supp = {}
    sp = OUT / f"b3supp_{MODEL}.jsonl"
    with sp.open() as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                supp[s["index"]] = s
    print(f"{MODEL} V6: nat={len(nat)} val={len(vrows)} test={len(trows)}")

    def calibrate(fn_family, name):
        best, bacc = None, -1
        for P in SCI_GRID:
            nd, h, hm, n = natural_delta(nat, lambda r, P=P: fn_family(r, P))
            if nd < -band:
                continue
            va = np.mean([hit(r, c) for r in vrows
                          for c in [fn_family(r, P)] if c is not None])
            if va > bacc:
                bacc, best = va, P
        print(f"  {name} selected: {best} (valAcc {100*bacc:.2f})" if best
              else f"  {name}: no natural-safe config")
        return best

    sci3P = calibrate(lambda r, P: sci_n(r, P, "sci3"), "SCI3")
    sci5P = calibrate(lambda r, P: sci5_any(r, P), "SCI5")
    sci7P = calibrate(lambda r, P: sci_n(r, P, "sci7"), "SCI7")

    bestL, bacc = None, -1
    for tp in (5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0):
        for pct in (95.0, 85.0, 70.0, 50.0):
            C = derive6(nat, pct, tp)
            hl = hb = helped = harmed = 0
            for r in nat:
                p, bp = ladder6(r, C)
                a, b = hit(r, p), hit(r, bp)
                hl += a; hb += b
                helped += (a and not b); harmed += (b and not a)
            if 100.0 * (hl - hb) / len(nat) < -band:
                continue
            va = np.mean([hit(r, ladder6(r, C)[0]) for r in vrows])
            if va > bacc:
                bacc, bestL = va, C
    if bestL is None:
        print("  LADDER-full: NO natural-safe config in the widened grid — B2-FAIL for v6/ov15 (report honestly)")
        return
    print(f"  LADDER-full selected: {bestL} (valAcc {100*bacc:.2f})")

    gm3P = None
    best_ga = -1
    for lam in (0.5, 1.0, 2.0):
        def gp(r, lam=lam):
            s0 = r["s0"]
            if isinstance(s0, dict):
                mv = sorted(s0.values(), reverse=True)
                mg = mv[0] - mv[1] if len(mv) > 1 else 0
                if mg <= bestL["t1"]:
                    st = {k: s0[k] + lam * (s0[k] - r["blank"][k]) for k in s0}
                    return max(st, key=st.get)
                return max(s0, key=s0.get)
            t = torch.tensor(s0, dtype=torch.float64)
            b = torch.tensor(r["blank"], dtype=torch.float64)
            tc = t - t.mean()
            mg = float(torch.topk(tc, 2).values.diff().abs()) if tc.numel() > 1 else 0
            labels = r.get("labels") or ["Yes", "No"]
            return (labels[int((t + lam * (t - b)).argmax())] if mg <= bestL["t1"]
                    else labels[int(t.argmax())])
        nd, h, hm, n = natural_delta(nat, gp)
        if nd < -band:
            continue
        va = np.mean([hit(r, gp(r)) for r in vrows])
        if va > best_ga:
            best_ga, gm3P = va, lam
    print(f"  gm3 lambda: {gm3P}")

    if "--run" not in sys.argv:
        return
    for r in trows:
        p, bp = ladder6(r, bestL)
        r["_lad"], r["_base"] = hit(r, p), hit(r, bp)
        c3 = sci_n(r, sci3P, "sci3") if sci3P else None
        r["_s3"] = hit(r, c3) if c3 is not None else r["_base"]
        c5 = sci5_any(r, sci5P or SCI_GRID[0], supp)
        r["_s5"] = hit(r, c5) if c5 is not None else r["_base"]
        c7 = sci_n(r, sci7P, "sci7") if sci7P else None
        r["_s7"] = hit(r, c7) if c7 is not None else None
    for nm, k in (("base", "_base"), ("SCI3", "_s3"), ("SCI5", "_s5"),
                  ("LADDER-full", "_lad")):
        print(f"  {nm:<12} " + " ".join(
            f"{kk}={vv}" for kk, vv in cells(trows, lambda r, k=k: r[k]).items()))
    s7rows = [r for r in trows if r["_s7"] is not None]
    print(f"  SCI7 (mcq+binary only, n={len(s7rows)}): " + " ".join(
        f"{kk}={vv}" for kk, vv in cells(s7rows, lambda r: r["_s7"]).items()))
    print(f"  LADDER on same subset: " + " ".join(
        f"{kk}={vv}" for kk, vv in cells(s7rows, lambda r: r["_lad"]).items()))
    (OUT / f"b3v6_test_readout_{MODEL}.json").write_text(json.dumps(
        {"ladder": bestL, "sci3": sci3P, "sci5": sci5P, "sci7": sci7P,
         "gm3_lambda": gm3P}))
    print("wrote b3v6_test_readout")


if __name__ == "__main__":
    main()
