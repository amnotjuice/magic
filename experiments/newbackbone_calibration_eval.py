"""Phase-B B2 (calibration) + B3 (evaluation) on the b3prep evidence.
Pre-registration: experiments/data/newbackbone_generation/FINDINGS.md.

B2 (Val-only): t1 percentile chosen by natural_val no-harm sweep (labels
on Val = legal calibration); tau_T/t2/tau_V at the ledger's label-free
quantile positions on the model's own distributions.
B3: one pre-registered Test read on the B1 biased Test rows:
  ladder | SCI5-reimpl (raw logits; params = the repo's two configured
  SCI5 sets, Val-selected, disclosed) | gated-M3ID(lam=2) | base.
9-cell = B/S/BS x MCQ/Oth/All (modes: B=vcf_only+both, S=tcf_only+both).

Usage: --model qwen3|ov15  --calibrate | --test --confirm
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import natural_mcq_eval_qwen as NAT  # noqa: E402
from open_ended_scoring import paper_hit  # noqa: E402

OUT = ROOT / "experiments/data/newbackbone_generation"
MODEL = "ov15" if "ov15" in sys.argv else ("onevision" if "onevision" in sys.argv else "qwen3")
QPOS = dict(tauT=5.0, t2=30.0, tauV=70.0)  # ledger positions (t1 via sweep)
SCI5_PARAMS = [dict(beta=0.2, alpha=1.0, gamma=2.0, theta=0.3),
               dict(beta=0.1, alpha=1.0, gamma=2.5, theta=0.2)]
CAL = OUT / f"b2_constants_{MODEL}.json"


def load(surface=None):
    recs = []
    with (OUT / f"b3prep_{MODEL}.jsonl").open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if surface is None or r["surface"] == surface:
                    recs.append(r)
    v2p = OUT / f"b1v2_subsets_{MODEL}.json"
    if v2p.exists():  # fair protocol v3: retag/drop biased rows per B1-v2
        tag = {(x["ds"], x["index"]): x for x in json.loads(v2p.read_text())}
        out = []
        for r in recs:
            if r.get("surface") == "biased" or r.get("split") is not None:
                t = tag.get((r["ds"], r["index"]))
                if t is None:
                    continue
                r = {**r, "split": t["split"], "mode": t["mode"]}
            out.append(r)
        recs = out
    return recs


def m(x):
    if isinstance(x, dict):
        v = sorted(x.values(), reverse=True)
        return float(v[0] - v[1]) if len(v) > 1 else 0.0
    t = torch.tensor(x, dtype=torch.float64)
    return float(torch.topk(t, 2).values.diff().abs()) if t.numel() > 1 else 0.0


def cent(x):
    if isinstance(x, dict):
        mu = sum(x.values()) / len(x)
        return {k: v - mu for k, v in x.items()}
    t = torch.tensor(x, dtype=torch.float64)
    return t - t.mean()


def add(a, b):
    if isinstance(a, dict):
        return {k: a[k] + b[k] for k in a}
    return a + b


def sub(a, b):
    if isinstance(a, dict):
        return {k: a[k] - b[k] for k in a}
    return a - b


def maxi(a, b):
    if isinstance(a, dict):
        return {k: max(a[k], b[k]) for k in a}
    return torch.maximum(a, b)


def pick(x, rec):
    if isinstance(x, dict):
        return max(x, key=x.get)
    labels = rec.get("labels") or ["Yes", "No"]
    return labels[int(torch.tensor(x).argmax())]


def hit(rec, choice):
    if rec["typ"] == "open":
        from numword_scoring import fair_hit  # fair protocol v3 (registered)
        return fair_hit(rec["answer"], choice)
    gold = rec["answer"]
    if rec["typ"] == "binary":
        gold = "Yes" if gold.strip().lower() == "yes" else "No"
    return int(str(choice).strip().lower() == str(gold).strip().lower())


def evidence(rec):
    g = lambda k: (rec[k] if isinstance(rec[k], dict)  # noqa: E731
                   else torch.tensor(rec[k], dtype=torch.float64))
    s0, blank = g("s0"), g("blank")
    ar, ab = g("ans_real"), g("ans_blank")
    dd, ad = sub(s0, blank), sub(ar, ab)
    base = cent(s0)
    clean2 = cent(maxi(dd, ad))
    agree = float(pick(dd, rec) == pick(ad, rec))
    srcs = {k: g(k) for k in ("gray", "strongblur", "centermask")}
    # iic proxy: per-source top-margin drop is unavailable (no vocab maxp
    # saved) -> deployed Oth-style fixed choice is out of scope here; use
    # argmin of source top-margin (closest available answer-free signal),
    # disclosed.
    src = min(srcs, key=lambda k: m(srcs[k]))
    resid = cent(sub(s0, srcs[src]))
    return base, clean2, agree, resid


def ladder_run(rec, C):
    base, clean2, agree, resid = evidence(rec)
    state = base
    if m(state) <= C["t1"]:
        state = clean2 if (m(clean2) + agree) >= C["tauT"] else base
    if m(state) <= C["t2"]:
        cand = add(state, resid)
        if m(cand) - m(state) >= C["tauV"]:
            state = cand
    return pick(state, rec), pick(base, rec)


def derive_constants(nat, t1_pct, tauT_pct=None):
    ev = [evidence(r) for r in nat]
    m0 = [m(e[0]) for e in ev]
    t1 = float(np.percentile(m0, t1_pct))
    admits = [m(e[1]) + e[2] for e in ev if m(e[0]) <= t1]
    tp = QPOS["tauT"] if tauT_pct is None else tauT_pct
    tauT = float(np.percentile(admits, tp)) if admits else 0.0
    post1 = []
    for e in ev:
        if m(e[0]) <= t1:
            st = e[1] if (m(e[1]) + e[2]) >= tauT else e[0]
            post1.append(m(st))
    t2 = float(np.percentile(post1, QPOS["t2"])) if post1 else 0.0
    gains = []
    for e in ev:
        if m(e[0]) <= t1:
            st = e[1] if (m(e[1]) + e[2]) >= tauT else e[0]
            if m(st) <= t2:
                gains.append(m(add(st, e[3])) - m(st))
    tauV = float(np.percentile(gains, QPOS["tauV"])) if gains else 0.625
    return dict(t1=t1, tauT=tauT, t2=t2, tauV=tauV, t1_pct=t1_pct)


def calibrate():
    """AMENDED B2 (see FINDINGS): 2-D {t1_pct × tauT_pct} Val-only sweep
    for natural no-harm at max coverage; per-type deltas reported."""
    nat = load("natural_val")
    band = 100.0 / len(nat)
    print(f"{MODEL}: natural_val n={len(nat)} band=±{band:.3f}")
    best, best_cov = None, -1
    for tp in (5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0):
        for pct in (95.0, 85.0, 70.0):
            C = derive_constants(nat, pct, tp)
            hl = hb = helped = harmed = 0
            bytyp = {}
            for r in nat:
                p, bp = ladder_run(r, C)
                a, b = hit(r, p), hit(r, bp)
                hl += a; hb += b
                helped += (a and not b); harmed += (b and not a)
                d0 = bytyp.setdefault(r["typ"], [0, 0])
                d0[0] += (a and not b); d0[1] += (b and not a)
            d = 100.0 * (hl - hb) / len(nat)
            cov = pct - tp  # crude coverage proxy: fire range minus rejects
            ok = d >= -band
            print(f"  tauT_pct={tp:4.1f} t1_pct={pct:5.1f}  natΔ={d:+.3f} "
                  f"({helped}/{harmed}) types={ {k: tuple(v) for k, v in bytyp.items()} }"
                  f"{'  OK' if ok else ''}")
            if ok and cov > best_cov:
                best, best_cov = dict(C, tauT_pct=tp), cov
    if best is None:
        print("  B2 FAIL: no config meets natural no-harm — Phase-B "
              "acceptance fails at B2 for this model (report as such).")
        return
    CAL.write_text(json.dumps(best))
    print(f"frozen: {best}")


def sci5(rec, P, supp=None):
    if rec["typ"] == "open":
        # candidate-level adaptation (disclosed): views from b3prep
        # (s0/blank/ans_real=tcf1) + supplement (noise500/tcf2v)
        sp = (supp or {}).get(rec["index"])
        if not sp:
            return None
        pool = [c for c in rec["pool"] if c in sp["noise500"] and c in sp["tcf2v"]
                and c in rec["s0"] and c in rec["ans_real"]]
        if not pool:
            return None  # dup-window key drift; disclosed
        o = torch.tensor([rec["s0"][c] for c in pool], dtype=torch.float64)
        v1 = torch.tensor([rec["blank"][c] for c in pool], dtype=torch.float64)
        v2 = torch.tensor([sp["noise500"][c] for c in pool], dtype=torch.float64)
        t1 = torch.tensor([rec["ans_real"][c] for c in pool], dtype=torch.float64)
        t2 = torch.tensor([sp["tcf2v"][c] for c in pool], dtype=torch.float64)
        labels = pool
    else:
        g = lambda k: torch.tensor(rec[k], dtype=torch.float64)  # noqa: E731
        o, v1, v2 = g("s0_raw"), g("blank_raw"), g("noise500_raw")
        t1, t2 = g("tcf1_raw"), g("tcf2_raw")
        labels = rec.get("labels") or ["Yes", "No"]
    consistent = torch.stack([o, t1, t2]).max(0).values / P["gamma"]
    unbiased = (o - (v1 + v2) / 2.0) / P["beta"]
    final = consistent + unbiased
    cutoff = float(np.log(P["theta"])) + float(consistent.max())
    final = final.masked_fill(consistent < cutoff, -float("inf"))
    return labels[int(final.argmax())]


def gated_m3id(rec, C):
    g = lambda k: (rec[k] if isinstance(rec[k], dict)  # noqa: E731
                   else torch.tensor(rec[k], dtype=torch.float64))
    s0, blank = g("s0"), g("blank")
    base = cent(s0)
    if m(base) <= C["t1"]:
        state = add(s0, sub(s0, blank)) if not isinstance(s0, dict) else \
            {k: s0[k] + 2.0 * (s0[k] - blank[k]) for k in s0}
        if not isinstance(s0, dict):
            state = s0 + 2.0 * (s0 - blank)
        return pick(state, rec)
    return pick(base, rec)


def cells(rows, fn):
    groups = {"B": {"vcf_only", "both"}, "S": {"tcf_only", "both"},
              "BS": {"vcf_only", "tcf_only", "both"}}
    fmt = {"MCQ": lambda r: r["typ"] == "mcq",
           "Oth": lambda r: r["typ"] in ("binary", "open"),
           "All": lambda r: True}
    out = {}
    for gname, modes in groups.items():
        for fname, ff in fmt.items():
            sub_ = [r for r in rows if r["mode"] in modes and ff(r)]
            out[f"{gname}_{fname}"] = (round(100.0 * np.mean([fn(r) for r in sub_]), 2)
                                       if sub_ else None)
    return out


def test():
    if "--confirm" not in sys.argv:
        print("REFUSED: pre-registered single Test read requires --confirm")
        return
    C = json.loads(CAL.read_text())
    supp = {}
    sp_path = OUT / f"b3supp_{MODEL}.jsonl"
    if sp_path.exists():
        with sp_path.open() as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    supp[s["index"]] = s
    rows = [r for r in load("biased") if r.get("split") == "test"]
    print(f"{MODEL}: biased Test n={len(rows)}  constants={C}  supp={len(supp)}")
    for r in rows:
        p, bp = ladder_run(r, C)
        r["_ladder"], r["_base"] = hit(r, p), hit(r, bp)
        r["_gm3"] = hit(r, gated_m3id(r, C))
    # SCI5: Val-select params on biased Val
    vrows = [r for r in load("biased") if r.get("split") == "val"]
    best_p, best_acc = SCI5_PARAMS[0], -1
    for P in SCI5_PARAMS:
        vh = [hit(r, c) for r in vrows
              for c in [sci5(r, P, supp)] if c is not None]
        acc = np.mean(vh) if vh else 0
        print(f"  SCI5 params {P}: biased Val acc={100*acc:.2f} (n={len(vh)})")
        if acc > best_acc:
            best_acc, best_p = acc, P
    for r in rows:
        c = sci5(r, best_p, supp)
        r["_sci5"] = hit(r, c) if c is not None else r["_base"]
    print(f"  SCI5 selected: {best_p}")
    for name, key in (("base", "_base"), ("SCI5-reimpl", "_sci5"),
                      ("gated-M3ID", "_gm3"), ("LADDER", "_ladder")):
        c = cells(rows, lambda r, k=key: r[k])
        print(f"  {name:<12} " + " ".join(f"{k}={v}" for k, v in c.items()))
    (OUT / f"b3_test_readout_{MODEL}.json").write_text(json.dumps(
        {"constants": C, "sci5_params": best_p,
         "cells": {nm: cells(rows, lambda r, k=key: r[k])
                   for nm, key in (("base", "_base"), ("sci5", "_sci5"),
                                    ("gm3", "_gm3"), ("ladder", "_ladder"))}}))
    print("wrote b3_test_readout")


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
    elif "--test" in sys.argv:
        test()
    else:
        print(__doc__)
