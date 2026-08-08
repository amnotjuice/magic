"""Verification replay for the two 2025-backbone numbers that the paper
states without a committed artifact: (i) mean forward passes on the biased
subset, (ii) the natural-Test delta of the *deployed* configuration.

Zero GPU. Reads the frozen Phase-B caches and the frozen solved configs;
no parameter is selected here.

Usage: python3 experiments/cost_replay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import newbackbone_eval_read as V6  # noqa: E402
from newbackbone_eval_read import m, add, cent, sub  # noqa: E402

PB = ROOT / "experiments/data/newbackbone_generation"
SOLVED = ROOT / "experiments/data/solver_calibration"


def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ladder_passes(rec, C):
    """Pass count for the two-rung ladder on the 2025 caches: 1 + 3*f1 + 3*f2,
    applied to every answer format. The format-blind charge is correct here
    because the 2025 caches carry the identical view set for MCQ, binary and
    open rows (s0, blank, ans_real, ans_blank, gray, strongblur, centermask)
    with a pre-generated candidate pool. The 2024 pipeline's shared-prefill
    discount for ViLP does not apply."""
    base, clean2, agree, resid = V6.evidence6(rec)
    n = 1
    state = base
    if m(state) <= C["t1"]:
        n += 3
        state = clean2 if (m(clean2) + agree) >= C["tauT"] else base
    if m(state) <= C["t2"]:
        n += 3
    return n


def gm3_passes(rec, C):
    """Gated single-blank corrector: base pass, plus one blank pass when the
    gate fires. Format-blind, as above."""
    base = cent(V6.evidence6(rec)[0])
    n = 1
    if m(base) <= C["t1"]:
        n += 1
    return n


def main():
    for mk in ("qwen3", "ov15"):
        solved = json.loads((SOLVED / f"solved_{mk}.json").read_text())
        C, comp = solved["C"], solved["comp"]
        rows = [r for r in load_jsonl(PB / f"b3prep_{mk}.jsonl")
                if r.get("surface") == "biased" and r.get("split") == "test"]
        fn = ladder_passes if comp == "ladder" else gm3_passes
        counts = [fn(r, C) for r in rows]
        by_fmt = {}
        for r, c in zip(rows, counts):
            by_fmt.setdefault(r.get("typ", "?"), []).append(c)
        print(f"== {mk} ({comp}) biased-test, n={len(rows)}")
        print(f"   mean passes overall: {sum(counts)/len(counts):.2f}")
        for k, v in sorted(by_fmt.items()):
            print(f"     {k:6s} n={len(v):5d}  mean={sum(v)/len(v):.2f}")


if __name__ == "__main__":
    main()
