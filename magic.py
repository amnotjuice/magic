"""MAGIC — the margin ladder, self-contained reference implementation.

One test-time debiasing method (no retraining): the model's own decision
margin drives a two-rung ladder that spends correction compute only where
the answer is contestable.

    rung k fires iff  margin(state) <= theta_k          (allocation)
    rung 1 (prior isolation): REPLACE the state with the clean2 lift
    rung 2 (visual residual, ADD):     commit iff
            margin(state + residual) - margin(state) >= tau_V   (commitment)

The visual residual draws from a 3-channel corruption set {grayscale,
blur, centre-mask}.  Channel evaluation follows the same "spend where it
pays" rule: on a rich provided-option answer space (multiple-choice) the
method *routes* to the most task-relevant channel (argmin over
image_info_conf, precomputed here as `selected_vc`); on binary / generated
answers, where routing gives no measured gain, it uses the single reliable
default channel (grayscale, `vc_gray`).

This file needs nothing but the Python standard library and the two score
bundles shipped alongside it.  It reproduces the paper's frozen Test
numbers bit-exactly (run `python magic.py`).  Centering, present in the
research code, is dropped here because argmax and margin (= top1 - top2)
are invariant to a per-candidate constant shift.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Rung:
    theta: float   # allocation: spend iff margin(state) <= theta
    tau: float     # commitment threshold (rung 2)


@dataclass(frozen=True)
class Ladder:
    theta1: float  # rung-1 allocation threshold (prior isolation)
    rung2: Rung    # visual residual


# Biased-benchmark configs (thresholds selected on the DRBench validation
# split): t1 = ~95th pct of margin(S0); t2 = the measured VC value-band
# midpoint; tau_V = 0.625.
LADDER_MCQ = {
    "qwen":  Ladder(3.25, Rung(0.4, 0.625)),
    "llava": Ladder(5.03125, Rung(0.3, 0.625)),
}
# Open-ended configs: same t1 as MCQ, t2 = 0.5 plateau, tau_V per backbone.
LADDER_OTH = {
    "qwen":  Ladder(3.25, Rung(0.5, -0.220826)),
    "llava": Ladder(5.03125, Rung(0.5, 0.375)),
}
# Natural-distribution configs (thresholds selected on the natural
# validation split, joint grid, max-delta rule; MCQ shown — the bundled
# natural rows are MCQ).
LADDER_NAT = {
    "qwen":  Ladder(0.5, Rung(0.2, 0.625)),
    "llava": Ladder(2.0, Rung(0.2, 0.625)),
}

GROUPS = {"B": {"vcf_only", "both"},
          "S": {"tcf_only", "both"},
          "BS": {"vcf_only", "tcf_only", "both"}}


# --------------------------------------------------------------------------
# Score helpers (lists or {candidate: score} dicts)
# --------------------------------------------------------------------------

def _vals(x):
    return x.values() if isinstance(x, dict) else x


def _f32(x: float) -> float:
    """Round a float to float32 precision (the research code computes the
    margin in float32 via tensor.float(); we match it bit-for-bit)."""
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def margin(x) -> float:
    import math
    v = sorted((_f32(float(z)) for z in _vals(x) if math.isfinite(float(z))), reverse=True)
    return _f32(v[0] - v[1]) if len(v) > 1 else 0.0


def argmax_key(x):
    if isinstance(x, dict):
        return max(x, key=x.get)
    return max(range(len(x)), key=lambda i: x[i])


def _add(a, b):
    if isinstance(a, dict):
        return {k: a[k] + b[k] for k in a}
    return [a[i] + b[i] for i in range(len(a))]


# --------------------------------------------------------------------------
# The ladder — one control flow for both formats
# --------------------------------------------------------------------------

def run_ladder(state0, lift, residual, cfg: Ladder):
    """Format-agnostic margin ladder.  `state0`/`lift`/`residual` are
    opaque score containers (list for MCQ, dict for open-ended)."""
    state = state0
    if margin(state) <= cfg.theta1:                            # allocate rung 1
        state = lift
    if margin(state) <= cfg.rung2.theta:                       # allocate rung 2
        cand = _add(state, residual)
        if margin(cand) - margin(state) >= cfg.rung2.tau:      # commit
            state = cand
    return state


def predict_mcq(row, cfg: Ladder) -> str:
    """MCQ format spec: fixed provided options; residual on s0; routed
    channel (selected_vc = argmin-iic over the 3-channel set)."""
    s0, dd, ad, vc = row["s0"], row["default_delta"], row["answer_delta"], row["selected_vc"]
    lift = [max(dd[i], ad[i]) for i in range(len(s0))]
    residual = [s0[i] - vc[i] for i in range(len(s0))]
    state = run_ladder(s0, lift, residual, cfg)
    return row["labels"][argmax_key(state)]


def predict_oth(row, cfg: Ladder) -> str:
    """Open-ended format spec: materialised candidate set; base = envelope
    over real views; residual anchored on the envelope; single reliable
    channel (vc_gray)."""
    pool = row["pool"]
    rd, ra, bd, ba, vg = (row["real_def"], row["real_ans"], row["blank_def"],
                          row["blank_ans"], row["vc_gray"])
    base = {c: max(rd[c], ra[c]) for c in pool}
    l1 = {c: rd[c] - bd[c] for c in pool}
    l2 = {c: ra[c] - ba[c] for c in pool}
    lift = {c: max(l1[c], l2[c]) for c in pool}
    residual = {c: base[c] - vg[c] for c in pool}
    state = run_ladder(dict(rd), lift, residual, cfg)
    return argmax_key(state)


# --------------------------------------------------------------------------
# Parity: reproduce the frozen Test numbers bit-exactly
# --------------------------------------------------------------------------

def _cells(rows, predict, cfg):
    hits = {g: [] for g in GROUPS}
    for row in rows:
        ok = predict(row, cfg) == str(row["answer"]).strip().upper() if predict is predict_mcq \
            else _open_hit(row["answer"], predict(row, cfg))
        for g, modes in GROUPS.items():
            if row["mode"] in modes:
                hits[g].append(ok)
    return {g: round(100.0 * sum(v) / len(v), 2) for g, v in hits.items()}


def _open_hit(answer, pred) -> bool:
    """Prefix match with a word boundary (the benchmark's scoring rule)."""
    import re
    a, p = str(answer).lower(), str(pred).lower()
    if len(p) < len(a):
        return False
    if len(p) == len(a):
        return p == a
    if not re.fullmatch(r"[A-Za-z]", p[len(a)]):
        return p[:len(a)] == a
    return False


FROZEN = {
    "qwen":  {"mcq": {"B": 30.26, "S": 48.73, "BS": 32.30},
              "oth": {"B": 25.15, "S": 45.63, "BS": 35.16},
              "natural": (0.762, 1.542)},
    "llava": {"mcq": {"B": 26.95, "S": 40.99, "BS": 31.13},
              "oth": {"B": 41.83, "S": 58.30, "BS": 52.01},
              "natural": (1.5616, 2.434)},
}


def _natural_readout(rows, cfg):
    n = hits = base_hits = cost = 0
    for row in rows:
        state, passes = row["base"], 1
        if margin(state) <= cfg.theta1:
            passes += 3
            state = row["clean2"]
        if margin(state) <= cfg.rung2.theta:
            passes += 3
            cand = _add(state, row["vc_delta"])
            if margin(cand) - margin(state) >= cfg.rung2.tau:
                state = cand
        n += 1
        cost += passes
        hits += row["labels"][argmax_key(state)] == row["answer"]
        base_hits += row["labels"][argmax_key(row["base"])] == row["answer"]
    return round(100.0 * (hits - base_hits) / n, 4), round(cost / n, 3)


def main() -> None:
    with gzip.open(HERE / "magic_mcq_bundle.json.gz", "rt") as f:
        b = json.load(f)
    with gzip.open(HERE / "magic_oth_bundle.json.gz", "rt") as f:
        oth = json.load(f)

    ok = True
    for mk in ("qwen", "llava"):
        got = _cells(b["mcq"][mk], predict_mcq, LADDER_MCQ[mk])
        want = FROZEN[mk]["mcq"]
        ok &= got == want
        print(f"{mk:6} MCQ Test parity: {'OK ' if got == want else 'FAIL'} {got}")
        got_o = _cells(oth[mk], predict_oth, LADDER_OTH[mk])
        want_o = FROZEN[mk]["oth"]
        ok &= got_o == want_o
        print(f"{mk:6} Oth Test parity: {'OK ' if got_o == want_o else 'FAIL'} {got_o}")
    for mk in ("qwen", "llava"):
        d, avg = _natural_readout(b["natural"][mk], LADDER_NAT[mk])
        want = FROZEN[mk]["natural"]
        if want is None:
            print(f"{mk:6} natural Test readout: delta={d} avg_passes={avg} (freeze me)")
        else:
            want_d, want_avg = want
            ok &= (d == want_d and avg == want_avg)
            print(f"{mk:6} natural Test parity: {'OK ' if (d == want_d and avg == want_avg) else 'FAIL'} "
                  f"delta={d} avg_passes={avg}")
    print("ALL PARITY OK" if ok else "PARITY MISMATCH")


if __name__ == "__main__":
    main()
