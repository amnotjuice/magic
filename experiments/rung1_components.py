"""Rung-1 component ablation: the blank contrast and the answer-format
contrast alone, against their elementwise-max fusion.

Zero GPU: replays the frozen paper-form (noadm) ladder from the released
bundles, changing ONLY the rung-1 lift:

    fusion (deployed) : lift = max(default_delta, answer_delta)
    blank only        : lift = default_delta          (s0 - blank)
    answer-format only: lift = answer_delta           (ans_r - ans_b)

Everything else -- gates, rung-2 residual, commitment, scoring -- is the
deployed operator, so the three rows differ in one component.

Usage: python3 experiments/rung1_components.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import magic as M  # noqa: E402


def run_ladder_noadm(state0, lift, residual, cfg):
    state = state0
    if M.margin(state) <= cfg.theta1:
        state = lift
    if M.margin(state) <= cfg.rung2.theta:
        cand = M._add(state, residual)
        if M.margin(cand) - M.margin(state) >= cfg.rung2.tau:
            state = cand
    return state


def predict_mcq(row, cfg, mode):
    s0, dd, ad, vc = (row["s0"], row["default_delta"],
                      row["answer_delta"], row["selected_vc"])
    if mode == "fusion":
        lift = [max(dd[i], ad[i]) for i in range(len(s0))]
    elif mode == "blank":
        lift = list(dd)
    else:
        lift = list(ad)
    residual = [s0[i] - vc[i] for i in range(len(s0))]
    return row["labels"][M.argmax_key(run_ladder_noadm(s0, lift, residual, cfg))]


def predict_oth(row, cfg, mode):
    pool = row["pool"]
    rd, ra, bd, ba, vg = (row["real_def"], row["real_ans"], row["blank_def"],
                          row["blank_ans"], row["vc_gray"])
    base = {c: max(rd[c], ra[c]) for c in pool}
    l1 = {c: rd[c] - bd[c] for c in pool}
    l2 = {c: ra[c] - ba[c] for c in pool}
    if mode == "fusion":
        lift = {c: max(l1[c], l2[c]) for c in pool}
    elif mode == "blank":
        lift = dict(l1)
    else:
        lift = dict(l2)
    residual = {c: base[c] - vg[c] for c in pool}
    return M.argmax_key(run_ladder_noadm(dict(rd), lift, residual, cfg))


def load(name):
    with gzip.open(ROOT / name, "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    mcq_b, oth_b = load("magic_mcq_bundle.json.gz"), load("magic_oth_bundle.json.gz")
    out = {}
    for model in ("qwen", "llava"):
        out[model] = {}
        for mode in ("fusion", "blank", "answer"):
            hits = {g: [] for g in M.GROUPS}
            for fmt, rows, pred, cfgs in (
                    ("mcq", mcq_b["mcq"][model], predict_mcq, M.LADDER_MCQ),
                    ("oth", oth_b[model], predict_oth, M.LADDER_OTH)):
                cfg = cfgs[model]
                for row in rows:
                    p = pred(row, cfg, mode)
                    ok = (p == str(row["answer"]).strip().upper()) if fmt == "mcq" \
                        else M._open_hit(row["answer"], p)
                    for g, modes in M.GROUPS.items():
                        if row["mode"] in modes:
                            hits[g].append(ok)
            cells = {g: round(100.0 * sum(v) / len(v), 2) for g, v in hits.items()}
            out[model][mode] = cells
            print(f"{model:6s} {mode:7s} B/S/BS = "
                  f"{cells['B']:6.2f} {cells['S']:6.2f} {cells['BS']:6.2f}")
        f = out[model]["fusion"]["BS"]
        print(f"  fusion - blank  = {f - out[model]['blank']['BS']:+.2f} BS")
        print(f"  fusion - answer = {f - out[model]['answer']['BS']:+.2f} BS")
    p = ROOT / "experiments/data/rung1_components.json"
    p.write_text(json.dumps(out, indent=2))
    # parity: the fusion arm must reproduce the published cells
    assert out["qwen"]["fusion"]["BS"] == 33.13, out["qwen"]["fusion"]
    assert out["llava"]["fusion"]["BS"] == 36.22, out["llava"]["fusion"]
    print("\nPARITY OK (fusion reproduces 33.13 / 36.22); wrote", p.name)


if __name__ == "__main__":
    main()
