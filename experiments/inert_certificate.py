"""Proposition-1 certificate rate: what fraction of the rows the gate skips
are PROVABLY unchanged by the rung-2 correction.

For a skipped row (margin(s0) > t1) the additive rung-2 residual
D2 = center(s0 - V_src) satisfies ||D2||_inf <= B, so Proposition 1
certifies the row when

        margin(s0) > 2 * ||D2||_inf .

Zero GPU, replayed from the released bundles. Reported for the deployed
biased thresholds and, separately, for the natural operating point, since
the paper quotes the rate over the rows the gate actually skips.

Usage: python3 experiments/inert_certificate.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import magic as M  # noqa: E402


def load(name):
    with gzip.open(ROOT / name, "rt", encoding="utf-8") as f:
        return json.load(f)


def center(v):
    m = sum(v) / len(v)
    return [x - m for x in v]


def rate(rows, theta1, kind):
    """(#certified, #skipped) among rows the gate skips."""
    cert = skip = 0
    for row in rows:
        if kind == "mcq":
            s0, vc = row["s0"], row["selected_vc"]
            d2 = center([s0[i] - vc[i] for i in range(len(s0))])
        elif kind == "nat":
            s0 = row["base"]
            d2 = list(row["vc_delta"]) if isinstance(row["vc_delta"], list) \
                else list(row["vc_delta"].values())
        else:
            pool = row["pool"]
            rd, vg = row["real_def"], row["vc_gray"]
            base = {c: max(rd[c], row["real_ans"][c]) for c in pool}
            d2 = center([base[c] - vg[c] for c in pool])
            s0 = rd  # the open-ended ladder starts from the real/default view
        m = M.margin(s0)
        if m <= theta1:
            continue
        skip += 1
        if m > 2.0 * max(abs(x) for x in d2):
            cert += 1
    return cert, skip


def main():
    mcq_b, oth_b = load("magic_mcq_bundle.json.gz"), load("magic_oth_bundle.json.gz")
    out = {}
    for model in ("qwen", "llava"):
        c1, s1 = rate(mcq_b["mcq"][model], M.LADDER_MCQ[model].theta1, "mcq")
        c2, s2 = rate(oth_b[model], M.LADDER_OTH[model].theta1, "oth")
        c, s = c1 + c2, s1 + s2
        nat = mcq_b.get("natural", {}).get(model)
        line = {"biased_certified": c, "biased_skipped": s,
                "biased_rate": round(100.0 * c / s, 2) if s else None}
        if nat:
            cn, sn = rate(nat, M.LADDER_NAT[model].theta1, "nat")
            line.update({"natural_certified": cn, "natural_skipped": sn,
                         "natural_rate": round(100.0 * cn / sn, 2) if sn else None})
        out[model] = line
        print(f"{model:6s} biased  {c}/{s} = "
              f"{line['biased_rate']}%   natural " +
              (f"{line.get('natural_certified')}/{line.get('natural_skipped')} = "
               f"{line.get('natural_rate')}%" if nat else "n/a"))
    p = ROOT / "experiments/data/inert_certificate.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p.name)


if __name__ == "__main__":
    main()
