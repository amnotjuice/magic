"""Two-sample companion to the one-sample bootstrap in tab:main.

The bootstrap in the paper resamples MAGIC's own rows against the published
SCI cell treated as a constant, because SCI publishes points without
per-row records. That construction ignores the baseline's own sampling
variance. This script reports the two-sample normal-approximation test
that does account for it, using the published accuracy and the shared
subset size. Replay only, zero GPU.
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LED = json.loads((ROOT / "experiments/data/paper_reconciliation/ledger.json").read_text())

# published BS-Overall cells (SCI paper Table 2), and MAGIC's replayed cells
PUB = {"qwen": {"SCI5": 29.50, "SCI7": 31.72},
       "llava": {"SCI5": 34.19, "SCI7": 34.92}}


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def main():
    out = {}
    for mk in ("qwen", "llava"):
        n = LED[mk]["bs_overall_ci"]["n"]
        p1 = LED[mk]["cells"]["full"]["BS"]["All"] / 100
        row = {"n": n, "magic": round(100 * p1, 2)}
        for k, acc in PUB[mk].items():
            p2 = acc / 100
            se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
            z = (p1 - p2) / se
            row[k] = {"baseline": acc, "delta": round(100 * (p1 - p2), 2),
                      "z": round(z, 3),
                      "p_one_sided": round(norm_sf(z), 4),
                      "p_two_sided": round(2 * norm_sf(z), 4),
                      "p_one_sample_bootstrap": LED[mk]["bs_overall_ci"]["P_le"][k]}
        out[mk] = row
        print(f"{mk}: n={n} MAGIC={row['magic']}")
        for k in PUB[mk]:
            d = row[k]
            print(f"   vs {k}: delta {d['delta']:+.2f}  two-sample p="
                  f"{d['p_one_sided']:.4f} one-sided / {d['p_two_sided']:.4f} "
                  f"two-sided   (one-sample bootstrap "
                  f"{d['p_one_sample_bootstrap']})")
    (ROOT / "experiments/data/twosample_data/twosample.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
