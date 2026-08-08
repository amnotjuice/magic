"""Margin law on FOUR backbones (paper Fig. 2). U(m) = E[dacc | base margin]
for the UNGATED rung-1 corrector, measured separately on biased and on natural
rows. Zero GPU: the 2024 pair from the frozen release bundles, the 2025 pair
from the frozen phase-B caches. Writes figures/marginlaw_pop4.pdf.
"""
import gzip, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import magic as M

import figstyle as fs

fs.use()
BLUE, VERM = fs.OURS, fs.CONTRAST   # shared house style, research/magic/figstyle.py
EDGES = [0, 0.5, 1, 2, 3, 5, 8, 12, 20, 32]
# Deployed biased-benchmark gate per backbone: the 95th percentile of the base
# margin on the biased validation split (the label-free rule of Sec. 3.4).
# The 2025 pair previously carried a natural-split percentile here, which is
# not the gate they deploy on the biased benchmark.
T1 = {"Qwen2-VL-7B": 3.25, "LLaVA-NeXT-8B": 5.03,
      "Qwen3-VL-8B": 11.71, "LLaVA-OneVision-1.5-8B": 19.55}
# Gate fitted on the natural split, where the allocation actually selects.
# Only the 2024 pair carries one; the 2025 pair is deployed on the biased
# benchmark alone.
T1_NAT = {"Qwen2-VL-7B": M.LADDER_NAT["qwen"].theta1,
          "LLaVA-NeXT-8B": M.LADDER_NAT["llava"].theta1}


def center(v):
    mu = sum(v) / len(v)
    return [x - mu for x in v]


def collect_2024(mk):
    B = json.load(gzip.open(ROOT / "magic_mcq_bundle.json.gz", "rt"))
    m, d, b = [], [], []
    for r in B["mcq"][mk]:
        s0, lab, ans = r["s0"], r["labels"], str(r["answer"]).strip().upper()
        lift = [max(r["default_delta"][i], r["answer_delta"][i]) for i in range(len(s0))]
        hb = lab[M.argmax_key(s0)] == ans
        hl = lab[M.argmax_key(lift)] == ans
        m.append(M.margin(s0)); d.append(int(hl) - int(hb)); b.append(1)
    for r in B["natural"][mk]:
        s0, lab, ans = r["base"], r["labels"], str(r["answer"]).strip().upper()
        hb = lab[M.argmax_key(s0)] == ans
        hl = lab[M.argmax_key(r["clean2"])] == ans
        m.append(M.margin(s0)); d.append(int(hl) - int(hb)); b.append(0)
    return np.array(m), np.array(d), np.array(b)


def collect_2025(mk):
    m, d, b = [], [], []
    for line in open(ROOT / f"experiments/data/newbackbone_generation/b3prep_{mk}.jsonl"):
        r = json.loads(line)
        if r.get("typ") != "mcq" or r.get("ans_real") is None:
            continue
        s0, lab = r.get("s0"), r.get("labels")
        ans = str(r.get("answer", "")).strip().upper()
        if not s0 or not lab or ans not in lab:
            continue
        try:
            l1 = [s0[i] - r["blank"][i] for i in range(len(s0))]
            l2 = [r["ans_real"][i] - r["ans_blank"][i] for i in range(len(s0))]
        except Exception:
            continue
        lift = center([max(l1[i], l2[i]) for i in range(len(s0))])
        hb = lab[int(np.argmax(s0))] == ans
        hl = lab[int(np.argmax(lift))] == ans
        m.append(M.margin(s0)); d.append(int(hl) - int(hb))
        b.append(1 if r.get("split") in ("val", "test") else 0)
    return np.array(m), np.array(d), np.array(b)


def curve(m, d, min_n=25):
    """Binned utility with a 95% CI from the row-level sampling variance."""
    xs, ys, se, ns = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (m >= lo) & (m < hi)
        n = int(sel.sum())
        if n >= min_n:
            v = d[sel]
            xs.append(np.sqrt(lo * hi) if lo > 0 else 0.5 * hi)   # log-scale midpoint
            ys.append(100 * v.mean())
            se.append(100 * 1.96 * v.std(ddof=1) / np.sqrt(n))
            ns.append(n)
    return np.array(xs), np.array(ys), np.array(se), ns


def gap_significant(m, d, b, min_n=25):
    """Per-bin two-sample test on U_biased - U_natural. Returns the contiguous
    margin span over which the difference clears 95%, plus the per-bin record.
    The two populations are disjoint row sets, so the SE of the difference is
    sqrt(se_b^2 + se_n^2) -- overlapping single-curve bands are NOT the test."""
    rec, sig_edges = [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sb = (b == 1) & (m >= lo) & (m < hi)
        sn = (b == 0) & (m >= lo) & (m < hi)
        nb, nn = int(sb.sum()), int(sn.sum())
        if nb < min_n or nn < min_n:
            continue
        mb, mn = 100 * d[sb].mean(), 100 * d[sn].mean()
        se = np.hypot(100 * d[sb].std(ddof=1) / np.sqrt(nb),
                      100 * d[sn].std(ddof=1) / np.sqrt(nn))
        ci = 1.96 * se
        ok = abs(mb - mn) > ci
        rec.append({"bin": [lo, hi], "n": [nb, nn], "diff": round(mb - mn, 2),
                    "ci95": round(ci, 2), "significant": bool(ok)})
        if ok:
            sig_edges.append((lo if lo > 0 else EDGES[1] * 0.4, hi))
    span = (sig_edges[0][0], sig_edges[-1][1]) if sig_edges else None
    return span, rec


def zero_cross(x, y):
    """First downward zero crossing, linearly interpolated in log-x."""
    for i in range(len(y) - 1):
        if y[i] > 0 >= y[i + 1]:
            t = y[i] / (y[i] - y[i + 1])
            return float(np.exp(np.log(x[i]) + t * (np.log(x[i + 1]) - np.log(x[i]))))
    return None


def style_ax(ax):
    fs.frame(ax, grid="y")


def main():
    specs = [("qwen", "Qwen2-VL-7B", collect_2024),
             ("llava", "LLaVA-NeXT-8B", collect_2024),
             ("qwen3", "Qwen3-VL-8B", collect_2025),
             ("ov15", "LLaVA-OneVision-1.5-8B", collect_2025)]
    out = {}
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.05), sharey=True)
    for ax, (key, disp, fn) in zip(axes, specs):
        m, d, b = fn(key)
        adm_b = float((m[b == 1] <= T1[disp]).mean())
        adm_n = (float((m[b == 0] <= T1_NAT[disp]).mean())
                 if disp in T1_NAT else None)
        xb, yb, eb, nb = curve(m[b == 1], d[b == 1])
        xn, yn, en, nn = curve(m[b == 0], d[b == 0])
        out[disp] = {"n_biased": int((b == 1).sum()), "n_natural": int((b == 0).sum()),
                     "biased": {"m": xb.round(3).tolist(), "u": yb.round(2).tolist(),
                                "ci95": eb.round(2).tolist(), "n": nb},
                     "natural": {"m": xn.round(3).tolist(), "u": yn.round(2).tolist(),
                                 "ci95": en.round(2).tolist(), "n": nn},
                     "natural_zero_crossing": zero_cross(xn, yn),
                     "admits_biased": adm_b, "admits_natural": adm_n}

        # the claim: the room between the two populations, shaded only where a
        # two-sample test on the difference clears 95%
        span, rec = gap_significant(m, d, b)
        out[disp]["gap_test"] = rec
        lo, hi = max(xb.min(), xn.min()), min(xb.max(), xn.max())
        if span:
            lo, hi = max(lo, span[0]), min(hi, span[1])
        gx = np.exp(np.linspace(np.log(lo), np.log(hi), 200))
        gb = np.interp(np.log(gx), np.log(xb), yb)
        gn = np.interp(np.log(gx), np.log(xn), yn)
        ax.fill_between(gx, gn, gb, where=gb >= gn, color=BLUE, alpha=0.08, lw=0)

        ax.axhline(0, color=fs.RULE, lw=0.8, zorder=1)
        for x, y, e, c, mk_, lab in ((xb, yb, eb, BLUE, "o", "biased"),
                                     (xn, yn, en, VERM, "s", "natural")):
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.16, lw=0, zorder=2)
            ax.plot(x, y, mk_ + "-", color=c, lw=1.5, ms=2.6, mew=0, zorder=3, label=lab)

        zc = out[disp]["natural_zero_crossing"]
        if zc:
            ax.plot([zc], [0], marker="v", color=VERM, ms=3.6, mew=0, zorder=4,
                    clip_on=False)

        # No gate marker. On the biased subsets the fitted t_1 admits 93-100%
        # of samples (100% on LLaVA-OV-1.5, whose t_1 of 19.55 sits beyond the
        # whole population), so a rule drawn here reads as a cut that is not
        # being made, and only two of the four backbones carry a natural-split
        # gate to pair it with. The fitted values are reported in the text.
        lo = min(xb.min(), xn.min()); hi = max(xb.max(), xn.max())
        ax.set_xlim(lo * 0.75, hi * 1.15)

        ax.set_xscale("log")
        ax.set_title(disp.replace("LLaVA-OneVision-1.5-8B", "LLaVA-OV-1.5-8B"),
                     fontsize=8.2, pad=3)
        ax.set_xlabel("base decision margin", labelpad=1.5, fontsize=8.4)
        style_ax(ax)

    axes[0].set_ylabel("correction $\\Delta$acc (pp)", fontsize=8.4)
    axes[0].legend(loc="upper right", fontsize=7.6, frameon=False,
                   borderaxespad=0.5, handlelength=1.3, labelspacing=0.25)
    fig.tight_layout(pad=0.35)
    fig.savefig(ROOT / "figures/marginlaw_pop4.pdf", bbox_inches="tight")
    (ROOT / "experiments/data/marginlaw4.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print("%-24s biased n=%-5d natural n=%-5d  natural zero-crossing m=%s"
              % (k, v["n_biased"], v["n_natural"],
                 round(v["natural_zero_crossing"], 2) if v["natural_zero_crossing"] else "-"))
    print("wrote figures/marginlaw_pop4.pdf")


if __name__ == "__main__":
    main()
