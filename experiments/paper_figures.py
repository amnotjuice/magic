"""Publication figures for the MAGIC paper, released-form (magic.py) data.

Outputs vector PDFs into paper/figures/:
  teaser.pdf              — Fig 1: ladder flow + accuracy-vs-passes scatter
  marginlaw_pop.pdf       — restyle of the (L1)/(L2) margin-utility curves
  manifold_invariance.pdf — restyle of the probe-invariance bars
  pareto.pdf              — dual panel: biased anytime + natural delta

All accuracy/pass numbers are replayed from the frozen bundles (released
magic.py control flow) or read from the reconciliation ledger / published
SCI rows. Type, frame, grid and palette come from the shared house style in
research/magic/figstyle.py; fonts are sized for the final print dimensions
(ACL \\columnwidth ~3.03in, \\textwidth ~6.3in).

Run: python3 experiments/paper_figures.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import magic as M  # noqa: E402

FIGDIR = ROOT / "figures"
LEDGER = json.loads((ROOT / "experiments/data/paper_reconciliation/ledger.json").read_text())

import figstyle as fs  # noqa: E402

fs.use()
# Semantic roles from the shared house style (research/magic/figstyle.py).
BLUE, VERM, GREEN, ORANGE = fs.OURS, fs.CONTRAST, fs.THIRD, fs.FOURTH
GRAY, LGRAY = fs.MDGRAY, fs.LTGRAY

TITLE = {"qwen": "Qwen2-VL-7B", "llava": "LLaVA-NeXT-8B"}
PUB_BS = {  # published biased BS_MCQ @ nominal passes (matches the swept curve)
    "qwen":  {"TIE": (2, 20.27), "VCD": (2, 20.11), "M3ID": (2, 23.65),
              "SCI$_3$": (3, 24.54), "SCI$_5$": (5, 28.00), "SCI$_7$": (7, 29.61)},
    "llava": {"TIE": (2, 21.89), "VCD": (2, 22.54), "M3ID": (2, 24.15),
              "SCI$_3$": (3, 27.14), "SCI$_5$": (5, 28.80), "SCI$_7$": (7, 29.68)},
}
PUB_NAT = {  # published natural Overall deltas @ nominal passes
    "qwen":  {"TIE": (2, 0.15), "VCD": (2, 0.28), "M3ID": (2, 0.09), "SCI$_5$": (5, 0.40)},
    "llava": {"TIE": (2, -0.04), "VCD": (2, 0.20), "M3ID": (2, 0.20), "SCI$_5$": (5, 0.33)},
}
# measured natural overall passes (MCQ 10630 + Oth 2619 row-weighted) and deltas
NAT_MAGIC = {"qwen": (1.79, 0.78), "llava": (2.45, 1.26)}


def released_curve(mk):
    """Anytime trajectory of the released ladder on the biased BS rows:
    (avg passes, BS-All accuracy) at base / +rung1 / +rung2."""
    with gzip.open(ROOT / "magic_mcq_bundle.json.gz", "rt") as f:
        mcq = json.load(f)["mcq"][mk]
    with gzip.open(ROOT / "magic_oth_bundle.json.gz", "rt") as f:
        oth = json.load(f)[mk]
    cfg_m, cfg_o = M.LADDER_MCQ[mk], M.LADDER_OTH[mk]
    stats = {lv: [0, 0.0] for lv in ("base", "r1", "full")}  # hits, passes

    def tally(level, ok, p):
        stats[level][0] += ok
        stats[level][1] += p

    n = 0
    for row in mcq:
        gold = str(row["answer"]).strip().upper()
        s0, dd, ad, vc = row["s0"], row["default_delta"], row["answer_delta"], row["selected_vc"]
        lift = [max(dd[i], ad[i]) for i in range(len(s0))]
        resid = [s0[i] - vc[i] for i in range(len(s0))]
        f1 = M.margin(s0) <= cfg_m.theta1
        st1 = lift if f1 else s0
        f2 = M.margin(st1) <= cfg_m.rung2.theta
        st2 = st1
        if f2:
            cand = M._add(st1, resid)
            if M.margin(cand) - M.margin(st1) >= cfg_m.rung2.tau:
                st2 = cand
        tally("base", row["labels"][M.argmax_key(s0)] == gold, 1)
        tally("r1", row["labels"][M.argmax_key(st1)] == gold, 1 + 3 * f1)
        tally("full", row["labels"][M.argmax_key(st2)] == gold, 1 + 3 * f1 + 3 * f2)
        n += 1
    for row in oth:
        pool = row["pool"]
        rd, ra, bd, ba, vg = (row["real_def"], row["real_ans"], row["blank_def"],
                              row["blank_ans"], row["vc_gray"])
        base = {c: max(rd[c], ra[c]) for c in pool}
        lift = {c: max(rd[c] - bd[c], ra[c] - ba[c]) for c in pool}
        resid = {c: base[c] - vg[c] for c in pool}
        s0 = dict(rd)
        f1 = M.margin(s0) <= cfg_o.theta1
        st1 = lift if f1 else s0
        f2 = M.margin(st1) <= cfg_o.rung2.theta
        st2 = st1
        if f2:
            cand = M._add(st1, resid)
            if M.margin(cand) - M.margin(st1) >= cfg_o.rung2.tau:
                st2 = cand
        tally("base", M._open_hit(row["answer"], M.argmax_key(s0)), 2)
        tally("r1", M._open_hit(row["answer"], M.argmax_key(st1)), 2 + 2 * f1)
        tally("full", M._open_hit(row["answer"], M.argmax_key(st2)), 2 + 2 * f1 + 1 * f2)
        n += 1
    return [(stats[lv][1] / n, 100.0 * stats[lv][0] / n) for lv in ("base", "r1", "full")]


def style_ax(ax, ygrid_only=True):
    """House frame: four black spines, dashed grid behind the marks, no ticks."""
    fs.frame(ax, grid="y" if ygrid_only else "both", nbins=(4, 5))


def draw_biased_panel(ax, mk, curve, annotate_magic=True, label_points=True):
    pts = PUB_BS[mk]
    order = ["SCI$_3$", "SCI$_5$", "SCI$_7$"]
    ax.plot([pts[k][0] for k in order], [pts[k][1] for k in order],
            "o--", color=GRAY, lw=1.1, ms=4, mfc="white", mew=1.1,
            label="SCI (fixed budget)", zorder=2)
    if label_points:
        for k, dy in zip(order, (-11, -11, -11)):
            ax.annotate(k, pts[k], textcoords="offset points", xytext=(3, dy),
                        fontsize=7, color=GRAY)
    single = [pts[k] for k in ("TIE", "VCD", "M3ID")]
    ax.scatter([p[0] for p in single], [p[1] for p in single], color=LGRAY,
               marker="s", s=16, zorder=1, label="single-CF baselines")
    # our own cheapest gated arm: the reviewer-facing Pareto challenger
    gt = {"qwen": (1.9, 29.29), "llava": (1.9, 30.00)}[mk]
    ax.scatter([gt[0]], [gt[1]], color="#B5523E", marker="^", s=34, zorder=3,
               label="gated-TIE (ours, gate only)")
    ax.annotate("gated-TIE", gt, textcoords="offset points", xytext=(5, -3),
                fontsize=7, color="#B5523E", ha="left")
    xs, ys = zip(*curve)
    ax.plot(xs, ys, "o-", color=BLUE, lw=2.0, ms=4.5, label="MAGIC (ours)", zorder=3)
    ax.scatter([xs[-1]], [ys[-1]], color=BLUE, s=58, zorder=4,
               edgecolor="white", linewidth=1.1)
    if annotate_magic:
        ax.annotate("MAGIC", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(-4, 8), fontsize=8, color=BLUE, weight="bold",
                    ha="center")
    ax.set_xlabel("average forward passes")
    ax.set_xlim(0.5, 7.8)
    style_ax(ax)


# ------------------------------------------------------- teaser scatter --
PUB_BS_ALL = {  # published biased BS Overall @ nominal passes (tab:main)
    "qwen": {"Base": (1, 14.52), "TIE": (2, 22.32), "VCD": (2, 23.12),
             "M3ID": (2, 25.68), "SCI$_3$": (3, 26.94), "SCI$_5$": (5, 29.50),
             "SCI$_7$": (7, 31.72)},
}


def overall_frontier(mk):
    """Sweep t1 for BOTH answer formats together over the frozen Test
    bundles and return (mean passes, BS Overall) per value, noadm ladder.
    At the deployed t1 this must reproduce the published cells."""
    import gzip as _gz
    with _gz.open(ROOT / "magic_mcq_bundle.json.gz", "rt") as f:
        mrows = [r for r in json.load(f)["mcq"][mk] if r["mode"] in M.GROUPS["BS"]]
    with _gz.open(ROOT / "magic_oth_bundle.json.gz", "rt") as f:
        orows = [r for r in json.load(f)[mk] if r["mode"] in M.GROUPS["BS"]]
    cfg_m, cfg_o = M.LADDER_MCQ[mk], M.LADDER_OTH[mk]
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.25]
    curve = []
    for t1 in grid:
        hits, passes = [], []
        for row in mrows:
            s0, dd, ad, vc = (row["s0"], row["default_delta"],
                              row["answer_delta"], row["selected_vc"])
            lift = [max(dd[i], ad[i]) for i in range(len(s0))]
            resid = [s0[i] - vc[i] for i in range(len(s0))]
            st, f1, f2 = s0, False, False
            if M.margin(st) <= t1:
                st, f1 = lift, True
            if M.margin(st) <= cfg_m.rung2.theta:
                f2 = True
                cand = M._add(st, resid)
                if M.margin(cand) - M.margin(st) >= cfg_m.rung2.tau:
                    st = cand
            hits.append(row["labels"][M.argmax_key(st)]
                        == str(row["answer"]).strip().upper())
            passes.append(1 + 3 * f1 + 3 * f2)
        for row in orows:
            pool = row["pool"]
            rd, ra, bd, ba, vg = (row["real_def"], row["real_ans"],
                                  row["blank_def"], row["blank_ans"],
                                  row["vc_gray"])
            base = {c: max(rd[c], ra[c]) for c in pool}
            lift = {c: max(rd[c] - bd[c], ra[c] - ba[c]) for c in pool}
            resid = {c: base[c] - vg[c] for c in pool}
            st, f1, f2 = dict(rd), False, False
            if M.margin(st) <= t1:
                st, f1 = lift, True
            if M.margin(st) <= cfg_o.rung2.theta:
                f2 = True
                cand = M._add(st, resid)
                if M.margin(cand) - M.margin(st) >= cfg_o.rung2.tau:
                    st = cand
            hits.append(M._open_hit(row["answer"], M.argmax_key(st)))
            passes.append(2 + 2 * f1 + 1 * f2)
        curve.append((sum(passes) / len(passes),
                      100 * sum(hits) / len(hits), t1))
    return curve


def teaser():
    """Right-hand panel of the teaser: the accuracy-compute frontier traced by
    sweeping the allocation threshold, against the fixed-budget baselines."""
    # Pareto frontier: MAGIC is a tunable curve, every baseline a fixed-budget
    # point. Teaser-safe vocabulary only (no t1 / deployed / ablation jargon);
    # all series direct-labeled, no legend box.
    fig, ax = plt.subplots(figsize=(2.95, 1.95))
    mk = "qwen"
    pts = PUB_BS_ALL[mk]
    # BS Overall frontier: the same population and metric as Table 2
    curve = overall_frontier(mk)
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    # anchor: the uncorrected base model
    ax.scatter([pts["Base"][0]], [pts["Base"][1]], s=20, facecolor="white",
               edgecolor=fs.DKGRAY, linewidth=1.0, zorder=2)
    ax.annotate("Base", pts["Base"], textcoords="offset points",
                xytext=(5, -2), fontsize=7.6, color=fs.DKGRAY, ha="left")
    # the MAGIC frontier: measured operating points joined by a line
    ax.plot(xs, ys, "-", color=BLUE, lw=1.6, zorder=3,
            solid_capstyle="round")
    ax.scatter(xs[:-1], ys[:-1], s=7, color=BLUE, zorder=3.5)
    # baselines all in grayscale so the single accent colour stays on MAGIC:
    # SCI family = dark gray filled circles, the key comparison
    DK, MD = fs.DKGRAY, fs.MDGRAY
    order = ["SCI$_3$", "SCI$_5$", "SCI$_7$"]
    ax.plot([pts[k][0] for k in order], [pts[k][1] for k in order],
            "--", color=DK, lw=1.0, dashes=(3, 2.0), zorder=2)
    ax.scatter([pts[k][0] for k in order], [pts[k][1] for k in order],
               s=20, color=DK, edgecolor="white", linewidth=0.7,
               zorder=2.5)
    # SCI_3 sits where our frontier crosses the SCI line, so its label goes
    # above the point; below is the frontier and left is the M3ID marker
    for k, off, ha in zip(order, ((-2, 6), (1, -11), (1, -11)),
                          ("center", "left", "left")):
        ax.annotate(k, pts[k], textcoords="offset points", xytext=off,
                    fontsize=7.6, color=DK, ha=ha)
    # single-branch correctors = medium gray squares, minor context
    singles = [("TIE", 21.9), ("VCD", 23.4), ("M3ID", 25.9)]
    ax.scatter([pts[k][0] for k, _ in singles],
               [pts[k][1] for k, _ in singles], color=MD, marker="s",
               s=16, edgecolor="white", linewidth=0.6, zorder=1)
    # TIE (22.32) and VCD (23.12) sit 0.8 apart, so their labels need a
    # vertical stagger on top of the horizontal offset or they overprint.
    for (k, _), dy in zip(singles, (-6.0, 4.5, 0.5)):
        ax.annotate(k, pts[k], textcoords="offset points", xytext=(-6, dy),
                    fontsize=7.6, color=fs.MDGRAY, ha="right", va="center")
    # the operating point chosen on validation ends the sweep
    dep = curve[-1]
    ax.scatter([dep[0]], [dep[1]], color=BLUE, s=150, marker="*", zorder=4,
               edgecolor="white", linewidth=0.7)
    ax.annotate("MAGIC (ours)", (dep[0], dep[1]), textcoords="offset points",
                xytext=(8, -3), fontsize=8.2, color=BLUE, weight="bold",
                ha="left")
    ax.set_xlabel("average forward passes")
    ax.set_ylabel("BS Overall accuracy (%)")
    ax.set_title("Qwen2-VL-7B, DRBench biased subset", loc="left",
                 fontsize=8, color="#333333", pad=5)
    ax.set_xlim(0.4, 9.1)
    ax.set_ylim(12.6, 36.8)
    style_ax(ax)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_yticks([15, 20, 25, 30, 35])
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR / "teaser_scatter.pdf")
    plt.close(fig)
    print("wrote teaser_scatter.pdf")


# ---------------------------------------------------------- marginlaw ----
def marginlaw():
    d = json.loads((ROOT / "experiments/data/cost_profile/marginlaw_pop.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(3.05, 1.68), sharey=True)
    for ax, mk in zip(axes, ("qwen", "llava")):
        pb = d[mk]["biased"]
        pn = d[mk]["natural"]
        ax.fill_between(pb["m"], 0, np.asarray(pb["u"]) * 100,
                        color=BLUE, alpha=0.10, lw=0)
        ax.plot(pb["m"], np.asarray(pb["u"]) * 100, "o-", color=BLUE,
                lw=1.8, ms=3, label="biased")
        ax.plot(pn["m"], np.asarray(pn["u"]) * 100, "s-", color=VERM,
                lw=1.8, ms=3, label="natural")
        ax.axhline(0, color="0.35", lw=0.8)
        t1 = {"qwen": 3.25, "llava": 5.03}[mk]
        ax.axvline(t1, color="0.45", ls=":", lw=1.0)
        ax.text(t1, 0.94, " $t_1$", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color="0.35", va="top", ha="left")
        ax.set_title(TITLE[mk])
        ax.set_xlabel("base decision margin", labelpad=2)
        style_ax(ax)
    axes[0].set_ylabel("correction $\\Delta$acc (%)")
    axes[0].legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGDIR / "marginlaw_pop.pdf")
    plt.close(fig)
    print("wrote marginlaw_pop.pdf")


# ----------------------------------------------------------- manifold ----
def manifold():
    SRC = ROOT / "experiments/data/realcf_probe"
    fig, axes = plt.subplots(1, 2, figsize=(3.05, 1.78))
    for ax, (mk, sfx) in zip(axes, (("qwen", ""), ("llava", "_llava"))):
        recs = json.loads((SRC / f"stage1_records{sfx}.json").read_text())
        bia = [r for r in recs if r["split"] == "biased_val"]
        pairs = [("rand_ft_b1", "blank_ft_b1", "random"),
                 ("key_ft_b1", "blank_ft_b1", "keyword"),
                 ("key_all_b2", "blank_all_b2", "keyed-full")]
        labels, reals, blanks = [], [], []
        for rc, bc, lab in pairs:
            m = [r for r in bia if f"hit_{rc}" in r and f"hit_{bc}" in r]
            reals.append(100 * np.mean([r[f"hit_{rc}"] for r in m]))
            blanks.append(100 * np.mean([r[f"hit_{bc}"] for r in m]))
            labels.append(lab)
        x = np.arange(3)
        w = 0.36
        # bars carry a black edge, as in the reference bar style
        ax.bar(x - w / 2, blanks, w, label="blank", color=fs.LTGRAY,
               edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, reals, w, label="real image", color=fs.OURS,
               edgecolor="black", linewidth=0.5)
        for i in range(3):
            ax.annotate(f"{reals[i]-blanks[i]:+.2f}", (i, max(reals[i], blanks[i]) + 1),
                        ha="center", fontsize=6.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.6, rotation=12)
        ax.set_title(TITLE[mk], fontsize=7.6)
        ax.set_ylim(0, max(max(reals), max(blanks)) + 9)
        fs.frame(ax, grid="y")
        ax.tick_params(labelsize=6.8, pad=1.5)
    axes[0].set_ylabel("CF-corrected acc (%)", fontsize=7.2)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=6.8, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.0, handlelength=1.2)
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.93])
    fig.savefig(FIGDIR / "manifold_invariance.pdf")
    plt.close(fig)
    print("wrote manifold_invariance.pdf")


# ------------------------------------------------------------- pareto ----
def t1_frontier(mk):
    """Sweep the rung-1 allocation threshold over the frozen Test bundle and
    return (mean passes, BS Overall) for each value. This traces the whole
    accuracy-compute frontier the ladder can reach; the deployed point is the
    one selected on validation and is marked separately."""
    import gzip as _gz
    with _gz.open(ROOT / "magic_mcq_bundle.json.gz", "rt") as f:
        rows = json.load(f)["mcq"][mk]
    cfg = M.LADDER_MCQ[mk]
    curve = []
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.25,
            4.0, 5.0, 6.0, 8.0, 12.0]
    for t1 in grid:
        hits = []
        passes = []
        for row in rows:
            if row["mode"] not in M.GROUPS["BS"]:
                continue
            s0, dd, ad, vc = (row["s0"], row["default_delta"],
                              row["answer_delta"], row["selected_vc"])
            lift = [max(dd[i], ad[i]) for i in range(len(s0))]
            resid = [s0[i] - vc[i] for i in range(len(s0))]
            st, f1, f2 = s0, False, False
            if M.margin(st) <= t1:
                st, f1 = lift, True
            if M.margin(st) <= cfg.rung2.theta:
                f2 = True
                cand = M._add(st, resid)
                if M.margin(cand) - M.margin(st) >= cfg.rung2.tau:
                    st = cand
            hits.append(row["labels"][M.argmax_key(st)]
                        == str(row["answer"]).strip().upper())
            passes.append(1 + 3 * f1 + 3 * f2)
        curve.append((sum(passes) / len(passes),
                      100 * sum(hits) / len(hits), t1))
    return curve


def pareto():
    """Natural-distribution accuracy against compute. The biased-subset
    frontier is the teaser panel; this is the view no table carries."""
    fig, ax = plt.subplots(figsize=(3.05, 2.05))
    for mk, marker, label in (("qwen", "o", "Qwen2-VL"),
                              ("llava", "^", "LLaVA-NeXT")):
        pts = PUB_NAT[mk]
        xs = [v[0] for v in pts.values()]
        ys = [v[1] for v in pts.values()]
        ax.scatter(xs, ys, marker=marker, s=22, facecolors="white",
                   edgecolors=GRAY, linewidths=1.1, zorder=2, label=label)
        x, y = NAT_MAGIC[mk]
        ax.scatter(x, y, marker=marker, s=70, color=BLUE, zorder=4,
                   edgecolor="white", linewidth=1.1)
        ax.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                    xytext=(9, -3), fontsize=7.5, color=BLUE, weight="bold")
    # published SCI5 points get one shared label
    ax.annotate("SCI$_5$", PUB_NAT["qwen"]["SCI$_5$"], textcoords="offset points",
                xytext=(4, 4), fontsize=7.5, color=GRAY)
    ax.annotate("MAGIC", NAT_MAGIC["qwen"], textcoords="offset points",
                xytext=(-2, 11), fontsize=8, color=BLUE, weight="bold",
                ha="center")
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xlim(0.6, 6.4)
    ax.set_ylim(-0.35, 1.75)
    ax.set_xlabel("average forward passes")
    ax.set_ylabel("natural accuracy $\\Delta$ vs base")
    ax.legend(loc="upper right")
    style_ax(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR / "pareto.pdf")
    plt.close(fig)
    print("wrote pareto.pdf")


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = False
    teaser()
    marginlaw()
    manifold()
    pareto()
