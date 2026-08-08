"""Figure 3, redesigned: correction ERROR against the probe count k, so that
Eq. (3)'s two terms are both directly visible -- the fitted asymptote IS the
bias floor ||b||^2 and the vertical distance from the curve to it is the
variance term sigma^2/k. Adds row-level 95% CIs and a shared y-range so that
'rises' vs 'flat' is a real comparison rather than an auto-scaling artefact.

Same measurement and same probe sets as experiments/biasvar.py (Biased_Val
only, zero GPU, reads the frozen VCF dumps), with per-row outcomes retained
so the CI can be computed."""
import glob, itertools, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DUMP = Path("/data2/shunshungu/dump_tensors")
LMU = Path(os.path.expanduser("~/LMUData"))
SPLIT = "Biased_Val"
VCF = ["Color0", "Noise400", "Noise500", "Grayscale", "StrongBlur",
       "CenterMask", "HighPass"]
MCQ = ["MMStar", "CCBench", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11"]
BLUE, VERM = "#0072B2", "#D55E00"


def avail(enc):
    return [v for v in VCF
            if list((DUMP / f"{enc}-VCF-{v}" / f"{MCQ[0]}_{enc}_{SPLIT}").glob("logits_*.pt"))]


def opt_ids_llava():
    from transformers import AutoTokenizer
    for mp in ["/data/shunshungu/models/llama3-llava-next-8b-hf",
               "meta-llama/Meta-Llama-3-8B-Instruct"]:
        try:
            tk = AutoTokenizer.from_pretrained(mp)
            return {L: tk.encode(L, add_special_tokens=False)[0] for L in "ABCDE"}
        except Exception:
            continue
    return None


def load(enc, opt, probes):
    import pandas as pd, torch
    rows = []
    for ds in MCQ:
        tsv = LMU / f"{ds}_{enc}_{SPLIT}.tsv"
        bdir = DUMP / f"{enc}-Original" / f"{ds}_{enc}_{SPLIT}"
        if not tsv.exists() or not bdir.exists():
            continue
        df = pd.read_csv(tsv, sep="\t")
        for i in range(len(df)):
            opts = [L for L in "ABCDE" if L in df.columns and pd.notna(df.iloc[i].get(L))]
            ans = str(df.iloc[i]["answer"]).strip().upper()
            if len(opts) < 2 or ans not in opts or not (bdir / f"logits_{i}.pt").exists():
                continue
            ids = [opt[L] for L in opts]
            try:
                s0 = torch.load(bdir / f"logits_{i}.pt", map_location="cpu")[0, ids].float().numpy()
                P = []
                for v in probes:
                    pf = DUMP / f"{enc}-VCF-{v}" / f"{ds}_{enc}_{SPLIT}" / f"logits_{i}.pt"
                    if not pf.exists():
                        P = None; break
                    P.append(torch.load(pf, map_location="cpu")[0, ids].float().numpy())
                if P is None:
                    continue
            except Exception:
                continue
            rows.append((s0, np.stack(P), opts.index(ans)))
    return rows


def analyse(rows, K):
    """per-row hit RATE at each k (averaged over all C(K,k) probe subsets)."""
    per_row = {k: [] for k in range(1, K + 1)}
    for k in range(1, K + 1):
        subs = list(itertools.combinations(range(K), k))
        for s0, P, a in rows:
            per_row[k].append(np.mean([np.argmax(s0 - P[list(s)].mean(0)) == a for s in subs]))
    ks = np.arange(1, K + 1)
    R = {k: np.asarray(per_row[k]) for k in ks}
    acc = np.array([100 * R[k].mean() for k in ks])
    # The comparison across k is PAIRED (identical rows), so the uncertainty that
    # matters is that of the difference against k=1, not the level of each point.
    ci = np.array([0.0 if k == 1 else
                   100 * 1.96 * np.std(R[k] - R[1], ddof=1) / np.sqrt(len(R[k]))
                   for k in ks])
    err = 100 - acc
    X = np.vstack([np.ones_like(ks, float), 1.0 / ks]).T
    (a_, b_), *_ = np.linalg.lstsq(X, err, rcond=None)
    pred = X @ np.array([a_, b_])
    ssr = ((err - pred) ** 2).sum(); sst = ((err - err.mean()) ** 2).sum()
    return {"ks": ks.tolist(), "err": err.round(3).tolist(), "ci95": ci.round(3).tolist(),
            "a": round(float(a_), 3), "b": round(float(b_), 3),
            "r2": round(float(1 - ssr / sst) if sst > 0 else float("nan"), 4),
            "n": len(rows), "K": K}


def main():
    encs = [("Qwen2-VL-7B", {L: i for L, i in zip("ABCDE", [32, 33, 34, 35, 36])}, BLUE),
            ("LLaVA-NeXT-8B", opt_ids_llava(), VERM)]
    res = {}
    for enc, opt, _ in encs:
        if opt is None:
            print(f"{enc}: no tokenizer"); continue
        pr = avail(enc)
        res[enc] = analyse(load(enc, opt, pr), len(pr))
        res[enc]["probes"] = pr
        print(enc, json.dumps({k: v for k, v in res[enc].items() if k != "probes"}))
        sys.stdout.flush()

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "font.family": "serif", "mathtext.fontset": "cm"})
    # Plot the error REDUCTION relative to k=1 rather than the raw error, so
    # both encoders start at 0 and the quantity Eq. (3) is about -- how much
    # variance sits above the floor -- is read off directly as the depth each
    # curve reaches.  Sharing a raw-error axis instead flattens LLaVA-NeXT
    # (0.7 pp of range) against Qwen2-VL (3.6 pp) and hides the contrast.
    drop = {e: np.array(res[e]["err"])[0] - np.array(res[e]["err"])
            for e, _, _ in encs}
    hi = max(max(d) + max(res[e]["ci95"]) for e, d in drop.items()) + 0.35
    fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.85), sharey=True)
    for ax, (enc, _, col) in zip(axes, encs):
        r = res[enc]
        ks = np.array(r["ks"]); ci = np.array(r["ci95"]); d = drop[enc]
        kk = np.linspace(1, ks.max(), 200)
        # (a + b/1) - a : all the error averaging can ever remove.  This is
        # the REMOVABLE part, not the floor a, which sits off-plot.
        removable = r["b"]
        ax.axhline(removable, color="0.45", ls=(0, (3, 2)), lw=0.9, zorder=1)
        ax.plot(kk, r["b"] - r["b"] / kk, "-", color=col, lw=1.2, alpha=0.55,
                zorder=2)
        ax.errorbar(ks, d, yerr=ci, fmt="o", color=col, ms=3.0, mew=0,
                    elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)
        ax.axhline(0.0, color="0.75", lw=0.6, zorder=0)
        ax.set_ylim(-hi * 0.18, hi)
        ax.set_xticks(ks)
        ax.set_xlabel("probes averaged $k$", labelpad=1.5, fontsize=7.5)
        ax.set_title(enc, fontsize=7.5, pad=3)
        ax.tick_params(length=2.4, width=0.6, pad=1.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("error removed vs $k{=}1$ (pp)", fontsize=7.5)
    fig.tight_layout(pad=0.35)
    fig.savefig(ROOT / "paper/figures/biasvar.pdf", bbox_inches="tight")
    (ROOT / "experiments/data/biasvar_ci.json").write_text(json.dumps(res, indent=2))
    print("wrote paper/figures/biasvar.pdf")


if __name__ == "__main__":
    main()
