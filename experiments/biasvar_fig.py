"""Figure: bias-variance verification of Prop 2. acc(k) with fitted
100-(a+b/k) curve, per encoder."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6, "font.family": "serif",
    "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
})
res = json.loads(Path("experiments/data/biasvar.json").read_text())
enc_disp = {"Qwen2-VL-7B": "Qwen2-VL-7B", "LLaVA-NeXT-8B": "LLaVA-NeXT-8B"}
col = {"Qwen2-VL-7B": "#c0504d", "LLaVA-NeXT-8B": "#4472a8"}

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.2), sharey=False)
for ax, (enc, r) in zip(axes, res.items()):
    ks = np.array(sorted(int(k) for k in r["acc_k"]))
    acc = np.array([r["acc_k"][str(k)] for k in ks])
    a, b = r["fit_bias_floor_a"], r["fit_var_b"]
    kk = np.linspace(ks.min(), ks.max(), 100)
    fit_acc = 100 - (a + b / kk)
    ax.plot(kk, fit_acc, "-", color=col[enc], lw=1.2, alpha=0.55,
            label=r"fit $100-(a+b/k)$")
    ax.plot(ks, acc, "o", color=col[enc], ms=4, label="measured")
    ax.set_title(enc_disp[enc], fontsize=8)
    ax.set_xlabel("branches averaged $k$")
    ax.set_xticks(ks)
    ax.text(0.96, 0.06,
            f"$\\sigma^2\\!=\\!{b:.2f}$\n$R^2\\!=\\!{r['fit_r2']:.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
axes[0].set_ylabel("BS accuracy (\\%)")
fig.tight_layout(pad=0.4)
out = Path("figures/biasvar.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
print("Qwen b/LLaVA b ratio:",
      round(res["Qwen2-VL-7B"]["fit_var_b"] / res["LLaVA-NeXT-8B"]["fit_var_b"], 2))
