"""Figure: BS-accuracy sensitivity to the rung-1 threshold t1."""
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
d = json.loads(Path("experiments/data/t1_sweep.json").read_text())
col = {"qwen": "#c0504d", "llava": "#4472a8"}
disp = {"qwen": "Qwen2-VL-7B", "llava": "LLaVA-NeXT-8B"}

fig, ax = plt.subplots(figsize=(3.0, 2.1))
for mk in ("qwen", "llava"):
    sw = d[mk]["sweep"]
    ts = np.array(sorted(float(t) for t in sw))
    ys = np.array([sw[str(t)] if str(t) in sw else sw[t] for t in
                   [k for k in sorted(sw, key=float)]])
    ts = np.array(sorted(float(t) for t in sw)); ys = np.array([sw[k] for k in sorted(sw, key=float)])
    ax.plot(ts, ys, "-o", color=col[mk], ms=3, lw=1.1, label=disp[mk])
    dt = d[mk]["deployed_t1"]
    dy = sw[min(sw, key=lambda k: abs(float(k)-dt))]
    ax.plot([dt], [dy], "*", color=col[mk], ms=9, mec="k", mew=0.4)
ax.set_xlabel("rung-1 threshold $t_1$")
ax.set_ylabel("BS accuracy (\\%)")
ax.legend(fontsize=6.5, frameon=False, loc="lower right")
ax.text(0.03, 0.96, "$\\star$ deployed", transform=ax.transAxes, fontsize=6.5,
        va="top")
fig.tight_layout(pad=0.3)
out = Path("figures/sensitivity.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
