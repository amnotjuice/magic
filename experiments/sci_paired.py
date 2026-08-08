"""In-pipeline paired McNemar: MAGIC vs SCI5 on LLaVA-NeXT MCQ-BS, both
computed from the same dumped branch logits on the same rows. Resolves the
published-point two-sample comparison. Zero-GPU. Supplementary (main table
keeps published SCI numbers)."""
import glob, os, sys, json, math
from pathlib import Path
import numpy as np, pandas as pd, torch

DUMP = Path("/data2/shunshungu/dump_tensors"); LMU = Path(os.path.expanduser("~/LMUData"))
ENC = "LLaVA-NeXT-8B"; SPLIT = "Biased_Test"
OPT = {"A":32,"B":33,"C":34,"D":35,"E":36}
DS = ["MMStar","CCBench","MMBench_DEV_EN_V11","MMBench_DEV_CN_V11"]
# MAGIC LLaVA MCQ ladder: t1=5.03125, rung2 theta=0.3, tau=0.625
T1, THETA2, TAU = 5.03125, 0.3, 0.625
GAMMA, BETA, THETA_SCI = 2.0, 0.2, 0.3  # SCI5

def load(ds, variant):
    return DUMP / f"{ENC}-{variant}" / f"{ds}_{ENC}_{SPLIT}"

def margin(v):
    s = sorted(v, reverse=True)
    return s[0]-s[1] if len(s) > 1 else 0.0

def rows_for(ds):
    tsv = LMU / f"{ds}_{ENC}_{SPLIT}.tsv"
    if not tsv.exists(): return []
    df = pd.read_csv(tsv, sep="\t")
    need = ["Original","VCF-Color0","VCF-Noise500","TCF-V1","TCF-V2",
            "TCF-AnswerFormat","TCF-AnswerFormat-Blank","VCF-Grayscale",
            "VCF-StrongBlur","VCF-CenterMask"]
    out = []
    for i in range(len(df)):
        opts = [L for L in "ABCDE" if L in df.columns and pd.notna(df.iloc[i].get(L))]
        ans = str(df.iloc[i]["answer"]).strip().upper()
        if len(opts) < 2 or ans not in opts: continue
        ids = [OPT[L] for L in opts]
        try:
            vecs = {}
            ok = True
            for v in need:
                f = load(ds, v) / f"logits_{i}.pt"
                if not f.exists(): ok = False; break
                vecs[v] = torch.load(f, map_location="cpu")[0, ids].float().numpy()
            if not ok: continue
        except Exception: continue
        out.append({"opts": opts, "ans": opts.index(ans), "v": vecs})
    return out

def magic_pred(r):
    v = r["v"]; s0 = v["Original"]
    dd = s0 - v["VCF-Color0"]
    ad = v["TCF-AnswerFormat"] - v["TCF-AnswerFormat-Blank"]
    lift = np.maximum(dd, ad)
    chans = ["VCF-Grayscale","VCF-StrongBlur","VCF-CenterMask"]
    iic = [margin(s0 - v[c]) for c in chans]
    vc = v[chans[int(np.argmin(iic))]]
    resid = s0 - vc
    state = s0.copy()
    if margin(state) <= T1: state = lift
    if margin(state) <= THETA2:
        cand = state + resid
        if margin(cand) - margin(state) >= TAU: state = cand
    return int(np.argmax(state))

def sci5_pred(r):
    v = r["v"]; s0 = v["Original"]
    tc = np.maximum.reduce([s0, v["TCF-V1"], v["TCF-V2"]]) / GAMMA
    vcf = (v["VCF-Color0"] + v["VCF-Noise500"]) / 2.0
    unb = (s0 - vcf) / BETA
    lo, hi = tc.min(), tc.max()
    thr = lo + THETA_SCI * (hi - lo)
    sci = np.where(tc < thr, -np.inf, tc + unb)
    return int(np.argmax(sci))

def base_pred(r):
    return int(np.argmax(r["v"]["Original"]))

def mcnemar(a, b):  # a,b: bool arrays (correct?)
    b01 = int(np.sum(a & ~b)); b10 = int(np.sum(~a & b))
    n = b01 + b10
    if n == 0: return b01, b10, 1.0
    # exact two-sided binomial
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(k+1)) / (2**n) * 2
    return b01, b10, min(p, 1.0)

rows = []
for ds in DS: rows += rows_for(ds)
magic_ok = np.array([magic_pred(r) == r["ans"] for r in rows])
sci_ok  = np.array([sci5_pred(r) == r["ans"] for r in rows])
base_ok = np.array([base_pred(r) == r["ans"] for r in rows])
n = len(rows)
res = {
    "n": n,
    "MAGIC_MCQ_BS": round(100*magic_ok.mean(),2),
    "SCI5_reimpl_MCQ_BS": round(100*sci_ok.mean(),2),
    "base_MCQ_BS": round(100*base_ok.mean(),2),
    "published_MAGIC": 31.13, "published_SCI5": 28.80,
}
b01, b10, p = mcnemar(magic_ok, sci_ok)
res["paired_McNemar"] = {"MAGIC_right_SCI_wrong": b01, "SCI_right_MAGIC_wrong": b10,
                         "p_two_sided": round(p,4)}
print(json.dumps(res, indent=2))
Path("experiments/data/sci_paired.json").write_text(json.dumps(res, indent=2))
