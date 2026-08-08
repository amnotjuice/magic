"""Keystone: empirically verify the bias-variance model of the prior estimator
(Prop 2). From the dumped VCF probe logits, run the TIE-style correction
argmax(s0 - mean_k(VCF)) for k=1..7 averaged probes, measure accuracy(k) and
the cross-probe variance, and fit error(k) = bias^2 + sigma^2/k per encoder.

Analysis on Biased_Val only (no Test claim). Zero-GPU, reads frozen dumps."""
import glob, os, sys, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DUMP = Path("/data2/shunshungu/dump_tensors")
LMU = Path(os.path.expanduser("~/LMUData"))
SPLIT = "Biased_Val"

VCF = ["Color0", "Noise400", "Noise500", "Grayscale", "StrongBlur",
       "CenterMask", "HighPass"]
ENC = {
    "Qwen2-VL-7B":  {"opt": {"A":32,"B":33,"C":34,"D":35,"E":36},
                     "mcq": ["MMStar","CCBench","MMBench_DEV_EN_V11","MMBench_DEV_CN_V11"]},
    "LLaVA-NeXT-8B":{"opt": None,  # filled from tokenizer below
                     "mcq": ["MMStar","CCBench","MMBench_DEV_EN_V11","MMBench_DEV_CN_V11"]},
}

def opt_ids_llava():
    from transformers import AutoTokenizer
    for mp in ["/data/shunshungu/models/llama3-llava-next-8b-hf",
               "meta-llama/Meta-Llama-3-8B-Instruct"]:
        try:
            tk = AutoTokenizer.from_pretrained(mp)
            d = {}
            for L in "ABCDE":
                ids = tk.encode(L, add_special_tokens=False)
                d[L] = ids[-1]  # last token id of the letter
            return d
        except Exception:
            continue
    return None

def avail_probes(enc, mcq):
    """VCF variants with a non-empty dump for this encoder (on the first
    MCQ dataset that exists)."""
    for ds in mcq:
        base = DUMP / f"{enc}-Original" / f"{ds}_{enc}_{SPLIT}"
        if not base.exists():
            continue
        ok = [v for v in VCF
              if list((DUMP / f"{enc}-VCF-{v}" / f"{ds}_{enc}_{SPLIT}").glob("logits_*.pt"))]
        return ok
    return []

def load_dataset(enc, ds, opt, probes_list):
    tsv = LMU / f"{ds}_{enc}_{SPLIT}.tsv"
    ddir = lambda v: DUMP / f"{enc}-VCF-{v}" / f"{ds}_{enc}_{SPLIT}"
    base_dir = DUMP / f"{enc}-Original" / f"{ds}_{enc}_{SPLIT}"
    if not tsv.exists() or not base_dir.exists():
        return None
    df = pd.read_csv(tsv, sep="\t")
    n = len(df)
    rows = []
    for i in range(n):
        opts = [L for L in "ABCDE" if L in df.columns and pd.notna(df.iloc[i].get(L))]
        if len(opts) < 2:
            continue
        ans = str(df.iloc[i]["answer"]).strip().upper()
        if ans not in opts:
            continue
        ids = [opt[L] for L in opts]
        bf = base_dir / f"logits_{i}.pt"
        if not bf.exists():
            continue
        try:
            s0 = torch.load(bf, map_location="cpu")[0, ids].float().numpy()
            probes = []
            ok = True
            for v in probes_list:
                pf = ddir(v) / f"logits_{i}.pt"
                if not pf.exists():
                    ok = False; break
                probes.append(torch.load(pf, map_location="cpu")[0, ids].float().numpy())
            if not ok:
                continue
        except Exception:
            continue
        rows.append({"opts": opts, "ans": opts.index(ans),
                     "s0": s0, "probes": np.stack(probes)})  # probes: [7, n_opt]
    return rows

def analyze(enc, cfg):
    opt = cfg["opt"]
    probes_list = avail_probes(enc, cfg["mcq"])
    rows = []
    for ds in cfg["mcq"]:
        r = load_dataset(enc, ds, opt, probes_list)
        if r: rows += r
    if not rows:
        return None
    K = len(probes_list)
    # accuracy(k): mean over all C(7,k) subsets of argmax(s0 - mean_subset(probes))
    acc = {}
    for k in range(1, K+1):
        subs = list(itertools.combinations(range(K), k))
        hits = 0.0
        for r in rows:
            s0, P, a = r["s0"], r["probes"], r["ans"]
            sacc = 0.0
            for sub in subs:
                hk = P[list(sub)].mean(0)
                sacc += (np.argmax(s0 - hk) == a)
            hits += sacc / len(subs)
        acc[k] = 100.0 * hits / len(rows)
    base_acc = 100.0 * np.mean([np.argmax(r["s0"]) == r["ans"] for r in rows])
    # cross-probe variance sigma^2 (centered option logits), and sigma^2/k check
    var_probe = np.mean([r["probes"].var(0, ddof=1).mean() for r in rows])
    # fit error(k)=a+b/k  (error = 100-acc)
    ks = np.array(sorted(acc)); err = np.array([100-acc[k] for k in ks])
    X = np.vstack([np.ones_like(ks, float), 1.0/ks]).T
    coef, *_ = np.linalg.lstsq(X, err, rcond=None)
    a, b = coef
    pred = X @ coef
    ss_res = ((err-pred)**2).sum(); ss_tot = ((err-err.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    return {"n": len(rows), "base_acc": round(base_acc,2),
            "acc_k": {k: round(v,2) for k,v in acc.items()},
            "sigma2_probe": round(float(var_probe),4),
            "fit_bias_floor_a": round(float(a),3), "fit_var_b": round(float(b),3),
            "fit_r2": round(float(r2),4)}

if __name__ == "__main__":
    ENC["LLaVA-NeXT-8B"]["opt"] = opt_ids_llava()
    out = {}
    for enc, cfg in ENC.items():
        if cfg["opt"] is None:
            print(f"{enc}: no tokenizer, skip"); continue
        res = analyze(enc, cfg)
        out[enc] = res
        print(f"\n=== {enc} ===")
        if res: print(json.dumps(res, indent=2))
        else: print("no data")
    Path("experiments/data").mkdir(exist_ok=True, parents=True)
    Path("experiments/data/biasvar.json").write_text(json.dumps(out, indent=2))
