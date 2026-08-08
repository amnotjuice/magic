"""Why rung 2 routes over grayscale / strong blur / center mask.

Holds the whole ladder fixed and varies only the rung-2 source: each single
fixed counterfactual, then the deployed per-sample register of Eq. (6), then
registers that also admit the content-free probes. Each arm is compared to the
deployed register by an exact paired McNemar on discordant rows. Biased_Val
only, zero GPU, reads the frozen VCF/TCF dumps.

Writes experiments/data/rung2_source.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import magic as M  # noqa: E402

DUMP = Path("/data2/shunshungu/dump_tensors")
LMU = Path(os.path.expanduser("~/LMUData"))
SPLIT = "Biased_Val"
MCQ = ["MMStar", "CCBench", "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11"]
SRC = {"blank": "VCF-Color0", "noise": "VCF-Noise500", "gray": "VCF-Grayscale",
       "blur": "VCF-StrongBlur", "mask": "VCF-CenterMask"}
# arms: five fixed sources, the deployed register, and two wider registers
ARMS = {
    "blank": ["blank"], "noise": ["noise"], "gray": ["gray"],
    "blur": ["blur"], "mask": ["mask"],
    "deployed": ["gray", "blur", "mask"],
    "+noise": ["gray", "blur", "mask", "noise"],
    "all five": ["gray", "blur", "mask", "noise", "blank"],
}
ENC = {"qwen": "Qwen2-VL-7B", "llava": "LLaVA-NeXT-8B"}
OPT = {"qwen": {L: i for L, i in zip("ABCDE", [32, 33, 34, 35, 36])}, "llava": None}


def opt_ids_llava():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("/data/shunshungu/models/llama3-llava-next-8b-hf")
    return {L: tk.encode(L, add_special_tokens=False)[0] for L in "ABCDE"}


def center(v):
    return v - v.mean()


def mp(v):
    e = np.exp(v - v.max())
    return float((e / e.sum()).max())


def load(enc, opt):
    import pandas as pd, torch
    rows = []
    for ds in MCQ:
        tsv = LMU / f"{ds}_{enc}_{SPLIT}.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t")
        dirs = {k: DUMP / f"{enc}-{v}" / f"{ds}_{enc}_{SPLIT}" for k, v in SRC.items()}
        dirs["orig"] = DUMP / f"{enc}-Original" / f"{ds}_{enc}_{SPLIT}"
        dirs["ans_r"] = DUMP / f"{enc}-TCF-AnswerFormat" / f"{ds}_{enc}_{SPLIT}"
        dirs["ans_b"] = DUMP / f"{enc}-TCF-AnswerFormat-Blank" / f"{ds}_{enc}_{SPLIT}"
        if not all(d.exists() for d in dirs.values()):
            continue
        for i in range(len(df)):
            opts = [L for L in "ABCDE" if L in df.columns and pd.notna(df.iloc[i].get(L))]
            ans = str(df.iloc[i]["answer"]).strip().upper()
            if len(opts) < 2 or ans not in opts:
                continue
            ids = [opt[L] for L in opts]
            vec = {}
            try:
                for k, d in dirs.items():
                    f = d / f"logits_{i}.pt"
                    if not f.exists():
                        vec = None
                        break
                    vec[k] = torch.load(f, map_location="cpu")[0, ids].float().numpy()
            except Exception:
                continue
            if vec is None:
                continue
            rows.append({"ds": ds, "ans": opts.index(ans), "v": vec})
    return rows


def outcomes(rows, cfg, register):
    """Per-row hit vector for the ladder with this rung-2 register."""
    out = []
    for r in rows:
        v = r["v"]
        s0 = v["orig"]
        state = s0
        if M.margin(list(s0)) <= cfg.theta1:
            state = center(np.maximum(s0 - v["blank"], v["ans_r"] - v["ans_b"]))
        if M.margin(list(state)) <= cfg.rung2.theta:
            c = (register[0] if len(register) == 1
                 else min(register, key=lambda k: mp(s0) - mp(v[k])))
            cand = state + center(s0 - v[c])
            if M.margin(list(cand)) - M.margin(list(state)) >= cfg.rung2.tau:
                state = cand
        out.append(int(np.argmax(state) == r["ans"]))
    return np.asarray(out)


def mcnemar(a, b):
    """Exact two-sided McNemar on discordant rows, arm b against arm a."""
    w = int(((b == 1) & (a == 0)).sum())
    l = int(((b == 0) & (a == 1)).sum())
    n = w + l
    if n == 0:
        return w, l, 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return w, l, min(1.0, 2 * p)


def main():
    OPT["llava"] = opt_ids_llava()
    out = {}
    for mk in ("qwen", "llava"):
        rows = load(ENC[mk], OPT[mk])
        cfg = M.LADDER_MCQ[mk]
        hits = {a: outcomes(rows, cfg, reg) for a, reg in ARMS.items()}
        ref = hits["deployed"]
        res = {"n": len(rows)}
        for a, h in hits.items():
            w, l, p = mcnemar(ref, h)
            res[a] = {"acc": round(100.0 * h.mean(), 2),
                      "vs_deployed": None if a == "deployed" else
                      {"win": w, "loss": l, "p": round(p, 3)}}
        out[mk] = res
        print(mk, "n=%d" % len(rows))
        for a in ARMS:
            r = res[a]
            tag = "" if r["vs_deployed"] is None else \
                "   vs deployed %d:%d p=%.3f" % (r["vs_deployed"]["win"],
                                                 r["vs_deployed"]["loss"],
                                                 r["vs_deployed"]["p"])
            print("   %-9s %6.2f%s" % (a, r["acc"], tag))
        sys.stdout.flush()
    (ROOT / "experiments/data/rung2_source.json").write_text(json.dumps(out, indent=2))
    print("wrote experiments/data/rung2_source.json")


if __name__ == "__main__":
    main()
