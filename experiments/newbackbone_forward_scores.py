"""N1 forwards: new backbones on {old-Qwen biased MCQ Test rows} ∪
{MMStar full} — 7 scoring views per row (NAT-parity construction).
Pre-registration: experiments/data/newbackbone_forward_cache/FINDINGS.md.

  --model qwen25|onevision   which new backbone
  --run                      forwards (checkpointed jsonl)
  --report                   P-N1/P-N2/P-N3 readouts (offline)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import natural_mcq_eval_qwen as NAT  # noqa: E402

OUT = ROOT / "experiments/data/newbackbone_forward_cache"
OUT.mkdir(parents=True, exist_ok=True)
LMU = Path.home() / "LMUData"

MODEL = "qwen25"
for cand in ("qwen25", "onevision", "qwen3", "ov15"):
    if cand in sys.argv:
        MODEL = cand
MP = {"qwen25": "/data2/shunshungu/models/Qwen2.5-VL-7B-Instruct",
      "onevision": "/data2/shunshungu/models/llava-onevision-qwen2-7b-ov-hf",
      "qwen3": "/data2/shunshungu/models/Qwen3-VL-8B-Instruct",
      "ov15": "/data2/shunshungu/models/LLaVA-OneVision-1.5-8B-Instruct"}[MODEL]
CKPT = OUT / f"forwards_{MODEL}.jsonl"
BIASED_TSVS = [f"{d}_Qwen2-VL-7B_Biased_Test.tsv" for d in
               ("MMStar", "CCBench", "MMBench_DEV_CN_V11", "MMBench_DEV_EN_V11")]
VIEWS = ("orig", "blank", "answer_real", "answer_blank",
         "gray", "strongblur", "centermask")
# quantile positions of the deployed design (param ledger) for P-N3:
QPOS = dict(t1=95.0, tauT=5.0, t2=30.0, tauV=70.0)  # mid-points of the two backbones' measured positions


def load_rows():
    rows, seen = [], set()
    for tsv in BIASED_TSVS:
        df = pd.read_csv(LMU / tsv, sep="\t")
        ds = tsv.split("_Qwen2")[0]
        for _, r in df.iterrows():
            labels = [L for L in "ABCD" if L in df.columns and pd.notna(r.get(L))
                      and str(r.get(L)).strip().lower() not in ("", "nan")]
            if len(labels) < 2:
                continue
            key = (ds, str(r["index"]))
            seen.add(key)
            rows.append(dict(pop="biased", dataset=ds, index=str(r["index"]),
                             row={**{L: r[L] for L in labels},
                                  "question": str(r["question"]), "image": r["image"]},
                             labels=labels, answer=str(r["answer"]).strip()))
    df = pd.read_csv(LMU / "MMStar.tsv", sep="\t")
    for _, r in df.iterrows():
        key = ("MMStar", str(r["index"]))
        if key in seen:
            continue
        labels = [L for L in "ABCD" if L in df.columns and pd.notna(r.get(L))
                  and str(r.get(L)).strip().lower() not in ("", "nan")]
        if len(labels) < 2:
            continue
        rows.append(dict(pop="natural", dataset="MMStar", index=str(r["index"]),
                         row={**{L: r[L] for L in labels},
                              "question": str(r["question"]), "image": r["image"]},
                         labels=labels, answer=str(r["answer"]).strip()))
    return rows


def _load_model():
    if MODEL == "qwen25":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(MP, min_pixels=1280 * 28 * 28,
                                             max_pixels=16384 * 28 * 28)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MP, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    elif MODEL == "qwen3":  # nbenv only (transformers>=4.57)
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(MP, min_pixels=1280 * 28 * 28,
                                             max_pixels=16384 * 28 * 28)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MP, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    elif MODEL == "ov15":  # nbenv only (trust_remote_code)
        from transformers import AutoModelForCausalLM, AutoProcessor
        proc = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MP, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    else:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        proc = AutoProcessor.from_pretrained(MP)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MP, torch_dtype=torch.bfloat16).to("cuda").eval()
    return proc, model


@torch.no_grad()
def first_logits(model, proc, image, prompt):
    if MODEL in ("qwen25", "qwen3"):
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
    else:
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = proc(images=image, text=text, return_tensors="pt").to("cuda")
    return model(**inp).logits[0, -1, :].float().cpu()


def run():
    proc, model = _load_model()
    tok = proc.tokenizer
    rows = load_rows()
    done = set()
    if CKPT.exists():
        with CKPT.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["dataset"], r["index"]))
    print(f"{MODEL}: {len(rows)} rows, {len(done)} done", flush=True)
    with CKPT.open("a") as out:
        for i, r in enumerate(rows):
            if (r["dataset"], r["index"]) in done:
                continue
            img = NAT.load_image(r["row"])
            if img is None:
                continue
            labels = r["labels"]
            ids = {L: NAT.letter_token_ids(tok, L) for L in labels}
            p_def = NAT.build_prompt(r["row"], labels)
            p_ans = NAT.prompt_answer_format(r["row"], labels)
            logits = {
                "orig": first_logits(model, proc, NAT.transform_image(img, "default", 0), p_def),
                "blank": first_logits(model, proc, NAT.transform_image(img, "blank", 0), p_def),
                "answer_real": first_logits(model, proc, NAT.transform_image(img, "default", 0), p_ans),
                "answer_blank": first_logits(model, proc, NAT.transform_image(img, "blank", 0), p_ans),
                "gray": first_logits(model, proc, NAT.transform_image(img, "gray", 0), p_def),
                "strongblur": first_logits(model, proc, NAT.transform_image(img, "strongblur", 0), p_def),
                "centermask": first_logits(model, proc, NAT.transform_image(img, "centermask", 0), p_def),
            }
            rec = {"dataset": r["dataset"], "index": r["index"], "pop": r["pop"],
                   "labels": labels, "answer": r["answer"]}
            for v in VIEWS:
                lp = NAT.option_logprobs(logits[v], labels, ids)
                rec[v] = [float(x) for x in lp]
            rec["iic"] = {s: NAT.maxp(logits["orig"]) - NAT.maxp(logits[s])
                          for s in ("gray", "strongblur", "centermask")}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 50 == 0:
                print(f"progress {i+1}/{len(rows)}", flush=True)
    print("DONE forwards", flush=True)


# ---------------- offline readouts ----------------
def evidence(rec):
    t = lambda k: torch.tensor(rec[k], dtype=torch.float64)  # noqa: E731
    cent = NAT.centered
    s0, blank = t("orig"), t("blank")
    ar, ab = t("answer_real"), t("answer_blank")
    base = cent(s0)
    clean2 = cent(torch.maximum(s0 - blank, ar - ab))
    agree = float(int(cent(s0 - blank).argmax()) == int(cent(ar - ab).argmax()))
    src = min(rec["iic"], key=rec["iic"].get)
    resid = cent(s0 - t(src))
    prior = float(torch.topk(cent(blank), 2).values.diff().abs()) if len(rec["labels"]) > 1 else 0.0
    m = lambda x: float(torch.topk(x, 2).values.diff().abs())  # noqa: E731
    return base, clean2, agree, resid, prior, m


def report():
    recs = []
    with CKPT.open() as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    print(f"{MODEL}: n={len(recs)} "
          f"(biased {sum(r['pop']=='biased' for r in recs)}, "
          f"natural {sum(r['pop']=='natural' for r in recs)})")
    # P-N1: base accuracy on old prior-conflict rows
    for pop in ("biased", "natural"):
        sub = [r for r in recs if r["pop"] == pop]
        if not sub:
            continue
        acc = 100.0 * np.mean([r["labels"][int(np.argmax(r["orig"]))] == r["answer"] for r in sub])
        print(f"  P-N1 base acc [{pop}]: {acc:.2f} (n={len(sub)})")
    # P-N2: margin law per population (ungated clean2)
    for pop in ("biased", "natural"):
        sub = [r for r in recs if r["pop"] == pop]
        if len(sub) < 100:
            continue
        pts = []
        for r in sub:
            base, clean2, agree, resid, prior, m = evidence(r)
            hb = int(r["labels"][int(base.argmax())] == r["answer"])
            hl = int(r["labels"][int(clean2.argmax())] == r["answer"])
            pts.append((m(base), hl - hb, prior))
        pts.sort()
        nb = 5
        chunks = np.array_split(np.arange(len(pts)), nb)
        bins = [round(float(np.mean([pts[i][1] for i in c])), 4) for c in chunks]
        print(f"  P-N2 U(m) [{pop}] low→high margin bins: {bins}")
        # mediation within mid-margin
        mid = [pts[i] for c in chunks[1:4] for i in c]
        ps = np.array([p[2] for p in mid]); dd = np.array([p[1] for p in mid], float)
        qs = np.quantile(ps, [1/3, 2/3])
        med = [round(float(dd[(ps > lo) & (ps <= hi)].mean()), 4)
               for lo, hi in ((-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf))]
        print(f"       mediation (mid-margin, prior terciles): {med}")
    # P-N3: self-calibrated ladder (quantile rules, label-free on ALL rows)
    ev = {id(r): evidence(r) for r in recs}
    m0 = [ev[id(r)][5](ev[id(r)][0]) for r in recs]
    t1 = float(np.percentile(m0, QPOS["t1"]))
    admits = [ev[id(r)][5](ev[id(r)][1]) + ev[id(r)][2] for r in recs
              if ev[id(r)][5](ev[id(r)][0]) <= t1]
    tauT = float(np.percentile(admits, QPOS["tauT"]))
    post1 = []
    for r in recs:
        base, clean2, agree, resid, prior, m = ev[id(r)]
        if m(base) <= t1:
            st = clean2 if m(clean2) + agree >= tauT else base
            post1.append(m(st))
    t2 = float(np.percentile(post1, QPOS["t2"]))
    gains = []
    for r in recs:
        base, clean2, agree, resid, prior, m = ev[id(r)]
        if m(base) <= t1:
            st = clean2 if m(clean2) + agree >= tauT else base
            if m(st) <= t2:
                gains.append(m(st + resid) - m(st))
    tauV = float(np.percentile(gains, QPOS["tauV"])) if gains else 0.625
    print(f"  P-N3 self-calibrated: t1={t1:.3f} tauT={tauT:.3f} t2={t2:.3f} tauV={tauV:.3f}")
    for pop in ("natural", "biased"):
        sub = [r for r in recs if r["pop"] == pop]
        if not sub:
            continue
        hl = hb = helped = harmed = 0
        for r in sub:
            base, clean2, agree, resid, prior, m = ev[id(r)]
            st = base
            if m(st) <= t1:
                st = clean2 if m(clean2) + agree >= tauT else base
            if m(st) <= t2:
                cand = st + resid
                if m(cand) - m(st) >= tauV:
                    st = cand
            a = int(r["labels"][int(st.argmax())] == r["answer"])
            b = int(r["labels"][int(base.argmax())] == r["answer"])
            hl += a; hb += b; helped += (a and not b); harmed += (b and not a)
        n = len(sub)
        print(f"       ladder [{pop}]: base={100*hb/n:.2f} → {100*hl/n:.2f} "
              f"(Δ{100*(hl-hb)/n:+.3f}, {helped}/{harmed}, band ±{100/n:.2f})")


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    elif "--report" in sys.argv:
        report()
    else:
        print(__doc__)
