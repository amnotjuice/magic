"""Option-permutation control + batched wall-clock.

PREREG: experiments/data/permutation_control/PREREG.md.

  --clock         batched wall-clock benchmark (Qwen2, ~15 min)
  --run qwen2|llava   permutation forwards on Biased_Val MCQ rows
  --read          single pre-registered readout
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import natural_mcq_eval_qwen as NAT  # noqa: E402
import magic as M  # noqa: E402

OUT = ROOT / "experiments/data/permutation_control"
OUT.mkdir(exist_ok=True)
LMU = Path.home() / "LMUData"
MCQ = ["MMStar", "CCBench", "MMBench_DEV_CN_V11", "MMBench_DEV_EN_V11"]
MODEL_NAME = {"qwen2": "Qwen2-VL-7B", "llava": "LLaVA-NeXT-8B"}
LLAVA_PATH = "/data/shunshungu/models/llama3-llava-next-8b-hf"
VIEWS = ("default", "blank", "answer_real", "answer_blank",
         "gray", "strongblur", "centermask")


def load_rows(model_name):
    rows = []
    for ds in MCQ:
        df = pd.read_csv(LMU / f"{ds}_{model_name}_Biased_Val.tsv", sep="\t")
        vcf = set(pd.read_csv(LMU / f"{ds}_{model_name}_VCF_Val.tsv",
                              sep="\t")["index"].astype(str))
        tcf = set(pd.read_csv(LMU / f"{ds}_{model_name}_TCF_Val.tsv",
                              sep="\t")["index"].astype(str))
        for _, r in df.iterrows():
            d = r.to_dict()
            idx = str(d["index"])
            iv, it = idx in vcf, idx in tcf
            d["mode"] = ("both" if iv and it else
                         "vcf_only" if iv else
                         "tcf_only" if it else "both")
            d["dataset"] = ds
            rows.append(d)
    return rows


def permute(row, labels):
    """Cyclic shift of option contents; gold letter remapped."""
    out = dict(row)
    n = len(labels)
    for i, l in enumerate(labels):
        out[l] = row[labels[(i + 1) % n]]     # new[l] <- old[next(l)]
    g = str(row["answer"]).strip().upper()
    gi = labels.index(g)
    out["answer"] = labels[(gi - 1) % n]      # old gold content sits at prev
    return out


def load_model(which):
    if which == "qwen2":
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(NAT.MODEL_PATH,
                                             min_pixels=1280 * 28 * 28,
                                             max_pixels=16384 * 28 * 28)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            NAT.MODEL_PATH, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
    else:
        from transformers import (LlavaNextForConditionalGeneration,
                                  LlavaNextProcessor)
        proc = LlavaNextProcessor.from_pretrained(LLAVA_PATH)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            LLAVA_PATH, torch_dtype=torch.float16).to("cuda").eval()
    return proc, model


@torch.no_grad()
def flog(which, model, proc, image, prompt):
    if which == "qwen2":
        return NAT.first_logits(model, proc, image, prompt)
    conv = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True)
    inp = proc(images=image, text=text, return_tensors="pt").to("cuda")
    return model(**inp).logits[0, -1, :].float().cpu()


def row_views(which, model, proc, tok, row, labels):
    """The deployed MCQ view set for one row: option scores + vocab msp."""
    img = NAT.load_image(row)
    if img is None:
        return None
    # The vision tower's attention length is pixels/14^2, so a 5 MP image
    # is a 26k-token sequence whose fp32 attention weights are 42 GiB.
    # Cap area at ~2 MP; both conditions share the image, so the
    # within-control comparison is unaffected.
    if img.width * img.height > 2_000_000:
        img = img.copy()
        img.thumbnail((1400, 1400))
    lid = {l: NAT.letter_token_ids(tok, l) for l in labels}
    p_def = NAT.build_prompt(row, labels)
    p_ans = NAT.prompt_answer_format(row, labels)
    blank = NAT.transform_image(img, "blank", 0)
    imgs = {"default": (img, p_def), "blank": (blank, p_def),
            "answer_real": (img, p_ans), "answer_blank": (blank, p_ans),
            "gray": (NAT.transform_image(img, "gray", 0), p_def),
            "strongblur": (NAT.transform_image(img, "strongblur", 0), p_def),
            "centermask": (NAT.transform_image(img, "centermask", 0), p_def)}
    out = {}
    for k, (im, pr) in imgs.items():
        lg = flog(which, model, proc, im, pr)
        out[k] = {"opt": [float(x) for x in
                          NAT.option_logprobs(lg, labels, lid)],
                  "msp": NAT.maxp(lg)}
    return out


def ladder_pred(views, labels, cfg):
    s0 = views["default"]["opt"]
    dd = [a - b for a, b in zip(s0, views["blank"]["opt"])]
    ad = [a - b for a, b in zip(views["answer_real"]["opt"],
                                views["answer_blank"]["opt"])]
    iic = {k: views["default"]["msp"] - views[k]["msp"]
           for k in ("gray", "strongblur", "centermask")}
    src = min(iic, key=iic.get)
    lift = [max(a, b) for a, b in zip(dd, ad)]
    resid = [a - b for a, b in zip(s0, views[src]["opt"])]
    state = M.run_ladder(s0, lift, resid, cfg)
    return labels[M.argmax_key(state)], labels[M.argmax_key(s0)]


def run(which):
    mk = "qwen" if which == "qwen2" else "llava"
    cfg = M.LADDER_MCQ[mk]
    rows = load_rows(MODEL_NAME[which])
    ck = OUT / f"perm_{which}.jsonl"
    done = set()
    if ck.exists():
        for line in ck.open():
            if line.strip():
                done.add(json.loads(line)["key"])
    proc, model = load_model(which)
    tok = proc.tokenizer
    print(f"{which}: {len(rows)} rows, {len(done)} done", flush=True)
    with ck.open("a") as f:
        for i, row in enumerate(rows):
            key = f"{row['dataset']}|{row['index']}"
            if key in done:
                continue
            labels = NAT.option_labels(row)
            if len(labels) < 2:
                continue
            rec = {"key": key, "mode": row["mode"]}
            for cond, r in (("orig", row), ("perm", permute(row, labels))):
                v = row_views(which, model, proc, tok, r, labels)
                if v is None:
                    rec = None
                    break
                fp, bp = ladder_pred(v, labels, cfg)
                gold = str(r["answer"]).strip().upper()
                rec[cond] = {"base_ok": bp == gold, "magic_ok": fp == gold}
            if rec:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {which} {i+1}/{len(rows)}", flush=True)
    print(f"DONE perm {which}", flush=True)


def read():
    rep = {}
    for which in ("qwen2", "llava"):
        ck = OUT / f"perm_{which}.jsonl"
        if not ck.exists():
            continue
        recs = [json.loads(l) for l in ck.open() if l.strip()]
        rep[which] = {}
        for grp, modes in (("B", {"vcf_only", "both"}),
                           ("BS", {"vcf_only", "tcf_only", "both"})):
            sub = [r for r in recs if r["mode"] in modes]
            n = len(sub)
            e = {}
            for cond in ("orig", "perm"):
                b = 100 * sum(r[cond]["base_ok"] for r in sub) / n
                m = 100 * sum(r[cond]["magic_ok"] for r in sub) / n
                e[cond] = {"base": round(b, 2), "magic": round(m, 2),
                           "delta": round(m - b, 2)}
            e["n"] = n
            e["delta_survival"] = (round(e["perm"]["delta"] /
                                         e["orig"]["delta"], 3)
                                   if e["orig"]["delta"] else None)
            rep[which][grp] = e
        print(which, json.dumps(rep[which], indent=1))
    (OUT / "readout.json").write_text(json.dumps(rep, indent=2))


# ------------------------------------------------------------- clock ------
def clock():
    which = "qwen2"
    rows = load_rows(MODEL_NAME[which])[:32]
    # Amended after an OOM: batch-of-7 at the deployed image budget
    # (max 16384x28x28) needs 41 GiB of attention workspace and does not
    # fit on one A100-80G -- itself worth reporting. The clock therefore
    # runs at a uniform reduced budget (max 5120 patches) for every arm,
    # so the ratios are budget-matched; disclosed in the PREREG.
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(NAT.MODEL_PATH,
                                         min_pixels=256 * 28 * 28,
                                         max_pixels=1024 * 28 * 28)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        NAT.MODEL_PATH, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    from qwen_vl_utils import process_vision_info

    def batch_forward(pairs):
        # qwen_vl_utils resizes with its OWN defaults unless the budget is
        # embedded in the message dict; the processor's min/max is bypassed.
        msgs = [[{"role": "user", "content": [
            {"type": "image", "image": im,
             "min_pixels": 256 * 28 * 28, "max_pixels": 1024 * 28 * 28},
            {"type": "text", "text": pr}]}] for im, pr in pairs]
        texts = [proc.apply_chat_template(m, tokenize=False,
                                          add_generation_prompt=True)
                 for m in msgs]
        imgs, vids = process_vision_info([m[0] for m in msgs])
        inp = proc(text=texts, images=imgs, videos=vids,
                   padding=True, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inp)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        del inp
        torch.cuda.empty_cache()
        return dt

    def views_for(row, labels, keys):
        img = NAT.load_image(row)
        # pre-shrink ourselves: qwen_vl_utils' smart_resize fails to cap
        # extreme-aspect images (observed 4056 patches under a 1024 budget)
        if img.width * img.height > 1024 * 784:
            img = img.copy()
            img.thumbnail((896, 896))
        p_def = NAT.build_prompt(row, labels)
        p_a = NAT.prompt_answer_format(row, labels)
        p_t1 = NAT.prompt_tcf1(row, labels)
        p_t2 = NAT.prompt_tcf2(row, labels)
        blank = NAT.transform_image(img, "blank", 0)
        table = {"default": (img, p_def), "tcf1": (img, p_t1),
                 "tcf2": (img, p_t2),
                 "color0": (blank, p_def),
                 "noise500": (NAT.transform_image(img, "noise500", 0), p_def),
                 "noise400": (NAT.transform_image(img, "noise500", 1), p_def),
                 "gray": (NAT.transform_image(img, "gray", 0), p_def),
                 "blank": (blank, p_def),
                 "answer_real": (img, p_a), "answer_blank": (blank, p_a),
                 "strongblur": (NAT.transform_image(img, "strongblur", 0), p_def),
                 "centermask": (NAT.transform_image(img, "centermask", 0), p_def)}
        return [table[k] for k in keys]

    res = {}
    arms = {
        "sci5_batched": ["default", "tcf1", "tcf2", "color0", "noise500"],
        "sci7_batched": ["default", "tcf1", "tcf2", "color0", "noise500",
                         "noise400", "gray"],
        "magic_s0": ["default"],
        "magic_rung1": ["blank", "answer_real", "answer_blank"],
        "magic_rung2": ["gray", "strongblur", "centermask"],
    }
    usable = []
    for row in rows:
        labels = NAT.option_labels(row)
        if len(labels) >= 2 and NAT.load_image(row) is not None:
            usable.append((row, labels))
        if len(usable) == 16:
            break
    # warmup
    r0, l0 = usable[0]
    batch_forward(views_for(r0, l0, arms["sci5_batched"]))
    for name, keys in arms.items():
        ts = []
        for row, labels in usable:
            reps = [batch_forward(views_for(row, labels, keys))
                    for _ in range(3)]
            ts.append(statistics.median(reps))
        res[name] = round(statistics.median(ts), 4)
        print(name, res[name], flush=True)
    # biased-MCQ firing rates from the frozen exit histogram
    f1, f2 = 0.929, 0.402
    magic_mean = res["magic_s0"] + f1 * res["magic_rung1"] + f2 * res["magic_rung2"]
    magic_worst = res["magic_s0"] + res["magic_rung1"] + res["magic_rung2"]
    res.update({"f1": f1, "f2": f2,
                "magic_mean_path": round(magic_mean, 4),
                "magic_worst_path": round(magic_worst, 4),
                "ratio_mean_vs_sci7": round(magic_mean / res["sci7_batched"], 3),
                "ratio_mean_vs_sci5": round(magic_mean / res["sci5_batched"], 3)})
    (OUT / "batched_clock.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", action="store_true")
    ap.add_argument("--run", choices=("qwen2", "llava"))
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args()
    if a.clock:
        clock()
    if a.run:
        run(a.run)
    if a.read:
        read()
