"""Phase-B B0: step0-equivalent dumps for the NEW backbones.
Pre-registration: experiments/data/newbackbone_generation/FINDINGS.md.

5 variants per row, NAT-parity construction:
  MCQ  (options given) : first-token option logits under
        orig=default/real | vcf1=default/blank | vcf2=default/noise500 |
        tcf1=NAT.prompt_tcf1/real | tcf2=NAT.prompt_tcf2/real
  MME  (binary yes/no)  : Yes/No first-token logits under
        orig=GEN default/real | vcf1=default/blank | vcf2=default/noise500 |
        tcf1=answer_format/real | tcf2=paraphrase/real
  ViLP (open)           : greedy generations (MAXNEW=12) under the same
        five combos as MME.
Disclosures: MMBench hint prepended to question when present; MCQ TCF =
the repo's SCI-template transcriptions; open-ended TCF = the deployed
pipeline's answer_format/paraphrase templates.

Usage: --model qwen3|ov15 --run | --report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import natural_mcq_eval_qwen as NAT  # noqa: E402
from newbackbone_forward_scores import _load_model, first_logits  # noqa: E402
import newbackbone_forward_scores as N1  # noqa: E402

OUT = ROOT / "experiments/data/newbackbone_generation"
OUT.mkdir(parents=True, exist_ok=True)
LMU = Path.home() / "LMUData"

MODEL = "ov15" if "ov15" in sys.argv else ("onevision" if "onevision" in sys.argv else "qwen3")
N1.MODEL = MODEL  # first_logits/_load_model dispatch
CKPT = OUT / f"b0_{MODEL}.jsonl"
MAXNEW = 12

GEN_PROMPTS = {
    # v4 protocol-faithful: OFFICIAL SCI templates (model.py prompt_variation1/2)
    "default": "{q}\nAnswer the question directly using a single word or phrase.",
    "answer_format": ("{q}\nThink about the question based on details in the given image. "
                      "Answer the question directly using a single word or phrase."),  # = official tcf_v1 (open)
    "paraphrase": ("{q}\n\u8bf7\u4ed4\u7ec6\u89c2\u5bdf\u56fe\u50cf\u4e2d\u7684\u7ec6\u8282\uff0c\u7136\u540e\u7ed3\u5408\u56fe\u50cf\u4e0a\u7684\u4fe1\u606f\u56de\u7b54\u95ee\u9898\uff0c\u8bf7\u76f4\u63a5\u7528\u4e00\u4e2a\u7b80\u77ed\u7684\u82f1\u8bed\u5355\u8bcd\u6216\u6570\u5b57\u56de\u7b54\u3002"),  # = official tcf_v2 (open, Chinese)
}
DATASETS = ["MMStar", "CCBench", "MMBench_DEV_CN_V11", "MMBench_DEV_EN_V11",
            "MME", "ViLP"]


def load_all_rows():
    rows = []
    for ds in DATASETS:
        df = pd.read_csv(LMU / f"{ds}.tsv", sep="\t")
        has_idx = "index" in df.columns
        # VLMEvalKit image dedup: a short image value is an index REFERENCE
        # to another row's base64 within the same TSV — resolve it.
        img_by_idx = {}
        if has_idx:
            for _, rr in df.iterrows():
                v = str(rr["image"])
                if len(v) >= 64:
                    img_by_idx[str(rr["index"])] = rr["image"]
        def resolve_image(val):
            s = str(val)
            return img_by_idx.get(s, val) if len(s) < 64 else val
        for pos in range(len(df)):
            r = df.iloc[pos]
            idx = str(r["index"]) if has_idx else str(pos)
            q = str(r["question"])
            hint = r.get("hint")
            if pd.notna(hint) and str(hint).strip().lower() not in ("", "nan"):
                q = f"{hint}\n{q}"
            labels = [L for L in "ABCD" if L in df.columns and pd.notna(r.get(L))
                      and str(r.get(L)).strip().lower() not in ("", "nan")]
            if ds == "ViLP":
                typ = "open"
            elif ds == "MME":
                typ = "binary"
            elif len(labels) >= 2:
                typ = "mcq"
            else:
                continue
            rows.append(dict(ds=ds, index=idx, typ=typ, q=q,
                             row={**{L: r[L] for L in labels}, "question": q,
                                  "image": resolve_image(r["image"])},
                             labels=labels, answer=str(r["answer"]).strip()))
    return rows


@torch.no_grad()
def generate(model, proc, image, prompt):
    if MODEL == "qwen3":
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
    out = model.generate(**inp, max_new_tokens=MAXNEW, do_sample=False)
    new = out[0][inp["input_ids"].shape[1]:]
    return proc.tokenizer.decode(new, skip_special_tokens=True).strip()


def variant_specs(r):
    """(name, template_kind, image_kind) — template_kind: mcq prompts use
    NAT builders; open/binary use GEN_PROMPTS keys."""
    if r["typ"] == "mcq":
        return [("orig", "mcq_default", "default"),
                ("vcf1", "mcq_default", "blank"),
                ("vcf2", "mcq_default", "noise500"),
                ("tcf1", "mcq_tcf1", "default"),
                ("tcf2", "mcq_tcf2", "default")]
    return [("orig", "default", "default"),
            ("vcf1", "default", "blank"),
            ("vcf2", "default", "noise500"),
            ("tcf1", "answer_format", "default"),
            ("tcf2", "paraphrase", "default")]


CN_DS = {"MMBench_DEV_CN_V11", "CCBench"}
def build_prompt(r, kind):
    if kind.startswith("mcq"):
        # v4: OFFICIAL vlmeval instructions (prompt_variation replace-list)
        cn = r["ds"] in CN_DS
        opts = "\n".join(f"{L}. {r['row'][L]}" for L in r["labels"])
        if kind == "mcq_default":
            ins = "请直接回答选项字母。" if cn else "Please select the correct answer from the options above."
        elif kind == "mcq_tcf1":
            ins = ("结合问题与选项仔细观察图像中的信息，请直接回答选项字母。" if cn else
                   "Think about the question based on details in the given image. Please select the correct answer from the options above.")
        else:  # mcq_tcf2
            ins = ("Please carefully examine the information in the image, then consider the question and options, and reply directly with the letter corresponding to the correct answer from the options above." if cn else
                   "请仔细观察图像中的信息，然后结合问题与选项，从上述所有选项中直接回答正确选项对应的字母。")
        return f"{r['q']}\n{opts}\n{ins}"
    return GEN_PROMPTS[kind].format(q=r["q"])


def run():
    proc, model = _load_model()
    tok = proc.tokenizer
    rows = load_all_rows()
    done = set()
    if CKPT.exists():
        with CKPT.open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["ds"], rec["index"]))
    print(f"{MODEL}: {len(rows)} rows, {len(done)} done", flush=True)
    yn_ids = {L: NAT.letter_token_ids(tok, L) for L in ("Yes", "No")}
    with CKPT.open("a") as out:
        for i, r in enumerate(rows):
            if (r["ds"], r["index"]) in done:
                continue
            img = NAT.load_image(r["row"])
            if img is None:
                continue
            rec = {"ds": r["ds"], "index": r["index"], "typ": r["typ"],
                   "answer": r["answer"], "labels": r["labels"]}
            for name, tkind, ikind in variant_specs(r):
                prompt = build_prompt(r, tkind)
                image = NAT.transform_image(img, ikind, 0)
                if r["typ"] == "open":
                    rec[name] = generate(model, proc, image, prompt)
                else:
                    lg = first_logits(model, proc, image, prompt)
                    if r["typ"] == "mcq":
                        ids = {L: NAT.letter_token_ids(tok, L) for L in r["labels"]}
                        lp = NAT.option_logprobs(lg, r["labels"], ids)
                        rec[name] = r["labels"][int(lp.argmax())]
                        rec[name + "_lp"] = [round(float(x), 4) for x in lp]
                    else:
                        lp = NAT.option_logprobs(lg, ["Yes", "No"], yn_ids)
                        rec[name] = ["Yes", "No"][int(lp.argmax())]
                        rec[name + "_lp"] = [round(float(x), 4) for x in lp]
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 100 == 0:
                print(f"progress {i+1}/{len(rows)}", flush=True)
    print("DONE b0", flush=True)


def report():
    from collections import Counter
    recs = []
    with CKPT.open() as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    print(f"{MODEL}: {len(recs)} rows", Counter(r["ds"] for r in recs))
    for typ in ("mcq", "binary", "open"):
        sub = [r for r in recs if r["typ"] == typ and "orig" in r]
        if not sub:
            continue
        if typ == "open":
            from open_ended_scoring import paper_hit
            acc = 100.0 * sum(paper_hit(r["answer"], r["orig"]) for r in sub) / len(sub)
        else:
            acc = 100.0 * sum(str(r["orig"]).lower() == str(r["answer"]).strip().lower()
                              or r["orig"] == r["answer"] for r in sub) / len(sub)
        print(f"  {typ}: n={len(sub)} orig acc={acc:.2f}")


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    elif "--report" in sys.argv:
        report()
    else:
        print(__doc__)
