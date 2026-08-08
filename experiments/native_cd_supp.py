"""GPU top-up for NATIVE (SCI-codebase) VCD / M3ID on the new backbones.

Pre-registration: experiments/data/native_cd_newgen/PREREG.md

The SCI codebase applies the VCD / M3ID contrast ONLY at the prefill step
(modeling_qwen2_vl.py: `logits.shape[1] > 1`; modeling_llava_next.py:
`decode_flag == 1`), i.e. at the FIRST generated token, with the
counterfactual branch computed once.  Both methods use vcf_noise500 as
that branch, with the plausibility mask
    cutoff = log(theta) + org.max();  fused[org < cutoff] = -inf
and
    VCD  : fused = (1+alpha)*org - alpha*cf,          alpha=1.0,  theta=0.3
    M3ID : fused = org + (1-a_t)/a_t * (org - cf),    theta=0.3,
           a_t   = exp(-0.02 * (n_prompt_tokens - n_image_tokens))

For MCQ / binary rows the answer IS the first token, so the native rule
is exactly a fusion of the cached first-token option scores -- no forward
is needed and the cached s0 / noise500 views are reused verbatim.  This
script supplies the two things the frozen cache lacks:

  1. n_text = n_prompt_tokens - n_image_tokens   (M3ID's per-row weight),
     for EVERY biased Test row;
  2. the `noise500` pool view for the OPEN (ViLP) rows, which b3prep
     scored under LADDER_VIEWS only -- computed here with b3prep's own
     code path (same prompt, same transform, same first-token scoring),
     plus a recomputed `s0` pool view as a determinism check against the
     frozen cache.

Usage (qwen3 -> nbenv, ov15 -> ov15env):
  python3 experiments/native_cd_supp.py qwen3
  python3 experiments/native_cd_supp.py ov15
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import natural_mcq_eval_qwen as NAT  # noqa: E402
import newbackbone_forward_scores as N1  # noqa: E402
from newbackbone_forward_scores import _load_model, first_logits  # noqa: E402
from newbackbone_generate import load_all_rows  # noqa: E402
import newbackbone_eval_prep as PREP  # noqa: E402

MODEL = "ov15" if "ov15" in sys.argv else "qwen3"
N1.MODEL = MODEL
PREP.MODEL = MODEL
OUT = ROOT / "experiments/data/native_cd_newgen"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / f"supp_{MODEL}.jsonl"


def _image_token_id(model, proc):
    """The placeholder id whose count equals n_image_features."""
    cfg = model.config
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(cfg, attr, None)
        if v is None:
            v = getattr(getattr(cfg, "text_config", object()), attr, None)
        if isinstance(v, int):
            return v
    for tokname in ("<|image_pad|>", "<image>", "<|image|>"):
        tid = proc.tokenizer.convert_tokens_to_ids(tokname)
        if tid is not None and tid >= 0:
            return tid
    raise RuntimeError("cannot resolve image token id")


@torch.no_grad()
def prompt_token_counts(model, proc, image, prompt, img_tok):
    """(n_prompt_tokens, n_image_tokens) for the deployed input build.

    Mirrors first_logits' input construction exactly, so the difference
    equals the `logits.shape[1] - n_image_features` the SCI codebase uses.
    """
    if MODEL in ("qwen25", "qwen3"):
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt")
    else:
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = proc(images=image, text=text, return_tensors="pt")
    ids = inp["input_ids"][0]
    return int(ids.shape[0]), int((ids == img_tok).sum().item())


def main():
    import newbackbone_calibration_eval as EV
    EV.MODEL = MODEL
    rows_by_key = {(r["ds"], r["index"]): r for r in load_all_rows()}
    test = [r for r in EV.load("biased") if r.get("split") == "test"]
    print(f"{MODEL}: {len(test)} biased Test rows", flush=True)

    done = set()
    if CKPT.exists():
        with CKPT.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["ds"], r["index"]))
    print(f"  resuming, {len(done)} done", flush=True)

    proc, model = _load_model()
    tok = proc.tokenizer
    img_tok = _image_token_id(model, proc)
    print(f"  image token id = {img_tok}", flush=True)

    n_open = 0
    with CKPT.open("a") as out:
        for i, r in enumerate(test):
            key = (r["ds"], r["index"])
            if key in done:
                continue
            src = rows_by_key.get(key)
            if src is None:
                continue
            img = NAT.load_image(src["row"])
            if img is None:
                continue
            work = {**src, "typ": r["typ"], "ds": r["ds"], "index": r["index"]}
            prompt = PREP.build_prompt(work, "default")
            n_tot, n_img = prompt_token_counts(model, proc, img, prompt, img_tok)
            rec = {"ds": r["ds"], "index": r["index"], "typ": r["typ"],
                   "n_prompt": n_tot, "n_image": n_img, "n_text": n_tot - n_img}
            if r["typ"] == "open":
                pool = r.get("pool") or []
                if pool:
                    cand_ids = {c: tok(c, add_special_tokens=False).input_ids[0]
                                for c in pool}
                    for name, ik in (("s0", "default"), ("noise500", "noise500")):
                        lg = first_logits(model, proc,
                                          NAT.transform_image(img, ik, 0), prompt)
                        lp = torch.log_softmax(lg.float(), -1)
                        rec[name] = {c: round(float(lp[t].item()), 4)
                                     for c, t in cand_ids.items()}
                        rec[name + "_vocabmax"] = round(float(lp.max().item()), 4)
                    n_open += 1
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 100 == 0:
                print(f"  progress {i+1}/{len(test)} (open done {n_open})", flush=True)
    print(f"DONE {MODEL}: open rows scored {n_open}", flush=True)


if __name__ == "__main__":
    main()
