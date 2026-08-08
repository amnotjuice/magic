"""M3ID's own contrast branch: the UNCONDITIONED (text-only) model.

Pre-registration: experiments/data/native_cd_newgen/PREREG.md (amendment
2026-07-27, before any read: M3ID is defined against the image-free
conditional p(y|x), not against a corrupted image; the released DRBench
implementation substitutes a noised image, so we dump the paper's own
branch and give M3ID the stronger of the two in Table 4).

Dumps first-token scores with NO image in the context, over the same
candidate sets the frozen population uses (option letters / Yes-No /
materialised pool).

Usage:
  python3 experiments/native_m3id_textonly.py qwen3
  python3 experiments/native_m3id_textonly.py ov15
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import natural_mcq_eval_qwen as NAT  # noqa: E402
import newbackbone_forward_scores as N1  # noqa: E402
from newbackbone_forward_scores import _load_model  # noqa: E402
from newbackbone_generate import load_all_rows  # noqa: E402
import newbackbone_eval_prep as PREP  # noqa: E402

MODEL = "ov15" if "ov15" in sys.argv else "qwen3"
N1.MODEL = MODEL
PREP.MODEL = MODEL
OUT = ROOT / "experiments/data/native_cd_newgen"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / f"textonly_{MODEL}.jsonl"


@torch.no_grad()
def first_logits_textonly(model, proc, prompt):
    """First-token logits with the image dropped from the context."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if MODEL in ("qwen25", "qwen3"):
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=None, videos=None, return_tensors="pt").to("cuda")
    else:
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=text, return_tensors="pt").to("cuda")
    return model(**inp).logits[0, -1, :].float().cpu()


def main():
    import newbackbone_calibration_eval as EV
    EV.MODEL = MODEL
    rows_by_key = {(r["ds"], r["index"]): r for r in load_all_rows()}
    test = [r for r in EV.load("biased") if r.get("split") == "test"]
    done = set()
    if CKPT.exists():
        with CKPT.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["ds"], r["index"]))
    print(f"{MODEL}: {len(test)} Test rows, {len(done)} done", flush=True)

    proc, model = _load_model()
    tok = proc.tokenizer
    yn_ids = {L: NAT.letter_token_ids(tok, L) for L in ("Yes", "No")}

    with CKPT.open("a") as out:
        for i, r in enumerate(test):
            key = (r["ds"], r["index"])
            if key in done:
                continue
            src = rows_by_key.get(key)
            if src is None:
                continue
            work = {**src, "typ": r["typ"]}
            prompt = PREP.build_prompt(work, "default")
            lg = first_logits_textonly(model, proc, prompt)
            rec = {"ds": r["ds"], "index": r["index"], "typ": r["typ"]}
            if r["typ"] == "open":
                pool = r.get("pool") or []
                if not pool:
                    continue
                lp = torch.log_softmax(lg.float(), -1)
                rec["textonly"] = {
                    c: round(float(lp[tok(c, add_special_tokens=False).input_ids[0]].item()), 4)
                    for c in pool}
            else:
                labels = r.get("labels") or ["Yes", "No"]
                ids = ({L: NAT.letter_token_ids(tok, L) for L in labels}
                       if r["typ"] == "mcq" else yn_ids)
                rec["textonly"] = [round(float(x), 4)
                                   for x in NAT.option_logprobs(lg, labels, ids)]
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 200 == 0:
                print(f"  progress {i+1}/{len(test)}", flush=True)
    print(f"DONE {MODEL}", flush=True)


if __name__ == "__main__":
    main()
