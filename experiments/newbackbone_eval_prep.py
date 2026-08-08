"""Phase-B B3-prep: evidence pass on the B1 subsets + natural surfaces.
Pre-registration: experiments/data/newbackbone_generation/FINDINGS.md (B3 design decisions).

Surfaces (per model):
  biased    : ALL B1 rows (Val+Test).  Views = 7 ladder views
              (default/answer_format × real/blank + gray/strongblur/
              centermask real) + RAW option logits for SCI on
              {orig, vcf2=noise500, tcf1, tcf2} (blank raw comes free
              from the default/blank view).  MCQ + binary only; ViLP
              open rows get candidate-pool scoring views instead.
  natural_val : all_data_val rows (i%5==0 over each dataset) × 7 views.
  natural_test: stratified sample of all_data_test (3000 rows, seed 0,
              proportional per dataset) × 7 views.

Usage: --model qwen3|ov15 --run | --report
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import natural_mcq_eval_qwen as NAT  # noqa: E402
import newbackbone_forward_scores as N1  # noqa: E402
from newbackbone_forward_scores import _load_model, first_logits  # noqa: E402
from newbackbone_generate import load_all_rows, GEN_PROMPTS, generate  # noqa: E402

OUT = ROOT / "experiments/data/newbackbone_generation"
MODEL = "ov15" if "ov15" in sys.argv else ("onevision" if "onevision" in sys.argv else "qwen3")
N1.MODEL = MODEL
CKPT = OUT / f"b3prep_{MODEL}.jsonl"
NAT_TEST_N = 3000


def raw_option_logits(logits, labels, label_ids):
    vals = []
    for L in labels:
        vals.append(max(float(logits[t].item()) for t in label_ids[L]))
    return vals


def build_surfaces():
    """Return list of work rows with surface tags."""
    all_rows = {(r["ds"], r["index"]): r for r in load_all_rows()}
    b1 = json.loads((OUT / f"b1_subsets_{MODEL}.json").read_text())
    biased_keys = {(x["ds"], x["index"]): x for x in b1}
    # natural = complement split assignment (i%5) over enumeration order
    work = []
    per_ds_order = defaultdict(list)
    for r in load_all_rows():
        per_ds_order[r["ds"]].append(r)
    nat_test_pool = []
    for ds, rows in per_ds_order.items():
        for i, r in enumerate(rows):
            key = (r["ds"], r["index"])
            if key in biased_keys:
                work.append({**r, "surface": "biased",
                             "split": biased_keys[key]["split"],
                             "mode": biased_keys[key]["mode"]})
            if i % 5 == 0:
                work.append({**r, "surface": "natural_val"})
            else:
                nat_test_pool.append(r)
    rng = np.random.RandomState(0)
    # stratified proportional sample of natural_test
    by_ds = defaultdict(list)
    for r in nat_test_pool:
        by_ds[r["ds"]].append(r)
    tot = sum(len(v) for v in by_ds.values())
    for ds, rows in by_ds.items():
        k = max(1, round(NAT_TEST_N * len(rows) / tot))
        idx = rng.choice(len(rows), size=min(k, len(rows)), replace=False)
        for i in idx:
            work.append({**rows[i], "surface": "natural_test"})
    return work


LADDER_VIEWS = [  # (name, prompt_kind, image_kind)
    ("s0", "default", "default"),
    ("blank", "default", "blank"),
    ("ans_real", "answer", "default"),
    ("ans_blank", "answer", "blank"),
    ("gray", "default", "gray"),
    ("strongblur", "default", "strongblur"),
    ("centermask", "default", "centermask"),
]
SCI_EXTRA = [("noise500", "default", "noise500"),
             ("tcf1", "tcf1", "default"),
             ("tcf2", "tcf2", "default")]


from newbackbone_generate import build_prompt as B0P  # official prompts (v5)
CN_DS = {"MMBench_DEV_CN_V11", "CCBench"}
def build_prompt(r, kind):
    # v5 protocol-aligned: default/tcf views = OFFICIAL (via b0 builder);
    # the ladder's own second framing (answer view) stays the method's.
    rr = dict(r); rr.setdefault("ds", r.get("ds"))
    if r["typ"] == "mcq":
        if kind == "default":
            return B0P(rr, "mcq_default")
        if kind == "answer":
            return NAT.prompt_answer_format(r["row"], r["labels"])
        if kind == "tcf1":
            return B0P(rr, "mcq_tcf1")
        if kind == "tcf2":
            return B0P(rr, "mcq_tcf2")
    q = r["q"]
    if kind == "default":
        return GEN_PROMPTS["default"].format(q=q)
    if kind == "answer":
        return "Look at the image and answer the question. Use only the final answer, no explanation.\nQuestion: " + q + "\nAnswer:"
    if kind == "tcf1":
        return GEN_PROMPTS["answer_format"].format(q=q)
    if kind == "tcf2":
        return GEN_PROMPTS["paraphrase"].format(q=q)
    raise ValueError(kind)


def get_pool(r, b0_by_key):
    """ViLP candidate pool = dedup of B0's three real-image generations."""
    rec = b0_by_key.get((r["ds"], r["index"]))
    if not rec:
        return None
    pool = []
    for k in ("orig", "tcf1", "tcf2"):
        c = str(rec.get(k, "")).strip()
        if c and c not in pool:
            pool.append(c)
    return pool or None


def run():
    proc, model = _load_model()
    tok = proc.tokenizer
    work = build_surfaces()
    b0_by_key = {}
    with (OUT / f"b0_{MODEL}.jsonl").open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["typ"] == "open":
                    b0_by_key[(rec["ds"], rec["index"])] = rec
    done = set()
    if CKPT.exists():
        with CKPT.open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["surface"], rec["ds"], rec["index"]))
    print(f"{MODEL}: {len(work)} surface-rows, {len(done)} done", flush=True)
    yn_ids = {L: NAT.letter_token_ids(tok, L) for L in ("Yes", "No")}
    with CKPT.open("a") as out:
        for i, r in enumerate(work):
            key = (r["surface"], r["ds"], r["index"])
            if key in done:
                continue
            img = NAT.load_image(r["row"])
            if img is None:
                continue
            rec = {"surface": r["surface"], "ds": r["ds"], "index": r["index"],
                   "typ": r["typ"], "answer": r["answer"],
                   "split": r.get("split"), "mode": r.get("mode")}
            if r["typ"] == "open":
                pool = get_pool(r, b0_by_key)
                if not pool:
                    continue
                rec["pool"] = pool
                # candidate scoring: token-score each candidate under the
                # 7 ladder views is expensive; use first-token scoring of
                # each candidate's first token under each view prompt
                # (deployed L2-style pool scoring).
                cand_ids = {c: NAT.letter_token_ids(tok, c.split()[0]) if False else
                            [tok(c, add_special_tokens=False).input_ids[0]]
                            for c in pool}
                for name, pk, ik in LADDER_VIEWS:
                    lg = first_logits(model, proc, NAT.transform_image(img, ik, 0),
                                      build_prompt(r, pk))
                    lp = torch.log_softmax(lg.float(), -1)
                    rec[name] = {c: float(lp[ids[0]].item()) for c, ids in cand_ids.items()}
            else:
                labels = r["labels"] if r["typ"] == "mcq" else ["Yes", "No"]
                ids = ({L: NAT.letter_token_ids(tok, L) for L in labels}
                       if r["typ"] == "mcq" else yn_ids)
                views = LADDER_VIEWS + (SCI_EXTRA if r["surface"] == "biased" else [])
                for name, pk, ik in views:
                    lg = first_logits(model, proc, NAT.transform_image(img, ik, 0),
                                      build_prompt(r, pk))
                    lp = NAT.option_logprobs(lg, labels, ids)
                    rec[name] = [round(float(x), 4) for x in lp]
                    if r["surface"] == "biased" and name in (
                            "s0", "blank", "noise500", "tcf1", "tcf2"):
                        rec[name + "_raw"] = [round(v, 4) for v in
                                              raw_option_logits(lg, labels, ids)]
                rec["labels"] = labels
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 100 == 0:
                print(f"progress {i+1}/{len(work)}", flush=True)
    print("DONE b3prep", flush=True)


def report():
    from collections import Counter
    c = Counter()
    with CKPT.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                c[(rec["surface"], rec["typ"])] += 1
    for k, v in sorted(c.items()):
        print(k, v)


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    elif "--report" in sys.argv:
        report()
    else:
        print(__doc__)
