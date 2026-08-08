"""GPU top-up: the three SCI branch views the OPEN (ViLP) biased-Test rows lack.

Why this exists
---------------
Table 4's SCI rows on the two newer backbones are currently incomplete or
non-native on the open-ended rows:

  * SCI7  -- `newbackbone_eval_read.sci_n` returns None for `typ == "open"`, so
    SCI7 has no number on 199/1759 (qwen3) and 709/2728 (ov15) rows.  The
    cause is our dump, not the method: SCI's own `prompt_variation3`
    (vlmeval/vlm/qwen2_vl/model.py:483) DOES carry a rule for the open
    instruction "Answer the question directly using a single word or
    phrase.", and `v6supp` simply never scored the candidate pool under it.
  * SCI3/SCI5 -- `newbackbone_calibration_eval.sci5` substitutes MAGIC's own
    answer-format view (`ans_real`) for SCI's official tcf_v1 on open rows.
    The official open tcf_v1 is GEN_PROMPTS["answer_format"] (b3prep), a
    different prompt.

This script scores the candidate pool under the three missing views, using
b3prep's own code path (same prompt builder, same image transforms, same
first-token pool scoring), so the new views are bit-consistent with the
frozen cache:

  tcf1      official SCI tcf_v1 for the open format, real image
  tcf3      official SCI tcf_v3 (prompt_variation3 of the default), real image
  noise400  default prompt, add_diffusion_noise(img, 400, 0)

`s0` is re-scored as a determinism gate against the frozen cache -- the same
gate G2 the 07-27 native-CD top-up passed at max |diff| = 0.0000.

Usage (qwen3 -> nbenv, ov15 -> ov15env):
  python3 experiments/sci7_open_supp.py qwen3
  python3 experiments/sci7_open_supp.py ov15
Optional: --limit N   score only the first N open rows (dry run)
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
OUT = ROOT / "experiments/data/sci7_open"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / f"sci7open_{MODEL}.jsonl"

LIMIT = None
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])

# SCI's own tcf_v3 rule for the open-ended instruction
# (vlmeval/vlm/qwen2_vl/model.py:483-484, verbatim).
OPEN_INSTR = "Answer the question directly using a single word or phrase."
OPEN_TCF3 = ("You are a smart student who is good at answering questions. "
             + OPEN_INSTR)


def tcf3_prompt(default_prompt: str) -> str:
    """Apply SCI's prompt_variation3 to an open-format default prompt.

    Raises rather than passing an unrephrased prompt through, matching the
    reference implementation's `else: raise ValueError`.
    """
    if OPEN_INSTR not in default_prompt:
        raise ValueError(f"no tcf_v3 rule for prompt: {default_prompt!r}")
    return default_prompt.replace(OPEN_INSTR, OPEN_TCF3)


def main() -> None:
    import newbackbone_calibration_eval as EV
    EV.MODEL = MODEL

    rows_by_key = {(r["ds"], r["index"]): r for r in load_all_rows()}
    test = [r for r in EV.load("biased") if r.get("split") == "test"]
    open_rows = [r for r in test if r["typ"] == "open"]
    if LIMIT:
        open_rows = open_rows[:LIMIT]
    print(f"{MODEL}: {len(test)} biased Test rows, {len(open_rows)} open",
          flush=True)

    done = set()
    if CKPT.exists():
        with CKPT.open() as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done.add((d["ds"], d["index"]))
    print(f"  resuming, {len(done)} done", flush=True)

    proc, model = _load_model()
    tok = proc.tokenizer

    n = 0
    max_s0_diff = 0.0
    with CKPT.open("a") as out:
        for i, r in enumerate(open_rows):
            key = (r["ds"], r["index"])
            if key in done:
                continue
            src = rows_by_key.get(key)
            if src is None:
                print(f"  SKIP {key}: no source row", flush=True)
                continue
            img = NAT.load_image(src["row"])
            if img is None:
                print(f"  SKIP {key}: no image", flush=True)
                continue
            pool = r.get("pool") or []
            if not pool:
                print(f"  SKIP {key}: empty pool", flush=True)
                continue

            work = {**src, "typ": r["typ"], "ds": r["ds"], "index": r["index"]}
            p_def = PREP.build_prompt(work, "default")
            p_tcf1 = PREP.build_prompt(work, "tcf1")     # official open tcf_v1
            p_tcf3 = tcf3_prompt(p_def)                  # official open tcf_v3

            cand_ids = {c: tok(c, add_special_tokens=False).input_ids[0]
                        for c in pool}
            rec = {"ds": r["ds"], "index": r["index"], "typ": r["typ"],
                   "n_pool": len(pool)}
            views = (("s0", img, p_def),                 # determinism gate
                     ("tcf1", img, p_tcf1),
                     ("tcf3", img, p_tcf3),
                     ("noise400", NAT.add_diffusion_noise(img, 400, 0), p_def))
            for name, image, prompt in views:
                lg = first_logits(model, proc, image, prompt)
                lp = torch.log_softmax(lg.float(), -1)
                rec[name] = {c: round(float(lp[t].item()), 4)
                             for c, t in cand_ids.items()}
                rec[name + "_vocabmax"] = round(float(lp.max().item()), 4)

            # gate: re-scored s0 must match the frozen cache
            frozen = r.get("s0") or {}
            diffs = [abs(rec["s0"][c] - frozen[c]) for c in pool if c in frozen]
            if diffs:
                rec["s0_maxdiff"] = round(max(diffs), 4)
                max_s0_diff = max(max_s0_diff, max(diffs))

            out.write(json.dumps(rec) + "\n")
            out.flush()
            n += 1
            if n % 50 == 0:
                print(f"  progress {n}/{len(open_rows)} "
                      f"(s0 max|diff| so far {max_s0_diff:.4f})", flush=True)

    print(f"DONE {MODEL}: scored {n} open rows, "
          f"s0 determinism max|diff| = {max_s0_diff:.4f}", flush=True)


if __name__ == "__main__":
    main()
