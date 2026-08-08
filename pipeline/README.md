# Pipeline: raw images → `magic_mcq_bundle.json.gz` / `magic_oth_bundle.json.gz`

`experiments/` reproduces the paper's tables and figures from the two
frozen score bundles at the repo root. This directory is the other half:
the code that produced those bundles in the first place, starting from a
real Qwen2-VL / LLaVA-NeXT checkpoint and the benchmark images.

This is real, working code (adapted from the internal research pipeline,
same logic, cleaned up and renamed) — but running it end-to-end requires
a GPU, the model checkpoints, and the benchmark datasets in `~/LMUData`.
Nothing in this directory has been re-executed to reverify the shipped
bundles bit-for-bit; see "Known gaps" below for exactly what that means.

## Stages

```
step0_run_basemodel.sh          run the base model + VCF/TCF variants on the
                                 full benchmarks, dump per-sample logits
                                 (needs vlmeval/, a checkpoint, GPU)
        |
step1_generate_biased_subsets.sh  detect where the model is biased, write
                                   the *_Biased_{Val,Test}.tsv subsets
        |
step2_reinference_vcf.sh          re-run on the biased subset under the
step2_reinference_tcf.sh          VCF / TCF views (this is what populates
                                   dump_tensors/ with the per-view logits
                                   the bundle-builders below read)
        |
build_mcq_bundle.py              assemble MCQ rows (s0/default_delta/
build_oth_bundle.py              answer_delta/selected_vc for MCQ; real_def/
                                  real_ans/blank_def/blank_ans/vc_gray for
                                  open-ended) from dump_tensors + the caches
                                  in data/
        |
   magic.py (repo root)          the method itself, run on the assembled
                                  bundle
```

`step3`/`step4` from the original SCI-reproduction pipeline (hyperparameter
search, running SCI's own live-generation algorithm) are not included here
— MAGIC doesn't need them.

## Two ways to run MAGIC

The bundle-and-replay path above (`build_*_bundle.py` → `magic.py`) is how
the paper's numbers were produced: views collected once via `vlmeval`,
the ladder fusion iterated offline on the cached scores — useful because
tuning `theta1`/`t2`/`tau_V` doesn't cost a GPU pass each time. But that
split is a research-process convenience, not something the algorithm
requires. MAGIC's fusion is exactly as embeddable in `generate_inner()`
as SCI's is — it needs the same *kind* of multi-view forward passes SCI
already runs live, just a different formula and three more corruption
channels (grayscale / strong-blur / centre-mask, for the rung-2 routing
SCI doesn't have).

So `vlmeval/vlm/qwen2_vl/model.py` and `vlm/llava/llava.py` also carry a
live, single-call `visual_type='MAGIC'` path (config entries
`Qwen2-VL-7B-MAGIC` / `LLaVA-NeXT-8B-MAGIC`) mirroring the existing
`'SCI-Adaptive'` branch: one `generate_inner()` call internally runs the
7 views MCQ needs (real/blank × default/answer-format, plus the 3
routing channels), reduces each to per-option-letter scores the same way
`build_mcq_bundle.py` does offline, runs `magic.py`'s exact ladder
(`magic_run_ladder`, a direct port — cross-checked against `magic.py`'s
own `run_ladder`/`margin` over 2000 randomized trials, zero mismatches),
and returns the answer in one pass — no `dump_tensors`, no bundle, no
offline replay step.

```bash
python run.py --data MMStar --model Qwen2-VL-7B-MAGIC --work-dir ./outputs_magic_live
```

This live path is **MCQ only** — it needs the option letters materialized
in the prompt to do the letter-score reduction; open-ended (Oth) scoring
stays on the offline `build_oth_bundle.py` path, matching how the paper's
Oth numbers were actually produced. It also hasn't been run against a
real checkpoint in this environment (no GPU here) — the ladder math
itself is verified (see above), the multi-pass generation plumbing
follows the exact pattern `'SCI-Adaptive'` already uses live in this same
file, but the live path as a whole is unexercised, not assumed correct
end-to-end.

`vlmeval/` is a trimmed copy of the VLMEvalKit v0.2 fork step0-2 depend
on: just the two patched model wrappers (`vlm/qwen2_vl/`, `vlm/llava/` —
these carry the actual VCF/TCF branching and logit-dump hooks) plus the
core framework they need (`dataset/`, `smp/`, `utils/`, `config.py`
trimmed to the Qwen2-VL-7B/LLaVA-NeXT-8B model registry). The ~60 other
model wrappers and the commercial-API adapters from the original fork are
not included — this pipeline only ever runs these two backbones.

## Support modules

`build_mcq_bundle.py` and `build_oth_bundle.py` call into several smaller
modules that live alongside them: `llava_vilp_inference.py` /
`llava_mme_inference.py` / `qwen_mme_inference.py` (live model inference
for the open-ended candidate pools), `oth_candidate_preflight.py` (loads
the TC candidate logs the open-ended pool is built from),
`legacy_scoring_framework.py` / `legacy_cached_score.py` /
`open_ended_scoring.py` / `action_grammar.py` / `adaptive_aug_policy.py`
(older scoring/matching utilities these still depend on). `data/` holds
the small caches (~19M) these read — mostly candidate pools and
calibration logs, not raw model outputs.

## Requirements

```
pip install -r ../requirements.txt
pip install torch transformers qwen-vl-utils pillow torchvision
```

Edit the hardcoded model-checkpoint paths near the top of `vlmeval/config.py`
(`model_path=...`) and in the support modules (`MODEL_PATH`, `LLAVA_PATH`,
etc.) to point at your own checkpoints. Datasets are expected under
`~/LMUData` (standard VLMEvalKit convention).

## Known gaps

This is disclosed here rather than glossed over, because it's a real
limit on what "run this pipeline" gets you:

- **Some raw dumps no longer exist.** `build_mcq_bundle.py` computes
  `answer_delta` from raw `TCF-AnswerFormat` view dumps when they're
  present. For Qwen's Test split, those specific dumps were never
  written in the original data-collection run — the code's own fallback
  (documented in `attach_answer_delta`) borrows `answer_delta` for those
  rows from the frozen `magic_mcq_bundle.json.gz` at the repo root,
  matched by the row's `s0` fingerprint. Re-running `step2` for Qwen's
  `TCF-AnswerFormat`/`TCF-AnswerFormat-Blank` views would close this gap,
  but that hasn't been done here.
- **One cache fallback points outside this repo.** `adaptive_aug_policy.py`
  and `oth_candidate_preflight.py` fall back to an archived snapshot
  (`Self-Critical-Inference-Framework__archive_preupload_20260618T033451Z`)
  for a handful of files if they're not found under `pipeline/data/`. That
  archive is not part of this repository; if you don't have it, that
  specific fallback branch is simply unreachable, not a hidden failure.
- **Not re-verified end-to-end.** Because this environment has no GPU,
  none of `step0`-`step2` or the bundle-builders were actually re-run
  here. The code is the same logic that produced the shipped bundles, but
  "same logic" hasn't been checked against "same output" for this
  specific copy — that's real follow-up work, not assumed done.
