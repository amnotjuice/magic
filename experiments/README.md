# Reproducing every paper table and figure

The reference implementation of the method itself is
[`../magic.py`](../magic.py) (stdlib-only, reproduces the main results
bit-exactly). Everything in this directory reproduces one of the
*secondary* tables or figures — sensitivity, ablations, cost accounting,
the newer-backbone comparison, and the paper's plots.

Run every script from the repository root, e.g.:

```bash
python3 experiments/cells_ci.py
```

Scripts import the reference implementation as `import magic as M` and
read their frozen data from `experiments/data/`.

## Dependency tiers

- **Tier 1 — zero extra dependencies.** Only needs `numpy`/`pandas`
  (`requirements.txt`). No GPU, no torch, no model checkpoints. All data
  is already in `experiments/data/`.
- **Tier 2 — needs the full research environment.** Needs `torch` +
  `transformers` (CPU is enough for most; a couple call a tokenizer or a
  full model constructor) and, for some, local model checkpoint paths
  hardcoded near the top of the file (edit `MODEL_PATH`/`LLAVA_PATH`/etc.
  to point at your own checkpoints). These are not required to verify the
  paper's headline numbers — only for the newer-backbone /
  generation-adjacent tables.

## Paper tables

| script | tier | backs |
|---|---|---|
| `cells_ci.py` | 1 | main table CIs + paired bootstrap tests |
| `twosample.py` | 1 | two-sample companion to the main table |
| `sci_paired.py` | 1 | paired McNemar, MAGIC vs SCI5 (LLaVA MCQ-BS) |
| `ablation_passes.py` | 1 | ablation table's cost column (avg. passes per arm) |
| `rung1_components.py` | 1 | rung-1 component ablation |
| `rung2_source.py` | 2 | why rung 2 routes over gray/blur/center-mask |
| `family_baselines.py` | 2 | gated/adaptive contrastive-decoding baselines |
| `native_cd.py` | 1 | newer-backbone native VCD/M3ID baselines |
| `native_cd_supp.py` | 2 | native baselines, supplementary rows |
| `native_m3id_textonly.py` | 2 | native M3ID, text-only variant |
| `sci7_open_supp.py` | 2 | native SCI7, open-ended supplement |
| `newgen_replay.py` | 2 | newer-backbone MAGIC rows, frozen replay |
| `heldout_cd.py` | 1 | held-out TIE/VCD on Qwen3-VL / LLaVA-OneVision-1.5 |
| `pope_calibrated.py` | 1 | POPE under the method's own contract (calibrated re-run) |
| `oldgen_screen.py` | 2 | calibration screen on the older-backbone pair |
| `permutation_control.py` | 2 | option-permutation control + batched wall-clock |
| `cost_replay.py` | 1 | verification replay of the newer-backbone cost numbers |
| `paper_stats.py` | 1 (~2.5 min: 100k-iteration bootstrap, pure Python) | released-form reconciliation statistics |

## Paper figures

| script | tier | backs |
|---|---|---|
| `paper_figures.py` | 1 | main publication figure set (teaser scatter, etc.) |
| `marginlaw4.py` | 1 | margin-law figure, four backbones |
| `biasvar.py` | 2 | bias-variance keystone fit |
| `biasvar_ci.py` | 2 | bias-variance CI computation |
| `biasvar_fig.py` | 1 | bias-variance figure render |
| `sens_fig.py` | 1 | BS-accuracy sensitivity to the rung-1 threshold |
| `inert_certificate.py` | 1 | certificate rate (gate-skip fraction) |

## Support modules

Not run directly — imported by the scripts above. `figstyle.py` and
`oth_grid_replay.py` are Tier 1. The rest implement the newer-backbone
(Qwen3-VL / LLaVA-OneVision-1.5) generation and evaluation pipeline and
are Tier 2: `natural_mcq_eval_qwen.py`, `newbackbone_forward_scores.py`,
`newbackbone_generate.py`, `newbackbone_eval_prep.py`,
`newbackbone_eval_read.py`, `newbackbone_calibration_eval.py`,
`newbackbone_symmetric_eval.py`, `open_ended_scoring.py`,
`numword_scoring.py`.

## Data

`experiments/data/` holds the ~43M of frozen intermediate results these
scripts read (bootstrap ledgers, calibration caches, cost profiles, the
newer-backbone generation cache, etc.) — precise enough to reproduce
every number above, small enough to version normally. Every file here is
read by at least one script in this directory (verified per-file, not by
directory-level heuristic); execution logs and superseded intermediate
files from the original runs were dropped. It does not include raw model
outputs or images; those live in the (unpublished) generation pipeline
described in the top-level README's Scope section.
