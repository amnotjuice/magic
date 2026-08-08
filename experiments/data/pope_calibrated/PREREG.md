# PREREG — protocol-compliant POPE readout (val/test re-calibration)

Registered 2026-07-28, before any readout. Motivation: the paper currently
reports POPE only for Qwen2-VL, under DRBench-frozen constants. The frozen
LLaVA arm fails (`research/log/cda_v8_pope/FINDINGS.md`: MALT −1.12 at
n=9000, p<1e-4), and so does SCI5 on the same rows (+0.22, p=0.36), i.e.
family-level non-transfer of frozen constants. Reporting one backbone and
staying silent on the other is the paper's largest open attack surface.

The paper's own contract (Limitations) says the three thresholds are fitted
per target distribution on a validation split. Frozen transfer is therefore
NOT the method's specified usage on a new distribution; it is a stress test.
This probe reads the SPECIFIED usage, symmetrically on both backbones.

## Gate answers

- **Candidate** = the deployed binary/Others ladder, unchanged in form.
  state0 = s0; lift = max(s0 − blank, ans_r − ans_b); residual =
  max(s0, ans_r) − gray; rung-2 commit rule unchanged. ONLY the three
  constants (t1, t2, tauV) are re-fitted.
- TC = answer-format rephrasing (cached). VC = gray (cached).
  adaptivity = margin gate. aggregation = elementwise max + centered add.
  active cost = 1 + 3*f1 + 3*f2 scoring passes, reported.
- **Detector** = none; margin only.
- **Calibration** = stratified 50/50 val/test split by (category, answer),
  seed 0, per backbone. Grid fitted on VAL ONLY by accuracy:
  t1 in {0, 0.25, 0.5, 1, 2, 3, 5, 8}; t2 in {0, 0.2, 0.3, 0.4, 0.6, 1.0};
  tauV in {0, 0.25, 0.5, 0.625, 1.0}. Ties broken toward the SMALLER t1
  (cheaper). The grid CONTAINS the no-op t1=0, so the fitted operator can
  fall back to the base model and no-harm is structurally available.
  ONE test read per backbone after the val fit is frozen.
- **Comparators on the same test rows**: base, SCI5 (paper params
  beta=.2 alpha=1 gamma=2.0 theta=.3, their formula verbatim on the cached
  raws), and MALT under the DRBench-frozen constants (the number the paper
  currently reports), so the calibrated and frozen readings are visible
  side by side.
- **Acceptance** = descriptive. No kill rule; whatever the test read says is
  reported. Pre-stated interpretation:
  - calibrated MALT > base on both backbones -> the paper can report POPE
    symmetrically under its own contract, and the frozen-transfer result
    becomes a disclosed stress test rather than a hidden failure;
  - calibrated MALT <= base on either backbone -> the transfer limitation
    is real under the method's own usage, and the paper says so in
    Limitations with the number.
- **Frontier** = n/a (no new mechanism; constants only).

## Traps checked

- First-token trap: POPE is binary yes/no, so first-token option scoring is
  provably equivalent to generation + prefix match. OK.
- Val-selection trap: val is 4500 rows (LLaVA) / ~1575 (Qwen), far above the
  ~65-row biased-Val regime that burned earlier probes.
- Test-never-selects: the grid is scored on val only; one test read.
- Environment: /data/shunshungu/tmp/miniforge3/envs/myenv/bin/python; zero GPU
  (replay of research/log/cda_v8_pope/pope_{llava,qwen2}.jsonl).

## Status

- [x] Pre-registration.
- [ ] Val fit (both backbones).
- [ ] ONE test read per backbone.
- [ ] Report to author; author decides whether the paper uses it.
