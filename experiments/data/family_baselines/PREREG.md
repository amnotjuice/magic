# PREREG — answer-level adaptations of two gated/adaptive-contrast baselines

Registered 2026-07-22, before any readout. Authorized by the author on
2026-07-22 ("做，两个都做"), overriding the compare-only-to-published
rule for this one purpose: every PDF-only review round names "no member
of the nearest family is run" as a top gap. Both arms are OUR
adaptations to the answer-scoring regime, will be labelled as such, and
get the same validation-fitting budget MALT's thresholds got.

## Data

Frozen per-sample first-token logits in `/data2/shunshungu/dump_tensors/`
(step-0 era dumps, never re-generated): `{model}-Original` plus visual
corruptions `VCF-{Color0, Noise400, Noise500, Grayscale, StrongBlur,
CenterMask, HighPass}` on the `*_Biased_Val` and `*_Biased_Test`
subsets of all six datasets, both 2024 backbones. Zero GPU; zero new
forwards.

Scoring: first-token option log-probs (letter tokens for MCQ, Yes/No
for MME), which is provably full-generation-equivalent for MCQ and
binary under the deployed greedy decoding. ViLP open-ended is NOT
evaluable from these dumps and is excluded; the arms therefore report
MCQ and binary cells only, marked n/a on Others.

## Gate answers

**Candidate 1 — VACoDe-A (answer-level VACoDe).** For each row: compute
the option-restricted softmax under the original image and under each of
the seven corruptions; select the corruption c* with the largest L2
distance from the original softmax (VACoDe's adaptive-augmentation
selection, restated at the answer level); emit
`argmax[(1+alpha) * s_orig - alpha * s_{c*}]` (the VCD operator it
plugs into). One free constant: alpha.

**Candidate 2 — CHASD-A (answer-level confidence-gated VCD).** For each
row: if the predictive entropy of the original option softmax exceeds
tau, emit `argmax[(1+alpha) * s_orig - alpha * s_noise500]` (VCD with
its native corruption); else emit the original argmax. Two free
constants: alpha, tau.

TC = none (both arms are visual-contrast members, faithful to their
sources). VC = as stated. adaptivity = per-sample augmentation choice
(arm 1) / per-sample firing (arm 2). aggregation = the VCD operator.
active cost = 8 passes charged for arm 1 (1 + 7 probes; we will also
report the 2-pass variant that scores only the selected corruption
after a fixed screening set, if the selection turns out concentrated),
and 1 + f passes for arm 2.

**Detector** = none; continuous evidence only (softmax distance,
entropy).

**Calibration** = alpha in {0.5, 1.0, 1.5, 2.0} and tau on a 6-point
entropy grid, fitted on the `*_Biased_Val` dumps by BS accuracy, per
backbone, frozen, then ONE Test read per arm per backbone.

**Acceptance / kill** = none in the usual sense: these are baselines,
not our candidates. Whatever the Test read says is reported as-is in
Table 1's MCQ/binary columns. No retuning after the Test read, no
second draw. If an arm beats MALT on some cell, that cell is reported
bold for the arm like any other baseline.

**Frontier (gate 8)** = not applicable; no new mechanism of ours is
being admitted.

## Traps checked

- First-token trap: MCQ + binary only; ViLP excluded (rule honored).
- Row alignment: dump `logits_{i}.pt` indexes follow the result-xlsx row
  order of the step-0 run, which follows the TSV order; alignment will
  be verified by re-deriving the Original arm's B/S/BS cells and
  matching them against the frozen bundle's base cells before any
  baseline number is read.
- Environment: `/data/shunshungu/tmp/miniforge3/envs/myenv/bin/python`
  only.

## Status

- [x] Pre-registration.
- [ ] Alignment check (Original arm reproduces known base cells).
- [ ] Val fit.
- [ ] One Test read per arm per backbone.

## Amendment 2026-07-22 (before any readout)

The Noise400 and HighPass dumps do not cover `Biased_Test` on both
backbones. The corruption pool is amended to the five with complete
Val+Test coverage: Color0, Noise500, Grayscale, StrongBlur, CenterMask.
Arm-1 active cost becomes 6 passes (1 + 5 probes). Nothing else changes;
no scores had been computed at amendment time (extraction had produced
one Val cache file and failed on the missing Test dir).

## Amendment 2 — 2026-07-22, after the first Test read

A reviewer noted both fits landed on the grid edge (alpha=2.0, tau=0.2),
i.e. the searched grid may have handicapped the baselines. The grids
were extended in the baseline-favouring direction (alpha up to 32, tau
down to 0), re-fitted on Val only, and ONE further Test read taken.
Sequence fully disclosed: read 1 at the original grid gave VACoDe-A
20.86/24.59 and CHASD-A 25.69/26.62; read 2 at the extended grid gives
VACoDe-A 22.28/26.78 (alpha 12/8) and CHASD-A 30.53/30.47 (alpha 12/32,
tau=0, firing 100%). The paper reports the stronger read-2 numbers.

The extension is itself a finding: with free rein on the biased
validation split, both adaptations open their gate completely and run
alpha toward the TIE limit ((1+a)s_orig - a s_corr -> s_orig - s_corr
direction as a grows), i.e. the step-level adaptivity of the source
methods contributes nothing at the answer level, and unconstrained
biased-split fitting drives the family toward exactly the ungated
anti-prior contrast whose natural-distribution harm the paper's
controls document.
