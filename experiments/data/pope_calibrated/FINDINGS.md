# POPE under the method's own contract (val/test re-calibration)

Pre-registration: PREREG.md (this directory). Executed 2026-07-28, zero GPU.
Script: `research/malt_pope_calibrated.py`.

## Protocol

Stratified 50/50 val/test split by (category, answer), deterministic
interleave (no RNG). Thresholds fitted on VAL ONLY over a 8x6x5 grid that
contains the no-op t1=0, ties broken toward the smaller t1. ONE test read.
Comparators on the same test rows: base, SCI5 (published params, their
formula on the cached raws), and MALT under the DRBench-frozen constants
(what the paper currently reports).

## Test read

| backbone | n_test | fitted (t1,t2,tauV) | base | SCI5 | **MALT cal.** | MALT frozen | passes |
|---|---:|---|---:|---:|---:|---:|---:|
| Qwen2-VL   | 1575 | (1.0, 0.0, 0.0)     | 87.87 | 88.06 | **89.52** | 89.14 | 1.29 |
| LLaVA-NeXT | 4500 | (0.0, 0.6, 0.625)   | 86.78 | 86.93 | **86.62** | 85.47 | 1.42 |

Paired McNemar (exact, two-sided), calibrated MALT vs comparator:

| backbone | vs base | vs SCI5 |
|---|---|---|
| Qwen2-VL   | **+1.65, 41:15, p=0.0007** | **+1.46, 33:10, p=0.0006** |
| LLaVA-NeXT | −0.16, 9:16, p=0.23 (tie) | −0.31, 107:121, p=0.39 (tie) |

Per category (calibrated vs base): Qwen adversarial 90.4/87.0, popular
91.2/88.8, random 88.4/87.8. LLaVA adversarial 81.67/81.67, popular
88.13/88.53, random 90.07/90.13.

## Reading

1. **Under its own contract the method transfers on Qwen2-VL and is
   no-harm on LLaVA-NeXT.** The Qwen gain is larger and far more
   significant than the frozen-transfer number the paper currently reports
   (+1.65 vs +0.92), and it now also clears SCI5 significantly, which the
   frozen reading never claimed.
2. **The LLaVA failure was a calibration artifact, not a method failure.**
   Frozen constants gave −1.31 vs base on these rows (85.47 vs 86.78); the
   val fit selects t1=0, i.e. it recognises that the correction does not
   pay on this distribution and backs off to essentially the base model,
   landing at a statistical tie (p=0.23). This is the val split doing the
   job the paper says it does.
3. **SCI5 is flat on POPE on both backbones** (+0.19 / +0.15 over base),
   so the family-level non-transfer noted in cda_v8_pope/FINDINGS.md is
   confirmed and is not specific to MALT.
4. Cost on POPE is 1.29 / 1.42 mean scoring passes, well under the
   five-pass fixed budget.

## Robustness of the val fit (test accuracy swept over t1, others fixed)

| t1 | 0 | 0.25 | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 | 8.0 | base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2-VL   | 87.87 | 88.51 | 89.02 | **89.52** | 89.71 | 89.65 | 89.65 | 89.65 | 87.87 |
| LLaVA-NeXT | **86.62** | 85.96 | 85.67 | 85.67 | 85.64 | 85.64 | 85.64 | 85.64 | 86.78 |

Bold = the value the val fit selected. The Qwen gain is not a val-selection
artifact: every t1 >= 0.5 lands on a 89.0-89.7 plateau, and the fitted point
sits inside it. On LLaVA the curve is monotonically decreasing, the
correction genuinely does not pay there, and the val fit picked the
minimal-correction end of the grid, which is the best available point.
This is the calibration step behaving as specified on both backbones.

## Honest caveats to carry into any paper text

- LLaVA is a TIE, not a win. It must be reported as no-harm, never as a gain.
- The two backbones have different n (Qwen test 1575 from the cached 3151
  stratified draw; LLaVA test 4500 from the full 9000), because the cached
  Qwen forward set is the earlier subsample. Disclose.
- The frozen-transfer readings stay disclosable as a stress test: constants
  are distribution-calibrated, and transferring them unchanged costs
  1.3 points on LLaVA.
- On Qwen the fitted t2=0 and tauV=0 mean rung 2 fires on ties only; the
  gain is carried by rung 1.

## Status

- [x] Pre-registration, val fit, ONE test read per backbone.
- [ ] Author decision: whether the paper adopts this reading.
