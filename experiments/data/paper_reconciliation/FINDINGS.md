# MALT paper reconciliation — writing-phase statistics (2026-07-21)

Candidate = the released `malt.py` ladder exactly (allocation t1,t2 +
rung-2 commitment tau_V; NO rung-1 admission). Pure replay of frozen
artifacts; zero GPU; no parameter selected here. Driver:
`research/malt_paper_stats.py` → `ledger.json`.

Purpose: the 07-19 paper prose was written on the "-both gates" (pruned)
accounting; the 07-21 audited tables (paper/tables/) are the released
noadm form. Every prose-dependent statistic recomputed for the released
form so the paper matches the shipped code bit-for-bit.

## Parity guards (all pass)

- 9-cell biased Test == paper/tables/tab_main.tex (both backbones).
- Natural MCQ delta/passes == malt.py FROZEN (0.762/1.542, 1.5616/2.434).
- Natural Oth == tab_natural (+0.84 Qwen @(0,0.5); +0.04 LLaVA @(0,0)).

## Key outputs (BS Overall unless noted)

| stat | Qwen2-VL | LLaVA-NeXT |
|---|---|---|
| MALT BS Overall | 33.13 | 36.22 |
| vs SCI5 pub (29.50/34.19) | +3.63 | +2.03 |
| vs SCI7 pub (31.72/34.92) | +1.41 | +1.30 |
| row-bootstrap 95% CI (1e5, seed 0) | [30.93, 35.33] | [34.57, 37.88] |
| P(≤SCI5) one-sided | 4.3e-4 | 0.0074 |
| P(≤SCI7) one-sided | 0.108 | 0.061 |
| biased passes MCQ/Oth | 5.00 / 4.53 | 4.86 / 4.20 |
| wall-clock (A100, SCI at t_score) | 4.57 s (SCI5 4.58, SCI7 6.41) | 0.69 s (SCI5 0.65, SCI7 0.91) |
| vs SCI7 wall-clock | 0.71× | 0.76× |
| natural MCQ delta (n=10,630) | +0.76 (300:219, p=4.3e-4) | +1.56 (662:496, p=1.2e-6) |
| natural MCQ exit@1 | 9104/10630 (86%) | 6362/10630 (60%) |
| natural Oth delta | +0.84 (59:37, p=0.032) | +0.04 (1:0, tie) |
| POPE transfer | +0.92 (89.50 vs 88.58; 102:73, p=0.034) | −1.18 (85.80 vs 86.98; 96:202, p<1e-4) |

Rung arms (BS All, base / R1-only / R2-only / full):
Qwen 13.09 / 31.32 / 16.70 / 33.13 · LLaVA 21.58 / 35.95 / 22.28 / 36.22.

## Notes

- SCI7 comparisons are now DIRECTION-ONLY on both backbones (P=0.108 /
  0.061) — the pruned-form significance (0.037/0.040) does not carry
  over. Prose must say descriptive lead, not resolved.
- Wall-clock vs SCI5: parity on Qwen (4.57 vs 4.58), +7% on LLaVA
  (0.69 vs 0.65). Efficiency claim remains vs SCI7 only.
- POPE Qwen cache holds 3151 unique qids (the stratified 3000 plus a
  151-row overfill from the checkpointing run); readout is on the full
  cached row set, matching the 07-19 prose base (88.58). LLaVA is the
  full n=9000 benchmark.
- Open-ended wall-clock accounting: base pass is one generation whose
  prefill doubles as the first scoring pass; all other passes priced at
  t_score. SCI priced entirely at t_score (unfavourable to us).
- LLaVA natural Oth (0,0): ladder fires on margin-0 ties only (1 flip);
  the +0.04 is a tie-band readout, not a claimed gain.
- Biased-frozen thresholds replayed on the bundle's natural rows
  (NAT-pathway instrument B): Qwen +0.52 (494:439), LLaVA +1.19
  (761:635) — POSITIVE. The −0.29/−0.77 harm readout at the same
  thresholds is instrument-A (dump-pipeline) only; the A/B fork spans
  the biased-frozen configuration too. Appendix flip-ledger passages
  attribute per-instrument accordingly.
- Ungated (allocation-deleted, commitment kept) arm: biased BS
  33.41/36.44 at 7 passes; natural MCQ +0.26 n.s. (497:469) / +1.01
  (761:654) — gated dominates natural on both accuracy and cost.

## Applied to the paper (2026-07-21, same session)

All .tex prose reconciled to the released form: method.tex (commitment
kept, ledger 3+1, alg/pipeline commit test), experiments.tex (audited
tables \input'd; contract section rewritten to the transfer verdict with
the mandated two-readout + fork disclosures; ablation/gated-TIE/POPE/
seconds updated), abstract/intro/conclusion/limitations/
implementation_details updated. Compile: 0 errors, 0 undefined refs.

## Reviewer-driven experiments (2026-07-21, zero GPU)

`research/malt_reviewer_experiments.py` -> `log/malt_reviewer_experiments/results.json`

**E1 gate-signal ablation** (answers "no confidence-gated baseline was run").
Corrector held fixed, gate statistic swapped at a matched firing rate:

| gate | Qwen2 BS | LLaVA BS |
|---|---|---|
| margin (deployed) | 32.30 | 31.13 |
| max softmax prob (CHASD/CCD-style) | 32.38 | 31.05 |
| predictive entropy | 32.22 | 31.05 |

Spread <= 0.16 BS at identical compute. **The allocation axis is
signal-agnostic.** Consequence for the paper: the delta from the
confidence-gated family is NOT the gate statistic; it is the ladder,
which the gate-only baseline prices at +3.01/+1.13 MCQ-BS. Written into
the paper as a measurement replacing the previous assertion.

**E2 rung-1 decomposition** (answers "rung 1 is a fused estimator, never
decomposed"). Allocation and rung 2 fixed:

| rung-1 arm | Qwen2 BS | LLaVA BS |
|---|---|---|
| blank contrast only | 31.11 | 30.44 |
| answer-format contrast only | 31.59 | 30.48 |
| elementwise max (deployed) | **32.30** | **31.13** |

max vs blank: 11:26 discordant p=0.020 (Qwen2), 91:108 p=0.257 (LLaVA).
max vs ans: 14:23 p=0.188 (Qwen2), 81:97 p=0.261 (LLaVA).
The fusion beats both components; complementary, not redundant.

**Qwen3-VL SCI3 added** to the 2025 table from the frozen log
(`cda_v8_phaseb/run_b3v6read_qwen3.log`): 21.16/40.94/24.79. Both 2025
blocks are now symmetric and MALT wins all three cells against SCI3 on
Qwen3.

**2025 pass counts corrected** to the replayed values (5.7 Qwen3, 2.2
OneVision) pending the provenance search for the original 5.8/1.9.

## Tier-1 review response (2026-07-21, all zero GPU)

**T1 theory scope.** Prop. 1 restated in the body as exact for the additive
rung-2 operator only; rung 1 replaces rather than subtracts, so the bound
does not license t1 and the paper no longer says it does. Contribution
bullet 2 renamed "Novel Method" (the "with Theoretical Analysis" claim is
withdrawn) and now leads with the risk-control guarantee.

**T2 distribution-free risk control** (`research/malt_risk_control.py` ->
`log/malt_reviewer_experiments/risk_control.json`). Learn-then-Test framing:
threshold grid = candidate family, risk = net accuracy loss vs base.
Conditioned on discordant pairs the harmful fraction is binomial, so an
exact Clopper-Pearson upper bound applies. Natural validation, n=2660:

| backbone | deployed t1 | point gain | harm UB (family-wise, 13 cands) | harm UB (uncorrected) |
|---|---|---|---|---|
| Qwen2-VL | 0.5 | +0.71 | <= +0.52 pts | <= +0.06 pts |
| LLaVA-NeXT | 2.0 | +1.69 | <= +0.13 pts | <= -0.56 pts (certified GAIN) |

Note: a Hoeffding bound on the [-1,1] risk is uselessly loose at this n
(p ~ 1 for every candidate); the discordant-pair binomial is the right
instrument and is what the paper reports.

**T3 OneVision natural delta CORRECTED.** The deployed configuration
(solver's gm3, lam=2.0) replayed on natural Test: **+0.77** (100:77,
p=0.098, 1.95 passes, n=3000). The paper previously printed +1.13, which
the logs attribute to the two-rung ladder at v5 constants. Paper updated;
the number is now weaker and non-significant, and is stated as such.

**T4 label-free threshold.** Setting t1 to a quantile of margin(S0) on
UNLABELLED natural data, other constants frozen:

| backbone | fitted t1 | quantile-95 t1 | fitted BS | quantile BS | passes |
|---|---|---|---|---|---|
| Qwen2-VL | 3.25 | 8.375 | 32.30 | 32.46 | 5.0 -> 5.2 |
| LLaVA-NeXT | 5.03 | 6.312 | 31.13 | 31.29 | 4.86 -> 4.96 |

Flat from the 90th to the 99th percentile. The labelled validation split is
not required for the allocation gate.

## Tuning-headroom answer (2026-07-21) — 2024 development pair

Asked whether the headline numbers can still be improved by tuning. The
project record says no, on both live knobs:

- **t1**: `log/cda_v8_t1_crossover_probe/FINDINGS.md` swept t1 over
  {2.0, 2.375, 2.75, 3.0, frozen, +0.5} on Val with everything else frozen.
  Verdict SAME-FRONTIER NULL: lowering t1 loses BS monotonically to buy
  little or no natural payoff (Qwen 3.25->2.0 costs BS -2.04 for nat +0.38;
  LLaVA strictly worse). The 95th-pct t1 sits at the biased-BS-maximizing
  point and the per-backbone split (3.25 vs 5.03) is confirmed, not
  arbitrary. Filed to the negative atlas.
- **t2**: `log/cda_v4_unified_config_probe/FINDINGS.md` records THREE
  sequential Test draws (0.75, 0.2, then 0.4 pre-committed) after which
  drawing was explicitly STOPPED: "Locating it by further Test draws would
  be binary search on Test - textbook sequential overfitting."

So there is no legitimate hyperparameter headroom left on the development
pair. The one thing that does move the Test number (a higher t1, per the
T4 quantile arm: 32.46 vs 32.30) was already checked on Val and rejected
there because it worsens the natural side; adopting it on the strength of
the Test readout would be Test selection.

Improving these cells requires new mechanism or new data, not new
hyperparameters.

## Prior art inside the project: the LTT idea was probed before

`log/cda_v3_ltt_certificate_probe/FINDINGS.md` ran the same certificate
machinery on the OLD biased-frozen constants and found that certifying at
eps=0.005 costs ~14 BS points, because the aggressive clean2 region cannot
be certified at n=2660. Its own conclusion was that the machinery "becomes
a paper layer (certified operating mode + price-of-guarantee table), not a
gate-passer" -- which is exactly how the new T2 result is used. The new
numbers are tighter because they certify the natural-fitted deployed
configuration rather than the biased-frozen one, and because they report a
bound on the loss rather than a binary verdict at eps=0.

## Remaining zero-cost review items (2026-07-21)

**t1 frontier swept and plotted.** `malt_paper_figures.t1_frontier` sweeps
the allocation threshold over the frozen Test bundle (15 points) and the
teaser panel now shows the measured accuracy-compute curve instead of a
three-point line. The curve dominates the SCI fixed-budget line over its
whole range. Selected values (MCQ-BS):

| t1 | Qwen passes / BS | LLaVA passes / BS |
|---|---|---|
| 1.5 | 4.09 / 28.49 | 3.86 / 27.56 |
| 2.0 | 4.44 / 31.03 | 4.12 / 29.23 |
| 3.25 (deployed Qwen) | 5.00 / 32.30 | 4.56 / 30.52 |
| 5.03 (deployed LLaVA) | 5.18 / 32.38 | 4.86 / 31.13 |
| 12.0 | 5.22 / 32.46 | 4.98 / 31.33 |

The deployed point is marked as validation-selected, not read off this
curve. SCI MCQ-BS references: Qwen 24.54/28.00/29.61 at 3/5/7 passes.

**Decoding-protocol contradiction resolved without an experiment.** The
paper stated three inconsistent things (bf16 + top-k sampling; deterministic
decoding; greedy throughout). The reconciling fact is that the Qwen2-VL
checkpoint ships k=1, so its "top-k sampler" is top-1 and deterministic.
Stated once in App. A; the contradictory sentences removed. This closes
R1's M6 and R3's W8 (no seed variance needed, because there is none).

**GPU corrected to A100.** `nvidia-smi` reports NVIDIA A100 80GB PCIe and
`cda_v8_cost_profile/wallclock.json` records `"gpu": "NVIDIA A100 80GB
PCIe"`. The "A800-80G" in the appendix was copied from
`llava_natural_blank/SCI_SPEC_AUDIT.md`, where it describes **SCI's own
stated environment**, not ours. The paper now says A100 for both the
experiments and the latency benchmark, and attributes only the software
stack to SCI.

**Ethics statement added**: scopes "debiasing" to statistical prior
reliance rather than social bias, and warns that skipping the calibration
step can degrade normal traffic.
