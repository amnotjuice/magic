# Phase-B: full SCI-protocol 9-cell tables on the new backbones

Date: 2026-07-12.  User authorization: "新模型实验要做完" — Phase-B is
the mandated results-axis path (old pair's negative cells are
dossier-terminal; new subsets do not inherit them).

## Design (pre-registered before any forward)

Per backbone (Qwen3-VL-8B in nbenv; OneVision-1.5-8B in ov15env):

**B0 — step0-equivalent dumps.**  5 variants × 6 datasets (MMStar,
CCBench, MMBench_DEV_CN/EN_V11, MME, ViLP), NAT-parity construction:
  - Original: real image, default prompt
  - VCF-Color0: blank image, default prompt   (SCI's vcf_color0)
  - VCF-Noise500: noise500 image, default     (SCI's vcf_noise500)
  - TCF-V1 / TCF-V2: real image, SCI's tcf prompt templates
    (prompt_tcf1/prompt_tcf2 in cda_v3_qwen_natural_final — the repo's
    transcriptions of SCI's TCF instructions)
  MCQ/binary: first-token option logits (proven equivalent for MCQ);
  ViLP: greedy generation (paper scorer downstream).
  DISCLOSURE: this is a reimplementation of step0 in the new models'
  canonical envs (the original vlmeval pipeline cannot load them);
  construction mirrors the frozen-model NAT path used everywhere else.

**B1 — step1-equivalent subset generation.**  SCI's construction rules
applied to the model's OWN predictions:
  - VCF-biased: wrong on real AND same answer under blank (prior-driven)
  - TCF-biased: inconsistent answers across {default, tcf_v1, tcf_v2}
  - Biased = union tagging (vcf_only / tcf_only / both); Val/Test split
    mirrors the repo's ratio (Val ≈ 20%).
  Sanity gates: subset sizes within 2× of the old backbones' per-dataset
  proportions; construction script + row lists archived.

**B2 — per-model calibration (natural first, label-free).**  Quantile
POSITIONS recalibrated on the model's natural Val margins (t1 pct via
natural no-harm sweep on Val ONLY; τ_T/t2/τ_V positions from the same
label-free procedure as the ledger).  The N1 lesson is binding:
positions are not model-invariant; each model gets its own Val-frozen
set, published in the parameter ledger.

**B3 — evaluation on the new Test subsets (ONE pre-registered draw per
backbone).**  Rows: our ladder; reimplemented SCI5 & SCI7 (β/α/γ/θ from
the paper's tuned values for the old models as the starting point,
disclosed; if a small grid is needed it is Val-only and published);
gated-naive (M3ID λ=2); base.  Deliverables: 9-cell biased table,
natural table, cost table (passes + wall-clock rates from the measured
per-pass latencies).

**Acceptance (SA-R1 discipline):** vs cost-matched reimplemented SCI —
no negative cell, majority claimable wins (outside band + CI excl. 0),
natural ≥ 0 (tie band), overall cost ≤ 5.
**Kill:** structural weak cells on the new subsets → dossier + honest
report; no rescoping; no second draw without user authorization.

Cost estimate: ~18k rows × 5 variants ≈ 90k scoring forwards + ViLP
generations per backbone (~25-35 GPU-h each); subset evaluations are
small by comparison.  Sequential: Qwen3 first.

## B3 design decisions (recorded before building)

**SCI formula (verbatim from modeling_qwen2_vl.py, SCI5 branch):**
`consistent = max(orig, tcf1, tcf2)/γ` (confidence type='constant');
`unbiased = (orig − (vcf1+vcf2)/2)/β`; `final = consistent + unbiased`,
masked −inf where `consistent < log(θ) + max(consistent)`.  SCI7 adds
vcf3(noise400)+tcf3 → for the new backbones we reimplement **SCI5 as the
primary baseline** (SCI7 needs 2 extra variants; deferred unless needed).

**Raw-logits fidelity issue:** B0 stores log-softmaxed option scores;
SCI's cross-branch max is NOT shift-invariant.  Resolution: B1 (subset
generation) uses predictions only (argmax = shift-invariant, unaffected);
B3's SCI readout uses RAW option logits collected in the B3 evidence
pass (below) on exactly the rows where SCI is evaluated.  No approximation
in any reported cell.

**B3 evidence pass surfaces (per model, pre-registered):**
- biased Val+Test rows (from B1): 7 ladder views + raw {orig, vcf2=noise500,
  tcf1, tcf2} for SCI → 10 forwards/row (blank shared);
- natural-Val = all_data_val (i%5==0, ≈3.3k): 7 views (calibration, B2);
- natural-Test readout: pre-registered stratified sample of all_data_test
  (3000 rows, seed 0, proportional per dataset) × 7 views — full 13k×7
  is disallowed by budget; the sample size gives band ≈ ±0.03.
Estimated ≈ 60-90k forwards/model (~13-18h).

**Open-ended (ViLP) on new backbones:** candidate pools via the deployed
3-template generation (already in B0's tcf variants? NO — B0's open
generations are single-template per variant; pools need the 3 GEN_PROMPTS
on real image = ALREADY in B0 (orig=default, tcf1=answer_format,
tcf2=paraphrase generations on real image) → pool = dedup of those 3 ✓
zero extra generation cost; scoring views in the B3 pass).

## B2 AMENDMENT (2026-07-12, BEFORE any Test read; disclosed)

The registered B2 (t1-pct sweep only; τ_T at the ledger's 5% tail) fails
natural no-harm on qwen3 at EVERY t1 percentile (Δ≈−1.5 flat in t1 —
harm enters through admission on low-margin rows, not allocation).
This replicates N1's lesson at higher resolution and matches the
design's own doctrine (admission is the harm controller).  AMENDED B2:
2-D Val-only sweep {t1_pct × τ_T_pct} on natural_val for no-harm at max
coverage, with per-type (mcq/binary/open) deltas reported; t2/τ_V stay
label-free.  Test remains untouched until a config passes Val no-harm;
if NONE passes, Phase-B acceptance fails at B2 and is reported as such.

## B3 qwen3 — PRE-REGISTERED TEST READ: **FAIL (kill rule executed)**

Constants (amended-B2, Val-frozen): t1=20.375(95pct) τ_T@20pct t2/τ_V
label-free.  n=1766 Test rows; SCI5 params Val-selected (β=.1 γ=2.5 θ=.2).

```
              B_All   S_All   BS_All
base           0.00   47.25   14.10
SCI5-reimpl   17.69   40.04   22.42   (ungated, no natural constraint)
gated-M3ID    15.71   45.54   23.22
LADDER        10.54   43.83   19.54   ← loses B/BS to BOTH baselines
```

Acceptance (no negative cell vs cost-matched SCI, majority claimable
wins): **FAIL decisively.**  No second draw (per registration).

**Post-mortem (the mechanism, honest):** the natural-no-harm constraint
on this strong model forces the admission tail from 5%→20% and the
label-free t2/τ_V positions land very high (6.26/3.25) — the calibrated
ladder keeps only ~60% of the ungated correctors' biased repair.  The
baselines in this table are NOT held to a natural constraint (their
natural harm is unmeasured here — SCI raw views were collected on biased
rows only).  This is the margin law's own tradeoff surfacing on a
new-generation model: **as the encoder improves, the correctable-vs-
harm frontier tightens, and a no-harm-constrained corrector retains less
of the unconstrained repair.**  The old backbones sat at a point where
the constraint was nearly free; qwen3 does not.
Caveats for interpretation (not excuses): t2/τ_V positions were never
re-validated for this regime (label-free transplant); the iic source
signal was replaced by a top-margin proxy (no vocab maxp saved); SCI5
comparison is first-token/candidate-level reimpl.  Any repair of these
would require a NEW registration (user-gated) — not attempted.

## B3 ov15 — PRE-REGISTERED TEST READ: **FAIL (kill rule executed)**

Constants (amended-B2): t1@95pct=10.02, τ_T@10pct (natural Val +0.39 —
far healthier than qwen3's band-edge; N1's 4× milder harm confirmed).
n=2248; SCI5 Val-selected (β=.1 γ=2.5 θ=.2).

```
              B_All   S_All   BS_All
base           0.16   26.62    8.76
SCI5-reimpl   29.99   49.73   31.49   (ungated; natural unmeasured)
gated-M3ID    20.49   48.38   24.96
LADDER        22.18   47.43   26.20   ← beats gated-M3ID on B/BS,
                                        loses all three All-cells to SCI5
```

Acceptance: **FAIL.**  No second draw.

## PHASE-B FINAL VERDICT (both arms adjudicated)

Both arms fail the registered acceptance for the same structural reason:
**on the new-generation models, the natural-no-harm-constrained ladder
retains less biased repair than the unconstrained SCI5 fusion** (qwen3
~60%, ov15 ~74% of SCI5's B_All).  On the old backbones the constraint
was nearly free (the frozen method beat SCI at matched cost); on the new
ones the correctable-vs-harm frontier has tightened past the crossing.

Fairness disclosure (cuts against the baselines, not us): the 9-cell is
biased-only; SCI5-reimpl's NATURAL harm on these models is unmeasured
(raw views collected on biased rows only).  A natural-constrained SCI5
would lose repair the same way — but measuring that was not in the
registration and is not claimed.

What Phase-B DID buy (all durable): the constraint-frontier tightening
is itself a measured, two-family finding (the law's own prediction);
per-model calibration machinery validated; new-generation biased subsets
+ full dumps archived for any future work.  The strong-accept结果轴
claim ("all-positive new-model 9-cell") is measured-CLOSED, not open.

## B3-v2 REGISTRATION (2026-07-13; user challenged the B3 reads — audit
found three asymmetries, ALL disfavoring the ladder)

Audit: (a) SCI5 was param-selected on biased-Val ACCURACY while the
ladder's calibration optimized only natural no-harm + a coverage proxy
(never biased-Val power among no-harm configs); (b) the natural
constraint bound ONLY the ladder — SCI5's natural harm on these models
is unmeasured; (c) the ladder ran with a degraded iic proxy.  The B3
FAIL verdicts are therefore CONTAMINATED reads; "frontier tightening"
is downgraded to pending-recheck.

B3-v2 symmetric contract (Val-only until re-authorized Test):
1. natural_val SCI views supplement (noise500/tcf1/tcf2 + raws) both
   models → measure SCI5's natural delta per param set;
2. BOTH methods calibrated identically: among Val natural-no-harm
   configs (ladder: {t1_pct×τ_T_pct}; SCI5: {β,γ,θ grid incl. its paper
   sets, θ as its conservatism knob}), select by biased-Val power;
3. real-iic supplement = optional step 2 pending results;
4. ONE new pre-registered Test read per model — REQUIRES explicit user
   authorization (kill-rule discipline); not run before it.

## B3-v2 FINAL VERDICTS (symmetric contract; authorized reads)

**qwen3: PASS — LADDER beats fair-SCI5 on ALL NINE cells.**  SCI5's
paper params FAIL natural no-harm on qwen3; its no-harm point (θ=0.7)
collapses below base (BS_All 13.48 < 14.10).  LADDER-v2 11.46/43.83/
20.16.  gm3(λ=2) also DQ'd on natural (−0.39); fair gm3(λ=1, Val-
selected) 12.10/46.68/20.78 = mixed 6/9 vs ladder (≤0.9 B/BS gaps,
ladder on degraded iic proxy) — disclosed as "the gate carries more on
new models".
**ov15: FAIL — SCI5's paper params PASS no-harm on ov15 (natΔ ≥ band)
and beat LADDER-v2 on all All-cells** (29.99/49.73/31.49 vs 22.18/
47.43/26.20).  Ladder does beat fair gm3 on B/BS here (opposite of
qwen3).  Kill rule: filed, no further draw.  Remaining registered
option: real-iic supplement (~10h; historical +3-6 MCQ unlikely to
close B_All −7.8) — user-gated.

**Cross-model law (the durable finding):** whether the constrained
ladder or the unconstrained-fusion wins is decided by WHERE the model's
natural no-harm constraint bites: qwen3 (strong, prior-light) forces
SCI to a collapsed operating point; ov15 tolerates SCI's full strength.
The old pair sat on the qwen3 side (published SCI5 natural +0.4 was
already at its ceiling).  One constraint, model-dependent binding =
the frontier story, now measured on both sides.

## ANCHOR TEST (2026-07-13, user doubted the SCI-reimpl numbers)

My SCI5 formula applied to the OFFICIAL old-Qwen2 dumps (full-vocab
first-token logits produced by the real SCI pipeline, 1261 biased-Test
MCQ rows):

```
β=.2 γ=2.0 θ=.3:  24.79-24.98 / 45.82-46.18 / 27.20-27.44
paper SCI5 MCQ :  24.91       / 47.22       / 28.00
```

Reimplementation reproduces the published cells to within tenths (S −1
consistent with the documented +17-row subset diff); option-restricted
≡ full-vocab to ≤0.24 (that deviation immaterial).  **The SCI5-reimpl
machinery is anchor-validated; the ov15 verdict stands** (and its
pattern matches the published record: LLaVA-family has always been
SCI-friendly — old LLaVA SCI5 Oth cells 37.97/60.65/51.01 were huge,
and our old-pair LLaVA S_All also lost to SCI).

## ov15 VERDICT TRIPLE-CHECKED (2026-07-13, user disbelief round 2)

1. SCI5-reimpl anchor-validated on official dumps (±tenths of paper).
2. Pool-collapse hypothesis REJECTED: ov15 pools avg 2.49 cands, 1%
   singletons, gold 47.3% (vs the old-LLaVA 3-template disaster 1.4/22.6%).
3. FULL 4-constant symmetric sweep (128 configs {t1,τ_T,t2,τ_V} pcts,
   natural no-harm constrained, Val-power selected): ladder ceiling
   22.82 Val acc vs frozen 21.81 (**+1.0 only**) vs SCI5's 28.69 — the
   half-frozen calibration was NOT the suppressor.

**Why "old LLaVA mostly won" does not transfer (the honest account):**
(i) old-pair wins were achieved by the fully-engineered production
pipeline (beam5 pools, 18-pass view batteries, real vocab-maxp iic,
extensively Val-tuned constants) — the B3 skeleton is a leaner
instantiation; real-iic remains its one unpriced handicap (~+3-6 MCQ
historical; cannot close the Oth-heavy −11.3 B_Oth gap).
(ii) more fundamentally: on the old pair, published SCI5 was ALREADY at
its natural-safety ceiling, so matched-constraint comparison left SCI no
headroom; on ov15 SCI5 is natural-safe at FULL power — there is no
constraint asymmetry to win from.  Same law, other side of the frontier.

## FAIR-PROTOCOL v3 FINAL READS (2026-07-13; user's 5th challenge fixed
the scorer: canon = lowercase + strip articles + digit↔word + plural
fold → prefix, else whole-word containment on ≤4-word answers;
registered BEFORE effect on methods was seen; B1-v2 subsets ⊆ B1)

Scorer-fairness magnitude: ov15 ViLP base 40.67→53.56 (+12.9; now >
old-LLaVA 51.53 = generational ordering restored); qwen3 63.44→67.67.
False-biased rows removed: 46 (qwen3) / 68 (ov15).

```
qwen3  (n_test=1734)      B_All   S_All   BS_All
  base                     0.57   48.24   13.73
  SCI5-fair (θ→0.5)        9.22   45.96   16.21
  gated-M3ID(λ2, unvetted) 16.45  49.90   23.99
  LADDER                  12.34   50.52   21.57   → beats SCI5 9/9 ✓✓
ov15   (n_test=2200)
  base                     0.64   32.40   10.09
  SCI5-fair (paper-family) 27.91  53.02   29.77
  gated-M3ID              21.00   52.72   25.59
  LADDER                  22.34   51.55   26.82   → loses All-cells to
                                    SCI5 (−5.6/−1.5/−2.9, narrowed) ✗
```

**FINAL two-family verdict (5 user challenges, anchor test, pool
autopsy, full-constant sweep, fair scorer — all absorbed):**
qwen3 PASS 9/9 vs fair-SCI5; ov15 FAIL with narrowed margins.  The
frontier law stands: the winner is set by where the model's natural
no-harm constraint bites (qwen3+old-pair side vs ov15 side).  gm3 rows
carry the λ-not-natural-vetted caveat on qwen3 (λ=2 failed no-harm
pre-fair; its fair-λ row is λ=1 per the earlier check).

## Progress log

- 2026-07-12: B0 qwen3 first pass DONE but INCOMPLETE (6681/16566) —
  VLMEvalKit image-dedup references (short `image` values pointing at
  another row's index) were silently skipped.  Fixed loader (resolve
  references; 16566/16566 loadable, 0 failures), invalidated the partial
  B1 output, resumed B0 for the missing ~9.9k rows (run_b0_qwen3_resume).
  Partial-dump B1 numbers (1047 biased test rows) are NOT to be cited.

## V5 FINAL VERDICTS (fully protocol-aligned: official pixels+prompts+
templates; subset/evidence aligned (base B≈0 both); ALL methods
symmetrically natural-gated incl. gm3.  THE citable reads.)

qwen3 (n=1759):            B_All  S_All  BS_All
  base                      0.49  34.65  12.56
  SCI5 (paper params)      22.36  47.40  26.38
  gm3(λ2)                  18.27  42.20  23.14
  LADDER                   24.05  45.98  27.80   → vs SCI5 6/9 (B,BS all
                            won +1.2~+3.4; S column −1.3~−1.9); > gm3 9/9.
ov15 (n=2728):
  base                      0.17  52.16  26.58
  SCI5 (θ=.7)              13.27  54.24  32.51
  gm3(λ2)                  18.16  57.48  36.99   ← beats LADDER 9/9 (!)
  LADDER                   14.05  56.33  34.86   → vs SCI5 8/9 (only
                            B_Oth −3.2); loses to own gated-naive.

**Two-model conclusions:**
1. vs the incumbent SCI: LADDER wins the strict majority on BOTH new
   models under the clean protocol (6/9, 8/9).  The earlier "loses to
   SCI on ov15" verdicts were PROTOCOL-ARTIFACT STACKS (pixels, prompts,
   scorer, asymmetric contracts) — the user's disbelief was correct at
   every step.
2. Honest internal finding: on ov15 the natural-vetted gated-M3ID beats
   the full ladder 9/9 (on qwen3 & the old pair the ladder wins big) —
   WHICH machinery earns its passes is model-dependent; the gate is the
   universal part.  This goes in the paper as a finding, not a footnote.
3. S-column on qwen3 (−1.3~−1.9) = the family's cross-generation soft
   cell (mirrors old-LLaVA S).  SAFE-2 (no negative headline cell) is
   NOT met by the method-first shape on either model.

## Natural-TEST readout, new models (v5 constants, held-out 3000-row
stratified samples): qwen3 +0.100 (91/88), ov15 +1.133 (81/47) — the
ladder's natural no-harm TRANSFERS to Test on both new-generation
models (SAFE natural clause met on the new pair).

## V6 PROGRAM (user directives 2026-07-14: full-strength instantiation —
"我prefer我的结果更强"; SCI3+SCI7 in the new-model tables; Model-Zoo-
grade documentation).  Registered components:
  (a) true iic (vocab maxp, 4 views) replacing the margin proxy;
  (b) SCI7 branches (noise400 + official prompt_variation3 tcf_v3) with
      symmetric natural gating; SCI3 computed offline from existing views;
  (c) ViLP beam pools + full-candidate scoring (stage 2 of v6);
  (d) MODEL ZOO section (envs/precision/decoding disclosed; greedy vs
      SCI's qwen top-k sampling = disclosed deviation).
v6 read = base/SCI3/SCI5/SCI7/gm3/LADDER-full, all natural-gated.

## V6 FINAL (full-strength ladder + complete SCI ladder, both models)

qwen3: LADDER-full 24.54/45.20/27.97 — beats SCI3 9/9, SCI5 B/BS swept
(B_All +2.18), **SCI7 (matched subset) B +2.96 / BS +2.37**; S column
−1.5~−2.6 remains the sole soft cell.  True iic earned +0.5 B_All.
ov15: **INVERTED SCALING** — SCI3 23.38/57.70/40.03 DOMINATES everything
(SCI3 > gm3 > SCI7 > LADDER-full ≈ SCI5): on the newest LLaVA, MORE
counterfactual branches HURT (SCI3>SCI5 reverses the published old-pair
ordering); the full ladder beats SCI5 (+0.5/+2.0/+2.0) but loses to the
simplest member by ~6 BS.  True iic did not help ov15 (≈ proxy).

**System-level law (the paper's cross-model finding):** the optimal
point on the correction-complexity axis is MODEL-DEPENDENT — qwen3
rewards the full gated machinery; ov15 rewards the simplest sufficient
corrector; the old pair sat in between.  No fixed pipeline is
universally optimal; the margin gate is the only universally-safe
component (natural no-harm holds everywhere it is applied).
Method-first headline on new models: dead (SAFE-2 unmet definitively).
Finding-first: strengthened (inverted scaling on ov15 is novel).

## DUAL-PROTOCOL SCI ROWS (paper protocol = biased-Val-acc selection, NO
natural gate — mirrors validation.py; vs our no-harm contract)

qwen3 (paper-protocol selection = same configs; natural was safe anyway):
  SCI3 21.16/40.94/24.79 nat+1.48 | SCI5 22.36/47.40/26.38 nat+1.24 |
  SCI7* 22.28/46.31/26.67 nat+1.08 — LADDER-full 24.54/45.20/27.97
  nat+0.10 beats SCI5/SCI7 on B/BS **under BOTH protocols**.
ov15 (paper protocol): SCI3 25.24/57.77/41.13 nat−0.48 | SCI5 24.79/
  52.09/38.49 nat−1.21 | SCI7* 33.68 nat−1.24 — **every paper-protocol
  SCI variant harms ov15's natural distribution** (noise-branch poison,
  measured −5.6 pure-contrast); LADDER nat+1.13.  Under no-harm: ladder
  > SCI5/SCI7-constrained; SCI3(γ1.5, natural-safe) 23.38/57.70/40.03
  still dominates = the inverted-scaling finding stands as an honest cell.
Presentation rule: ALL new-model tables carry both protocol rows with
the natural column adjacent — the tradeoff speaks for itself.
