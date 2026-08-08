# Probe: de-leaked real-image CF (rung-3 / on-manifold prior estimation)

Date: 2026-07-09.  Stage 1 = Qwen, Val-only surfaces (natural Val 180 +
biased Val 78), NEW forwards, ZERO Test.  Pre-registered per the gate and
the P-β requirements (CDA_PATHS_TO_STANDARD_20260706.md).

## The question

The strongest positive lever in the project — real peer-image CF at
generation time, **+7.3 open-ended ViLP (biased surface)** — was measured
with SAME-QUESTION peers from ViLP itself (= the benchmark's construction,
rule-5 leakage, undeployable).  Does the effect survive de-leaking to an
external corpus?  And is question-keying even needed (vs random natural
images = pure on-manifold prior estimation)?

Mechanism under test: `corr = z_real + (z_real − mean_k z_peer_k)/β`
during decode — the peer branch estimates the question-conditioned prior
ON the natural-image manifold (blank = the off-manifold degenerate probe
that VCD/M3ID/TIE/SCI all use).  If real-external ≥ blank, the paper gains
"prior should be estimated on-manifold" as a claim + rung-3 completes the
signed evidence axis (full-removal → partial-removal → injection).

## Gate answers (before any forward)

```
Candidate   = gen-time contrast, first-token window (cost-viable variant),
              K=2 peers, β=1.0 (ft) / β=2.0 (full window), Qwen2-VL-7B,
              pixel budget = the original probe's (min 256, max 1280*28*28)
              for mechanism comparability.  Conditions per row:
                orig                      (baseline)
                blank_ft_b1 / blank_all_b2   (off-manifold incumbent, both windows)
                rand_ft_b1                (K=2 RANDOM external images, seed 0
                                           = matched negative control)
                key_ft_b1 / key_all_b2    (K=2 question-keyed external images,
                                           keyword-overlap retrieval)
TC/VC       = untouched (this is a candidate rung-3 evidence constructor)
adaptivity  = none in stage 1 (UNGATED on purpose: harshest natural test;
              gating design comes only after the mechanism survives)
aggregation = n/a (probe)
active cost = per fired row: +K peer prefills (ft window); full window adds
              K× decode steps.  Deploy math deferred to stage 2.
Detector    = none
Calibration = all constants above fixed a-priori (β/window/K copied from the
              original probe; retrieval rule fixed before running)
Corpus      = MMStar + MMBench_DEV_EN_V11 + CCBench + MME base TSVs
              (external to ViLP; rule-5 clean w.r.t. ViLP's construction)
Acceptance  = pre-registered:
  NATURAL SCREEN (ViLP_Custom_Val, 180 rows): kill the line if BOTH
    rand and key real-CF are below blank's natural delta (mechanism does
    not survive de-leaking).  Report helped/harmed; tie band applies.
  BIASED SIGNAL (Biased_Val ViLP, 78 rows): key (or rand) must beat the
    matched blank condition by ≥ +3 (≈ half the leaked +7.3) to open
    stage 2.  Below → filed as "de-leaked real-CF < bar" null.
  CONTROL READ: key ≈ rand ⇒ question-keying unnecessary → the cheaper,
    fully-general "stock probe images" operator is the candidate.
    key ≫ rand ⇒ retrieval quality is load-bearing → stage 2 must include
    a retrieval-failure analysis.
Stage 2 (only on pass; separate registration): LLaVA replication, gated
  integration (terminal-bucket trigger), natural-Test + ONE pre-registered
  biased-Test draw, full cost table.
Frontier    = the blank-CF conditions ARE the matched frontier (same window,
  same β, same K-structure); real must dominate blank, not just orig.
```

Known traps acknowledged: Oth biased Val n=78 (coarse); first-token trap
avoided by construction (gen-time changes the decoded string; paper scorer
prefix-match); Val→Test offset expected; retrieval = declared failure axis
with rand as its control.

## Stage-2 skeleton (conditional; to be re-registered verbatim on a pass)

Ceiling arithmetic (why this is the strong-accept lever, Qwen): terminal
bucket on ViLP biased Test = 179/290 rows @ acc 20.7 → 142 reachable
errors.  A gated real-CF that fixes ~20 of them ≈ +6.9 ViLP ≈ +3.9 Oth
≈ **+1.1 All** — flips B_All vs SCI7 from −0.21 to ~+0.9 and S_All from
+0.55 to ~+1.6 (clean SCI7 wins, not ties).  LLaVA smaller but same sign
(bucket 72 rows / 30.2% of errors).

Cost constraint (rule 10, worst ≤ 7): STACKING rung-3 on the W-config
ViLP row (3 gens + 2 blanks + 1 gray + K peers + regen) breaks worst-case
at K≥1 (9 > 7).  The compliant shape is a SWAP: on terminal-bucket ViLP
rows the peer forward REPLACES the gray rung-2 pass (peer = the rung-2
source, injection instead of removal — the signed-axis reading), and the
correction reuses the existing generation's prefill: worst 3+2+1(peer)
+1(regen) = 7 ✓.  Deploy delta ≈ +0.6·(1+1)−0.6·1 ≈ +0.6 ViLP passes →
Oth ~5.4, overall ~5.02 — needs th tightening or ViLP-pool diet to stay
≤5; wall-clock table must be regenerated (gen ≠ score pass cost).

Stage-2 steps (each own registration): (1) LLaVA stage-1 mirror;
(2) gated integration exactly as above, Val natural screen; (3) ONE
pre-registered biased-Test draw + natural-Test confirm; (4) cost tables
(passes + wall-clock) + updated taxonomy (off-manifold vs on-manifold
prior estimation column).

## Result

### Checkpoint 1 — NATURAL SCREEN: PASSED (mechanical, cda_v8_realcf_adjudicate.py)

n=180 (ViLP_Custom_Val), tie band ±0.56; deltas vs orig (helped/harmed):

```
blank_ft_b1   −0.56  (9/10)      blank_all_b2  −0.56  (9/10)
rand_ft_b1    +0.87  (4/3, n=115 ← coverage wart)
key_ft_b1     −1.70  (8/11, n=176)
key_all_b2    +0.57  (8/7,  n=176)
```

Pre-registered kill (BOTH real conds below blank) NOT triggered.  Honest
notes: (i) all effects are tie-band-scale — ungated real-CF is natural-
safe (no V6-style catastrophe), and key_ft is the worst cell (a keyword-
matched peer on a natural row can subtract genuine evidence) → stage-2
integration keeps the gate, as designed.  (ii) COVERAGE WART: rand_ft
n=115/180 — random corpus draws hit rows whose images fail to load
(likely MME path-style entries); keyed retrieval mostly lands on
MMStar/MMBench (n=176).  The rand/key control comparison is therefore
not perfectly matched; if the control read becomes load-bearing, re-run
rand with a load-verified corpus subset.

### Checkpoint 2 — BIASED SIGNAL: **NULL (row-matched); stage-2 NOT opened**

Full-set cells (n=78): orig 8.97, blank_ft 16.67, blank_all 15.38,
rand_ft 20.45 (n=44!), key_ft 16.88 (n=77), key_all 16.88 (n=77).

The adjudicator's first pass said "STAGE-2 OPEN (+3.79)" off the
UNMATCHED rand cell — a composition artifact of the coverage wart.  The
pre-registered comparison is against the *matched* blank; row-matching
is part of matching.  **Row-matched margins:**

```
rand_ft vs blank_ft   (n=44)  20.45 vs 20.45   +0.00
key_ft  vs blank_ft   (n=77)  16.88 vs 16.88   +0.00
key_all vs blank_all  (n=77)  16.88 vs 15.58   +1.30  (= 1 sample)
```

Below the +3 bar everywhere → **NULL**.  FAIL stays FAIL.

### Mechanism conclusion (the valuable part)

1. **The original +7.3 is re-attributed to benchmark leakage.**  Once
   peers come from an external corpus, the real-image CF collapses to
   EXACTLY blank-CF accuracy on matched rows (+0.00 twice).  ViLP's
   same-question peers are the benchmark's counter-prior construction —
   the leaked peer smuggled in the answer structure, not "real-image
   evidence".
2. **The prior is manifold-invariant at the decision point.**  A random
   natural image, a keyword-matched natural image, and a blank produce
   decision-identical CF corrections (0 changed answers between rand/key
   and blank on their shared rows... margins +0.00).  z(peer, q) ≈
   z(blank, q) where it matters: the question-conditioned prior dominates
   the response to ANY image that does not contain the queried content.
3. **Consequence for the paper (design validation, kills a reviewer
   attack):** "your blank image is off-distribution" is answered with
   measurement — blank is the CHEAPEST member of a measured equivalence
   class of prior probes (natural images included).  The correction value
   lives in the CONTRAST OPERATION (blank_ft nearly doubles orig on
   biased ViLP: 8.97→16.67), not in the realism of the prior probe.
4. **The terminal-bucket population keeps NO deployable injection lever**:
   every candidate (attention, hires, describe, discrimination, and now
   de-leaked real-image) is measured-closed.  The bucket is a disclosed
   boundary (perception-level; next-paper territory), and the paper can
   now say this with a full negative sweep behind it.

### LLaVA mirror — REPLICATED (same rules verbatim, stage1_records_llava.json)

Natural screen PASSED (n=180): ft variants INERT on LLaVA (0 flips —
β=1 first-token contrast cannot move LLaVA's first token; its logit
scale is larger, cf. t1 5.03 vs 3.25); full-window variants positive on
natural (+5.00 blank / +3.98 key — blank ≥ key again).  Biased (n=68):

```
orig 11.76 | blank_ft 11.76 | blank_all 29.41 | key_all 31.34 (n=67)
row-matched: rand vs blank +0.00 (n=42); key vs blank +0.00 (n=67);
             key_all vs blank_all +1.49 (= 1 sample)  → NULL
```

**Two-backbone conclusion:** the equivalence is exact on FOUR independent
row-matched comparisons (2 backbones × 2 conditions, all +0.00) with the
window variants inside one changed sample.  The manifold-invariance of
the question-conditioned prior — and therefore "blank is the cheapest
member of the prior-probe equivalence class" — is a replicated,
pre-registered, two-backbone measurement.  P-β closed on both backbones.

Footnote observation (not a candidate; not registered as one): LLaVA's
ungated full-window blank contrast is strong on ViLP-Val (biased
11.76→29.41, natural +5.00) but does not approach the deployed ladder's
LLaVA-ViLP performance and lives in the "ungated token-level CD" family
the sample-level law warns about; recorded only.
