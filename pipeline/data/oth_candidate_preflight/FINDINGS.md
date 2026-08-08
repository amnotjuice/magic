# Current Candidate Oth Preflight Findings

Date: 2026-07-03

## Standard Gate

This run is a Val-only Oth feasibility preflight, not a main-method result.

Fixed protocol:

```text
TC = candidate-level clean2 visual lift
VC = not evaluated; route-free Val VC residual cache is missing
adaptivity = alpha_T = 1[margin(clean2) + source_agree >= 0.070279]
aggregation = use clean2 candidate scores if alpha_T else original candidate scores
active cost = Oth candidate generation + clean2 real/blank candidate scoring
detector = none
```

The threshold `0.070279` is frozen from the current MCQ candidate.  No Test
metric is used for parameter selection.

## Result

Source: `summary.json` and `comparison_val_oth.csv` in this directory.

Qwen Oth Val coverage is incomplete because only MME TC candidate scores are
cached:

```text
open rows = 146
rows with TC = 68
missing TC = 78 ViLP rows
VC Val cache = missing
```

Qwen Oth Val:

```text
baseline      B/S/BS = 20.00 / 45.83 / 35.29
tc_always     B/S/BS = 31.11 / 43.75 / 38.24
tc_admissible B/S/BS = 33.33 / 43.75 / 39.71
SCI7 Oth bar  B/S/BS = 29.66 / 45.98 / 36.84
```

The fixed TC analogue wins B and BS, but loses S.

LLaVA Oth Val coverage is still incomplete because 53 MME rows are missing TC
candidate scores, though MME and ViLP are both partially present:

```text
open rows = 201
rows with TC = 148
missing TC = 53 MME rows
VC Val cache = missing
```

LLaVA Oth Val:

```text
baseline      B/S/BS = 20.83 / 53.19 / 35.81
tc_always     B/S/BS = 38.89 / 60.64 / 50.00
tc_admissible B/S/BS = 38.89 / 59.57 / 49.32
SCI7 Oth bar  B/S/BS = 38.26 / 60.65 / 51.26
```

Always-on TC is slightly better than the fixed MCQ admissibility rule on LLaVA
Oth Val, but still misses S and BS by a small margin.

## Interpretation

The Oth analogue is real but incomplete:

- TC clean2 transfers conceptually to Oth candidate scoring and improves B/BS
  on Qwen Val and B on LLaVA Val.
- The MCQ `alpha_T` threshold is not clearly optimal for Oth; it mostly acts as
  always-on because `alpha_T` fires on 65/68 Qwen rows and 145/148 LLaVA rows.
- The current Oth Val data cannot test the full route-free framework because
  route-free candidate-level VC residual scores are missing.
- Existing `cda_v2_oth_grec_cache` Test results are useful historical evidence,
  but they rely on old detector-selected Oth VC scoring and cannot be used to
  tune the current route-free candidate.

## Next Step

Materialize Oth Val candidate-level VC residual scores under a route-free policy
before touching Oth Test:

```text
For every Oth Val row with a TC candidate set:
  score candidates on gray / strongblur / centermask
  choose c* per sample using the same image_info_conf rule family
  compute Delta_V(y) = S0(y) - V_c*(y)
  evaluate S_T + alpha_V * Delta_V with alpha_V selected only on Val
```

This is the smallest missing component needed to test whether the current MCQ
candidate can become a unified MCQ/Oth framework.
