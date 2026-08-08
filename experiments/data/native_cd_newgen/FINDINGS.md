# Native (reference-implementation) TIE / VCD / M3ID on the new backbones

Pre-registration: PREREG.md (this directory). Executed 2026-07-27.
Scripts: `research/malt_native_cd_supp.py` (GPU top-up),
`research/malt_native_cd.py` (scoring).

## What "native" turned out to mean

Reading the reference implementation this repo is built on:

- `vlmeval/vlm/qwen2_vl/model.py` — **all three** baselines build their
  counterfactual branch as `visual_type='vcf_noise500'` (TIE included;
  the earlier answer-level port had used the blank branch).
- `modeling_qwen2_vl.py` guards the fusion with `logits.shape[1] > 1`
  and `modeling_llava_next.py` with `decode_flag == 1`: the contrast is
  applied **only at the prefill step, i.e. the first generated token**.
- The authors state this in their own comment: *"Current BS-Subsets only
  predict one character or one word, so we don't need to take care of
  auto-regressive generation, focusing on the first token is enough."*

So for MCQ and binary rows — where the answer IS the first token — the
"token-level" recipe is mathematically identical to a fusion of the
first-token option scores. The token-level / answer-level distinction
raised in review does not exist on those rows.

Deployed formulas (published constants, nothing fitted):

    TIE  : fused = org - cf                       theta=0.5
    VCD  : fused = (1+a)org - a*cf,  a=1.0        theta=0.3
    M3ID : fused = org + (1-a_t)/a_t (org - cf)   theta=0.3
           a_t   = exp(-0.02 * (n_prompt - n_image))     [per row]
    all  : cutoff = log(theta) + org.max();  fused[org < cutoff] = -inf

## Validity gates (both PASSED, before any baseline number was read)

- **G1** base B/S/BS Overall bit-match the frozen phase-B v6 targets:
  qwen3 0.49 / 34.65 / 12.56 (n=1759), ov15 0.17 / 52.16 / 26.58 (n=2728).
- **G2** the GPU top-up recomputed the open rows' `s0` pool scores and
  they match the frozen cache exactly (max |diff| = 0.0000 on both
  backbones) — the forward path is deterministic, so the newly dumped
  `noise500` pool view is bit-consistent with the rest of the cache.
- Candidate-set vs full-vocabulary plausibility mask: the two maxima are
  identical on 100% (qwen3) / 97.5% (ov15) of open rows and differ by
  more than 0.7 nats on 0.3%, so restricting the mask to the shared
  candidate set (which every method in Table 4 is scored on) is exact in
  effect.

## Test read (ONE per arm per backbone, as registered)

Qwen3-VL (n=1759)          B      S      BS
  base                    0.49  34.65  12.56
  TIE   native            5.71  32.44  13.47   (port was 6.84/36.38/15.01)
  VCD   native            7.19  32.60  14.27   (port was 8.74/37.32/16.32)
  M3ID  native            8.74  33.54  15.18   (port was 9.24/37.80/16.66)
  SCI5                   22.36  47.40  26.38
  MALT                   24.54  45.20  27.97

OneVision-1.5 (n=2728)     B      S      BS
  base                    0.17  52.16  26.58
  TIE   native           11.75  52.81  31.60   (port was 12.31/55.61/32.88)
  VCD   native           12.03  53.09  32.00   (port was 13.04/56.19/33.61)
  M3ID  native           15.96  53.60  34.35   (port was 15.12/56.55/34.90)
  SCI5                   13.27  54.24  32.51
  MALT                   18.16  57.41  36.95

M3ID per-row weight (1-a_t)/a_t: median 1.77 (qwen3, median 51 text
tokens) and 2.32 (ov15, median 60).

## Reading

Native numbers sit below the earlier answer-level ports on almost every
cell, because the published recipes contrast against a noised image
rather than the (stronger) blank probe. Ordering is unchanged —
TIE < VCD < M3ID on both backbones, all far below SCI5 on Qwen3-VL, and
MALT best on every cell of both backbones. On OneVision-1.5 native M3ID
still beats SCI5 on B and BS, as the port did.

## AMENDMENT 2026-07-27 — per-paper branches (this is what the paper uses)

Running all three against noise500 (the released implementation's uniform
choice) turned out to *handicap* two of them, because only VCD is defined
against a noised image:

| method | branch its OWN paper defines            | released impl |
|--------|------------------------------------------|---------------|
| TIE    | void / dummy image  (Niu et al. 2021)    | noise500      |
| VCD    | diffusion-noised image (Leng et al. 2024)| noise500  ✓   |
| M3ID   | image-free conditional (Favero et al. 24)| noise500      |

So each baseline is now run with **its own paper's branch**; the
image-free branch for M3ID was dumped on GPU
(`malt_native_m3id_textonly.py`). Full matrix (B / S / BS):

Qwen3-VL        noise500              blank                 text-only
  TIE      5.71 32.44 13.47   |  6.84 36.38 15.01  |      --
  VCD      7.19 32.60 14.27   |  8.74 37.32 16.32  |      --
  M3ID     8.74 33.54 15.18   |  9.17 37.64 16.49  |  8.67 35.91 15.52

OneVision-1.5   noise500              blank                 text-only
  TIE     11.75 52.81 31.60   | 12.31 55.61 32.88  |      --
  VCD     12.03 53.09 32.00   | 13.04 56.19 33.61  |      --
  M3ID    15.96 53.60 34.35   | 15.96 56.69 35.41  | 16.02 56.33 35.04

DEPLOYED in Table 4 (each method's own branch):
  Qwen3-VL       TIE 6.84/36.38/15.01  VCD 7.19/32.60/14.27  M3ID 8.67/35.91/15.52
  OneVision-1.5  TIE 12.31/55.61/32.88 VCD 12.03/53.09/32.00 M3ID 16.02/56.33/35.04

Note M3ID's weight: its published schedule exp(-gamma*t) runs over the
GENERATED index, so at the first token it is inert (weight (1-a)/a -> 0)
and M3ID would collapse onto the base model on a single-token benchmark.
That is presumably why the released implementation substitutes the prompt
length; we keep that substitution and disclose it.

Robustness disclosed in App A: contrasting all three against blank raises
VCD by <= 2.1 BS and moves TIE/M3ID by < 1, and leaves every bold entry of
Table 4 unchanged (checked: ov15 blank VCD 33.61 > SCI5 32.51 but still
< MALT 36.95).

## Paper consequence

- `paper/tables/tab_newgen.tex`: TIE / VCD / M3ID rows replaced by the
  native numbers; bold/underline recomputed (ov15 S second-best moves
  from M3ID to SCI5).
- `paper/8_implementation_details.tex`: the answer-level-port sentence is
  replaced by the reference-implementation description above.
- Supersedes the TIE/VCD rows of `research/log/malt_heldout_cd/FINDINGS.md`
  (that run's blank-branch port is no longer used by the paper).
