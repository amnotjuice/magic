# PREREG — native (token-level) VCD + M3ID baselines on the new backbones

Registered 2026-07-27, before any forward. Authorized by the author
("我认为需要跑 ... 应该用token级"): replace the answer-level ports of
VCD/M3ID in Table 4 with their native implementations, so Table 4's
baseline identities match SCI's published Table 2 protocol and the
answer-level-port disclosure becomes unnecessary.

## Gate answers

- **Candidate** = NOT a MALT candidate. Two baseline rows per backbone
  (Qwen3-VL-8B, LLaVA-OneVision-1.5-8B), official recipes:
  - VCD: cf branch = diffusion-noised image (noise_step=500, the repo's
    existing add_diffusion_noise), fusion per token
    (1+a)*l_cond - a*l_cf with a=1.0 and adaptive plausibility cutoff
    beta=0.1 relative to max conditional prob (VCD paper defaults).
  - M3ID: cf branch = prompt-only (image dropped), per-token weight
    lambda_t = exp(-0.02 * t) applied as in the paper's Eq. (M3ID
    defaults; no tuning).
- TC = none. adaptivity = none. aggregation = the papers' per-token
  operators. active cost = 2 forwards per generated token (reported in
  the Passes column as 2, matching Table 2's convention).
- **Detector** = none. **Calibration** = all hyperparameters a-priori
  from the source papers; nothing fitted; Test never selects.
- **Population** = the frozen phase-B v6 Test rows, identical to
  base/SCI/MALT in replay_newgen_matched_ci.py / malt_heldout_cd.py.
  Validity gate BEFORE any baseline read: base B/S/BS Overall must
  bit-match the frozen targets (qwen3 0.49/34.65/12.56,
  ov15 0.17/52.16/26.58).
- **Acceptance** = descriptive baseline rows, reported as-is in Table 4
  (bold like any other method if a cell wins). No kill rule. ONE Test
  read per method per backbone; this document is the registration.
- **Frontier (gate 8)** = n/a (no new mechanism of ours).

## Mechanics (what actually runs)

1. MCQ + binary rows: single-token readout, so the native per-token
   fusion reduces exactly to fusion on the first-token option logits.
   - VCD: needs orig + noise500 logits — already in the b0/v6supp dumps
     except the supp rows noise500 gap (heldout FINDINGS: 244/887
     covered). GPU top-up: dump noise500 first-token logits for the
     missing supp rows, both models.
   - M3ID: needs a prompt-only (no-image) first-token dump for every BS
     Test row, both models. GPU.
   - Plausibility mask (VCD) applied on the option-restricted logits.
2. ViLP open rows: true token-level decoding. Dual-context greedy loop
   (real + cf KV caches), fused logits per step, MAXNEW=12 as b0;
   emitted string scored by the same generation + prefix-match scorer
   as the frozen base rows. GPU.
3. Environment: /data/shunshungu/tmp/miniforge3/envs/myenv/bin/python
   only. Model loading via cda_v8_newbackbone_n1._load_model.

## Traps checked

- First-token trap: ViLP is decoded and scored by generation + prefix
  match; first-token logic used only where provably equivalent
  (MCQ/binary greedy single token).
- Row alignment: validity gate above (base cells bit-match) before any
  baseline number is read.
- Sequential draws: one read per arm; sequence = this file.

## Paper consequence (pre-stated)

Table 4 VCD/M3ID rows replaced by the native numbers, whatever they
are. App A drops the answer-level-port sentence for VCD/M3ID (TIE stays
an answer-level subtraction, its native form). Expected magnitude of
change on MCQ/binary cells: small (single-token fusion differs from the
port only through the plausibility mask and the branch identity
noise500/prompt-only vs blank); ViLP cells may move more.

## Status

- [x] Pre-registration.
- [ ] noise500 supp top-up dumps (qwen3, ov15).
- [ ] prompt-only dumps (qwen3, ov15).
- [ ] ViLP dual-context decode (VCD, M3ID × qwen3, ov15).
- [ ] Validity gate (base cells bit-match).
- [ ] ONE Test read per arm; Table 4 + App A update.
