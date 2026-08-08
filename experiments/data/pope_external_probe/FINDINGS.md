# POPE external-benchmark probe (closes the circularity attack path)

Date: 2026-07-14.  Pre-registered BEFORE any forward.

Candidate = FROZEN old-pair deployment, zero POPE tuning:
  LADDER: the deployed binary-Oth path (envelope base, clean2 lift,
  agree, gray residual) at the v8 FROZEN OTH_LADDER constants
  (qwen 3.25/.070279 + 0.5/−.220826; llava 5.03125/.070279 + 0.5/.375).
  SCI5: paper params β=.2 α=1 γ=2.0 θ=.3 (the config that reproduced
  published Table-2 cells in the anchor test), their formula verbatim.
  base: first-token Yes/No argmax.
Surface = stratified sample: 500 rows × {adversarial, popular, random}
  × {yes, no} = 3000 rows (seed 0), lmms-lab/POPE test.  Prompt =
  question + "Please answer yes or no." (the official binary
  instruction; TCF templates from the official replace-list).
Views/row = 8: s0, blank, ans_real(tcf1-prompt real), ans_blank,
  gray, noise500, tcf2 (+raws from the same forwards for SCI).
Models = old pair (Qwen2-VL-7B, LLaVA-NeXT-8B) in myenv — the
  published-row backbones; this is a TRANSFER test of the frozen method.
Acceptance = per backbone: LADDER accuracy ≥ base − band (no-harm on an
  external distribution) AND ≥ SCI5 − band (parity acceptable, win better).
Kill = below either → filed as scoped limitation, no rewording.
Band = one changed sample ≈ ±0.033.

## Result
(filled by run)

## Result — LLaVA arm: KILL (filed as scoped limitation, no rewording)

base 86.50 | SCI5(paper) 87.07 | LADDER(frozen) 85.33  (n=3000, ±0.033)
LADDER fails both clauses (no-harm −1.17; vs SCI5 −1.74).  Mechanism
reading: frozen OTH constants transferred without any POPE-side
calibration — the operating-point law again (constants are
distribution-calibrated, not universal).  SCI5's fixed fusion happens to
transfer mildly positively here.  qwen2 arm pending.

## Result — qwen2 arm: **PASS with margin**

base 88.90 | SCI5(paper) 88.80 | **LADDER(frozen) 89.80**  (n=3000, ±0.033)
LADDER beats base +0.90 (no-harm ✓, outside band) AND SCI5 +1.00, with
gains in all three categories (adv +0.90 / pop +1.20 / rand +0.60).
Zero POPE tuning; frozen v8 constants.  FINAL POPE VERDICT: split —
Qwen arm transfers positively (circularity path closed positively on
the primary backbone); LLaVA arm = scoped limitation (operating-point
law; constants are distribution-calibrated).

## 2026-07-17 — FULL LLaVA run (n=9000; goal item ⑦)

Sampler switched to full POPE (POPE_N=500 reproduces the old 1/3 subsample).
LLaVA full-n readout (same frozen constants, zero POPE tuning):

  base 86.98 / SCI5 87.20 / MALT 85.86
  MALT vs base : −1.12  (96:197, McNemar z=−5.90, p<1e-4)  — replicates −1.17
  MALT vs SCI5 : −1.34  (196:317, z=−5.34, p<1e-4)
  SCI5 vs base : +0.22  (247:227, z=+0.92, p=0.36)  ← NEW: SCI5 itself does
                 not transfer on LLaVA-POPE; family-level mismatch.

Qwen full run was killed (cost call: ~24 h contended for one CI narrowing on
an already-significant positive cell); Qwen stays at the disclosed n=3000
subsample. Paper §POPE rewritten accordingly.
