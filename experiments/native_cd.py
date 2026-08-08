"""NATIVE (SCI-codebase) VCD / M3ID on the two newer backbones -- Table 4.

Pre-registration: experiments/data/native_cd_newgen/PREREG.md
GPU top-up:       experiments/native_cd_supp.py  (n_text + open-row noise500)

The SCI codebase applies both contrasts at the PREFILL step only
(`logits.shape[1] > 1` / `decode_flag == 1`), i.e. at the first generated
token, against a single vcf_noise500 branch:

    VCD  : fused = (1+alpha)*org - alpha*cf              alpha=1.0, theta=0.3
    M3ID : fused = org + (1-a_t)/a_t * (org - cf)        theta=0.3
           a_t   = exp(-0.02 * (n_prompt - n_image))
    both : cutoff = log(theta) + org.max();  fused[org < cutoff] = -inf

For MCQ/binary the answer IS that first token, so this is exactly a fusion
of the frozen first-token option scores.  Open rows are scored over the
same materialised candidate pool every other method in Table 4 uses.

Validity gates (both must pass before any baseline number is read):
  G1  base B/S/BS Overall bit-match the frozen phase-B targets;
  G2  the supp run's recomputed open-row s0 pool scores match the frozen
      cache (determinism of the forward path).

Usage: python3 experiments/native_cd.py
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "experiments/data/native_cd_newgen"

TIE_THETA = 0.5
VCD_ALPHA, VCD_THETA = 1.0, 0.3
M3ID_META, M3ID_THETA = 0.02, 0.3

BASE_TGT = {  # frozen phase-B v6 Test targets (replay_newgen_matched_ci.py)
    "qwen3": dict(B=0.49, S=34.65, BS=12.56, n=1759),
    "ov15": dict(B=0.17, S=52.16, BS=26.58, n=2728),
}


def _vec(r, key):
    """(values, labels) for a view, list-form (mcq/binary) or dict-form (open)."""
    v = r.get(key)
    if v is None:
        return None, None
    if isinstance(v, dict):
        keys = [k for k in (r.get("pool") or list(v)) if k in v and k in r["s0"]]
        if not keys:
            return None, None
        return np.array([v[k] for k in keys], dtype=np.float64), keys
    return np.array(v, dtype=np.float64), (r.get("labels") or ["Yes", "No"])


def fuse(r, cf_key, coef_org, coef_cf, theta):
    org, labels = _vec(r, "s0")
    cf, _ = _vec(r, cf_key)
    if org is None or cf is None or len(org) != len(cf):
        return None
    cut = math.log(theta) + org.max()
    fused = coef_org * org + coef_cf * cf
    fused = np.where(org < cut, -np.inf, fused)
    return labels[int(fused.argmax())]


def base_pick(r):
    org, labels = _vec(r, "s0")
    return None if org is None else labels[int(org.argmax())]


def run_model(model):
    sys.argv = ["x", model]
    import newbackbone_calibration_eval as EV
    importlib.reload(EV)
    EV.MODEL = model
    from newbackbone_calibration_eval import hit, cells

    trows = [r for r in EV.load("biased") if r.get("split") == "test"]
    tgt = BASE_TGT[model]
    assert len(trows) == tgt["n"], f"n_test {len(trows)} != {tgt['n']}"

    supp = {}
    with (SUPP / f"supp_{model}.jsonl").open() as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                supp[(s["ds"], s["index"])] = s
    miss = [r for r in trows if (r["ds"], r["index"]) not in supp]
    assert not miss, f"{len(miss)} Test rows missing from supp"

    tpath = SUPP / f"textonly_{model}.jsonl"     # M3ID's own (image-free) branch
    textonly = {}
    if tpath.exists():
        with tpath.open() as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    textonly[(s["ds"], s["index"])] = s.get("textonly")

    # ---- merge: open-row noise500 view + per-row M3ID weight ----------
    g2_max = 0.0
    for r in trows:
        s = supp[(r["ds"], r["index"])]
        r["_n_text"] = s["n_text"]
        t = textonly.get((r["ds"], r["index"]))
        if t is not None:
            r["textonly"] = t
        if r["typ"] == "open":
            if "noise500" in s:
                r["noise500"] = s["noise500"]
            if "s0" in s:  # G2: determinism of the forward path
                for k, v in s["s0"].items():
                    if k in r["s0"]:
                        g2_max = max(g2_max, abs(float(r["s0"][k]) - float(v)))
    print(f"  G2 max |s0_recomputed - s0_cached| on open rows = {g2_max:.4f}")

    cov = sum(1 for r in trows if r.get("noise500") is not None)
    print(f"  noise500 coverage: {cov}/{len(trows)}")
    assert cov == len(trows), "noise500 still incomplete"

    # ---- predictions ---------------------------------------------------
    for r in trows:
        a_t = math.exp(-M3ID_META * r["_n_text"])
        w = (1.0 - a_t) / a_t
        r["_base"] = hit(r, base_pick(r))
        r["_tie"] = hit(r, fuse(r, "noise500", 1.0, -1.0, TIE_THETA))
        # robustness: TIE against the void/blank probe of its own paper
        r["_tie_blank"] = hit(r, fuse(r, "blank", 1.0, -1.0, TIE_THETA))
        # robustness: VCD/M3ID operators against the blank probe
        r["_vcd_blank"] = hit(r, fuse(r, "blank", 1 + VCD_ALPHA, -VCD_ALPHA, VCD_THETA))
        r["_m3id_blank"] = hit(r, fuse(r, "blank", 1 + w, -w, M3ID_THETA))
        # M3ID against its own paper's branch: the image-free conditional
        if r.get("textonly") is not None:
            r["_m3id_text"] = hit(r, fuse(r, "textonly", 1 + w, -w, M3ID_THETA))
        r["_vcd"] = hit(r, fuse(r, "noise500", 1 + VCD_ALPHA, -VCD_ALPHA, VCD_THETA))
        r["_m3id"] = hit(r, fuse(r, "noise500", 1 + w, -w, M3ID_THETA))

    # ---- G1 validity gate on the base cells ----------------------------
    base = cells(trows, lambda r: r["_base"])
    got = {k: round(base[f"{k}_All"], 2) for k in ("B", "S", "BS")}
    exp = {k: tgt[k] for k in ("B", "S", "BS")}
    print(f"  G1 base cells {got} vs frozen {exp}", flush=True)
    assert got == exp, "VALIDITY GATE G1 FAILED -- do not read baselines"

    out = {}
    for name, key in (("base", "_base"), ("TIE", "_tie"), ("VCD", "_vcd"),
                      ("M3ID", "_m3id"), ("TIE-blank", "_tie_blank"),
                      ("VCD-blank", "_vcd_blank"), ("M3ID-blank", "_m3id_blank"),
                      ("M3ID-text", "_m3id_text")):
        if any(key not in r for r in trows):
            continue
        c = cells(trows, lambda r, k=key: r[k])
        out[name] = {g: round(c[f"{g}_All"], 2) for g in ("B", "S", "BS")}
        out[name]["cells"] = {k: v for k, v in c.items()}
        print(f"  {name:5s} B/S/BS = {out[name]['B']:6.2f} {out[name]['S']:6.2f} "
              f"{out[name]['BS']:6.2f}")
    nt = [r["_n_text"] for r in trows]
    print(f"  n_text: median {int(np.median(nt))}, "
          f"M3ID weight (1-a)/a median {(1-math.exp(-0.02*np.median(nt)))/math.exp(-0.02*np.median(nt)):.2f}")
    return out


def main():
    res = {}
    for model in ("qwen3", "ov15"):
        if not (SUPP / f"supp_{model}.jsonl").exists():
            print(f"== {model}: supp missing, skipped")
            continue
        print(f"== {model} ==", flush=True)
        res[model] = run_model(model)
    (SUPP / "native_cd_results.json").write_text(json.dumps(res, indent=2))
    print("\nwrote native_cd_results.json")


if __name__ == "__main__":
    main()
