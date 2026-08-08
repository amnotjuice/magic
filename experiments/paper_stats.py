"""Paper-form (released magic.py) reconciliation statistics. Zero GPU.

Candidate  = the released ladder exactly as magic.py ships it: allocation
             (t1, t2) + rung-2 commitment tau_V; NO rung-1 admission.
Protocol   = pure replay of frozen artifacts (magic_*_bundle.json.gz, the
             compliant natural Oth caches, the POPE caches). No parameter
             is selected here; every constant is magic.py's shipped value.
Outputs    = descriptive statistics for the paper prose: 9-cell parity,
             row-bootstrap CIs vs the published SCI5/SCI7 BS-Overall rows,
             rung-arm ablations, pass/wall-clock accounting, natural MCQ
             McNemar flips, natural Oth parity + flips, POPE transfer arms.
Acceptance = none (writing-phase transcription); adverse readouts are
             reported as-is.

Usage: python3 experiments/paper_stats.py  (stdlib only)
"""
import gzip
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import magic as M  # noqa: E402
from oth_grid_replay import replay as oth_replay, open_hit, margin as oth_margin  # noqa: E402

OUT = ROOT / "experiments/data/paper_reconciliation"
OUT.mkdir(exist_ok=True)

PUB = {  # published SCI BS-Overall rows (paper Tables 2/9)
    "qwen": {"SCI5": 29.50, "SCI7": 31.72},
    "llava": {"SCI5": 34.19, "SCI7": 34.92},
}


def run_flags(state0, lift, residual, cfg):
    state, f1, f2 = state0, False, False
    if M.margin(state) <= cfg.theta1:
        state, f1 = lift, True
    if M.margin(state) <= cfg.rung2.theta:
        f2 = True
        cand = M._add(state, residual)
        if M.margin(cand) - M.margin(state) >= cfg.rung2.tau:
            state = cand
    return state, f1, f2


def mcq_row(row, cfg, arm="full"):
    s0, dd, ad, vc = row["s0"], row["default_delta"], row["answer_delta"], row["selected_vc"]
    lift = [max(dd[i], ad[i]) for i in range(len(s0))]
    residual = [s0[i] - vc[i] for i in range(len(s0))]
    st, f1, f2 = arm_run(s0, s0, lift, residual, cfg, arm)
    return row["labels"][M.argmax_key(st)], 1 + 3 * f1 + 3 * f2


def oth_row(row, cfg, arm="full"):
    pool = row["pool"]
    rd, ra, bd, ba, vg = (row["real_def"], row["real_ans"], row["blank_def"],
                          row["blank_ans"], row["vc_gray"])
    base = {c: max(rd[c], ra[c]) for c in pool}
    lift = {c: max(rd[c] - bd[c], ra[c] - ba[c]) for c in pool}
    residual = {c: base[c] - vg[c] for c in pool}
    st, f1, f2 = arm_run(dict(rd), base, lift, residual, cfg, arm)
    return M.argmax_key(st), 2 + 2 * f1 + 1 * f2


def arm_run(state0, base, lift, residual, cfg, arm):
    """arm in {base, r1, r2, full, ungated}; full == released magic.py
    control flow; ungated == both rungs always allocated (commitment
    kept), the allocation-deleted arm."""
    if arm == "base":
        return state0, False, False
    if arm == "ungated":
        state = lift
        cand = M._add(state, residual)
        if M.margin(cand) - M.margin(state) >= cfg.rung2.tau:
            state = cand
        return state, True, True
    if arm == "r1":
        if M.margin(state0) <= cfg.theta1:
            return lift, True, False
        return state0, False, False
    if arm == "r2":
        state, f2 = state0, False
        if M.margin(state) <= cfg.rung2.theta:
            f2 = True
            cand = M._add(state, residual)
            if M.margin(cand) - M.margin(state) >= cfg.rung2.tau:
                state = cand
        return state, False, f2
    return run_flags(state0, lift, residual, cfg)


def mcnemar_p(a, b):
    """Two-sided exact binomial on discordant pair counts."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * 2 / 2 ** n
    return min(1.0, p)


def bootstrap(outcomes, published, iters=100000, seed=0):
    rng = random.Random(seed)
    n = len(outcomes)
    accs = []
    le = {k: 0 for k in published}
    for _ in range(iters):
        s = sum(outcomes[rng.randrange(n)] for _ in range(n))
        acc = 100.0 * s / n
        accs.append(acc)
        for k, v in published.items():
            le[k] += acc <= v
    accs.sort()
    ci = (round(accs[int(0.025 * iters)], 2), round(accs[int(0.975 * iters)], 2))
    return ci, {k: round(v / iters, 6) for k, v in le.items()}


def main():
    with gzip.open(ROOT / "magic_mcq_bundle.json.gz", "rt") as f:
        mcq_b = json.load(f)
    with gzip.open(ROOT / "magic_oth_bundle.json.gz", "rt") as f:
        oth_b = json.load(f)
    wc = json.loads((ROOT / "experiments/data/cost_profile/wallclock.json").read_text())
    ledger = {}

    for mk in ("qwen", "llava"):
        led = ledger[mk] = {}
        cfg_m, cfg_o = M.LADDER_MCQ[mk], M.LADDER_OTH[mk]

        # ---- biased replay: 9-cell + per-row BS outcomes + passes ----
        cells = {arm: {g: {"mcq": [], "oth": []} for g in M.GROUPS} for arm in ("base", "r1", "r2", "full", "ungated")}
        bs_outcomes, passes = [], {"mcq": [], "oth": []}
        exit_hist = {"mcq": {}, "oth": {}}
        for fmt, rows, fn, cfg in (("mcq", mcq_b["mcq"][mk], mcq_row, cfg_m),
                                   ("oth", oth_b[mk], oth_row, cfg_o)):
            for row in rows:
                gold = str(row["answer"]).strip().upper() if fmt == "mcq" else row["answer"]
                for arm in cells:
                    p, np_ = fn(row, cfg, arm)
                    ok = (p == gold) if fmt == "mcq" else M._open_hit(gold, p)
                    for g, modes in M.GROUPS.items():
                        if row["mode"] in modes:
                            cells[arm][g][fmt].append(ok)
                    if arm == "full":
                        passes[fmt].append(np_)
                        exit_hist[fmt][np_] = exit_hist[fmt].get(np_, 0) + 1
                        if row["mode"] in M.GROUPS["BS"]:
                            bs_outcomes.append(int(ok))

        def cell_table(arm):
            t = {}
            for g in M.GROUPS:
                v_m, v_o = cells[arm][g]["mcq"], cells[arm][g]["oth"]
                t[g] = {
                    "MCQ": round(100 * sum(v_m) / len(v_m), 2),
                    "Oth": round(100 * sum(v_o) / len(v_o), 2),
                    "All": round(100 * (sum(v_m) + sum(v_o)) / (len(v_m) + len(v_o)), 2),
                }
            return t

        led["cells"] = {arm: cell_table(arm) for arm in cells}
        led["passes"] = {f: round(sum(v) / len(v), 2) for f, v in passes.items()}
        led["exit_hist"] = exit_hist
        n_m, n_o = len(passes["mcq"]), len(passes["oth"])
        ts, tg = wc[mk]["t_score_s"], wc[mk]["t_gen_s"]
        # open-ended base pass is a generation whose prefill doubles as the
        # first scoring pass; every other pass is a scoring pass.
        magic_s = (n_m * led["passes"]["mcq"] * ts
                  + n_o * ((led["passes"]["oth"] - 1) * ts + tg)) / (n_m + n_o)
        led["seconds"] = {"magic": round(magic_s, 2),
                          "SCI5": round(5 * ts, 2), "SCI7": round(7 * ts, 2),
                          "ratio_SCI7": round(magic_s / (7 * ts), 2),
                          "ratio_SCI5": round(magic_s / (5 * ts), 2)}

        # ---- bootstrap CI on BS Overall vs published rows ----
        ci, ple = bootstrap(bs_outcomes, PUB[mk])
        led["bs_overall_ci"] = {"n": len(bs_outcomes), "ci95": ci, "P_le": ple}

        # ---- natural MCQ, allocation-deleted arm (always fire, commit) ----
        cfg_n = M.LADDER_NAT[mk]
        uw = ur = 0
        for row in mcq_b["natural"][mk]:
            state = row["clean2"]
            cand = M._add(state, row["vc_delta"])
            if M.margin(cand) - M.margin(state) >= cfg_n.rung2.tau:
                state = cand
            b_ok = row["labels"][M.argmax_key(row["base"])] == row["answer"]
            f_ok = row["labels"][M.argmax_key(state)] == row["answer"]
            uw += (not b_ok) and f_ok
            ur += b_ok and (not f_ok)
        n_nat0 = len(mcq_b["natural"][mk])
        led["natural_mcq_ungated"] = {
            "delta": round(100 * (uw - ur) / n_nat0, 4), "w2r": uw, "r2w": ur,
            "p": mcnemar_p(uw, ur)}

        w2r = r2w = 0
        n_nat = 0
        nat_passes = 0
        nat_hist = {}
        for row in mcq_b["natural"][mk]:
            state, p = row["base"], 1
            if M.margin(state) <= cfg_n.theta1:
                p += 3
                state = row["clean2"]
            if M.margin(state) <= cfg_n.rung2.theta:
                p += 3
                cand = M._add(state, row["vc_delta"])
                if M.margin(cand) - M.margin(state) >= cfg_n.rung2.tau:
                    state = cand
            n_nat += 1
            nat_passes += p
            nat_hist[p] = nat_hist.get(p, 0) + 1
            b_ok = row["labels"][M.argmax_key(row["base"])] == row["answer"]
            f_ok = row["labels"][M.argmax_key(state)] == row["answer"]
            w2r += (not b_ok) and f_ok
            r2w += b_ok and (not f_ok)
        led["natural_mcq"] = {
            "n": n_nat, "delta": round(100 * (w2r - r2w) / n_nat, 4),
            "w2r": w2r, "r2w": r2w, "p": mcnemar_p(w2r, r2w),
            "avg_passes": round(nat_passes / n_nat, 3),
            "exit_hist": dict(sorted(nat_hist.items()))}

        # ---- natural Oth parity + flips at the frozen (t1, t2) ----
        oth_cfg = {"qwen": (0.0, 0.5), "llava": (0.0, 0.0)}[mk]
        test_path = {
            "qwen": ROOT / "experiments/data/natural_open_ended/qwen_natural_oth_compliant_test.jsonl",
            "llava": ROOT / "experiments/data/natural_open_ended/llava_natural_oth_compliant_test_v2.jsonl",
        }[mk]
        led["natural_oth"] = oth_replay(str(test_path), *oth_cfg)
        w2r = r2w = n_r = 0
        for line in open(test_path):
            r = json.loads(line)
            st, fired = r["base"], False
            if oth_margin(st) <= oth_cfg[0]:
                st, fired = r["tc"], True
            if oth_cfg[1] > 0 and oth_margin(st) <= oth_cfg[1]:
                cand = {c: st[c] + r["rv"][c] for c in st}
                if oth_margin(cand) - oth_margin(st) >= 0.625:
                    st, fired = cand, True
            pred = max(st, key=st.get) if fired else r["original_pred"]
            b_ok, f_ok = bool(r["original_hit"]), open_hit(r["answer"], pred)
            w2r += (not b_ok) and f_ok
            r2w += b_ok and (not f_ok)
            n_r += 1
        led["natural_oth_flips"] = {"n": n_r, "w2r": w2r, "r2w": r2w,
                                    "p": round(mcnemar_p(w2r, r2w), 4)}

        # ---- POPE transfer, paper form ----
        pope_path = {"qwen": ROOT / "experiments/data/pope_external_probe/pope_qwen2.jsonl",
                     "llava": ROOT / "experiments/data/pope_external_probe/pope_llava.jsonl"}[mk]
        if pope_path.exists():
            recs = {}
            for line in open(pope_path):
                if line.strip():
                    r = json.loads(line)
                    recs[r["qid"]] = r  # dedup checkpoint re-runs by qid
            bh = mh = w2r = r2w = n_p = 0
            for r in recs.values():
                s0, bl = r["s0"], r["blank"]
                ar, ab = r["ar"], r["ab"]
                base_env = [max(s0[i], ar[i]) for i in (0, 1)]
                lift = [max(s0[i] - bl[i], ar[i] - ab[i]) for i in (0, 1)]
                residual = [base_env[i] - r["gray"][i] for i in (0, 1)]
                st, _, _ = run_flags(s0, lift, residual, cfg_o)
                gold = 0 if r["answer"] == "yes" else 1
                b_ok = M.argmax_key(s0) == gold
                f_ok = M.argmax_key(st) == gold
                bh += b_ok
                mh += f_ok
                w2r += (not b_ok) and f_ok
                r2w += b_ok and (not f_ok)
                n_p += 1
            led["pope"] = {
                "n": n_p, "base": round(100 * bh / n_p, 2),
                "magic": round(100 * mh / n_p, 2),
                "delta": round(100 * (mh - bh) / n_p, 2),
                "w2r": w2r, "r2w": r2w, "p": round(mcnemar_p(w2r, r2w), 4)}

    (OUT / "ledger.json").write_text(json.dumps(ledger, indent=2))
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
