"""Average passes per ablation config on the BS (biased) subset, both
backbones, MCQ+Others combined. Zero-GPU replay from the frozen bundles;
accuracy cells are the published ablation numbers, this only accounts cost."""
import gzip, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magic import margin, LADDER_MCQ, LADDER_OTH, GROUPS

HERE = Path(__file__).resolve().parents[1]
with gzip.open(HERE / "magic_mcq_bundle.json.gz", "rt") as f:
    B = json.load(f)
with gzip.open(HERE / "magic_oth_bundle.json.gz", "rt") as f:
    OTH = json.load(f)

BS = GROUPS["BS"]

def s0_margin_mcq(row):
    return margin(row["s0"])

def state_after_r1_mcq(row):
    s0, dd, ad = row["s0"], row["default_delta"], row["answer_delta"]
    return [max(dd[i], ad[i]) for i in range(len(s0))]

def s0_margin_oth(row):
    return margin(row["real_def"])

def state_after_r1_oth(row):
    rd, ra, bd, ba = row["real_def"], row["real_ans"], row["blank_def"], row["blank_ans"]
    pool = row["pool"]
    l1 = {c: rd[c] - bd[c] for c in pool}
    l2 = {c: ra[c] - ba[c] for c in pool}
    return {c: max(l1[c], l2[c]) for c in pool}

def fired_flags(m_s0, m_r1, cfg, config):
    # config: 'base','r1','r2only','always','magic'
    if config == "base":
        return (False, False)
    if config == "always":
        return (True, True)
    if config == "r1":
        return (m_s0 <= cfg.theta1, False)
    if config == "r2only":
        # rung 1 removed, so the rung-2 gate reads the base margin
        return (False, m_s0 <= cfg.rung2.theta)
    # magic
    f1 = m_s0 <= cfg.theta1
    m = m_r1 if f1 else m_s0
    f2 = m <= cfg.rung2.theta
    return (f1, f2)

def deploy_cost(f1, f2, is_mcq, is_binary, is_qwen):
    """Shared-prefill DEPLOY accounting, matching experiments/data/cost_profile (measurement notes).
    Qwen ViLP buys a 3-generation pool (default/answer_format double as the two
    scoring views, so rung1 buys only 2 blanks); open-ended rung2 uses a single
    visual source (cost 1); MCQ/binary are format-blind (deploy0=1, rung=3)."""
    open_ended = (not is_mcq) and (not is_binary)
    qwen_vilp = open_ended and is_qwen
    deploy0 = 3 if qwen_vilp else 1
    rung1_cost = 2 if qwen_vilp else 3
    rung2_cost = 1 if open_ended else 3
    p = deploy0
    if f1:
        p += rung1_cost
    if f2:
        p += rung2_cost
    return p

# Passes are reported on the MCQ-BS subset (the population the cost claim
# uses); MCQ carries no shared-prefill discount, so deploy == 1+3f1+3f2.
for mk in ("qwen", "llava"):
    is_qwen = (mk == "qwen")
    for config in ("base", "r1", "r2only", "always", "magic"):
        tot = n = 0
        for row in B["mcq"][mk]:
            if row["mode"] in BS:
                f1, f2 = fired_flags(s0_margin_mcq(row), margin(state_after_r1_mcq(row)),
                                     LADDER_MCQ[mk], config)
                tot += deploy_cost(f1, f2, True, False, is_qwen); n += 1
        print(f"{mk:6} {config:7} avg_passes={tot/n:.2f}  (n={n})")
