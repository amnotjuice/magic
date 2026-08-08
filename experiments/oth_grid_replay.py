"""Oth natural (t1,t2) joint-grid selection on Val -> Test readout.

Protocol (identical to old-config run 2026-07-19): grid t1 in {0,0.25,0.5,1.0}
x t2 in {0,0.25,0.5}, tau_V=0.625, selection rule = max pooled Others delta on
Val, true no-op semantics (unfired rows return original_pred), prefix-match
scoring. Zero GPU. Usage: python oth_grid_replay.py <tag> <val.jsonl> <test.jsonl>
"""
import json, re, sys

def open_hit(a, p):
    a, p = str(a).lower(), str(p).lower()
    if len(p) < len(a):
        return False
    if len(p) == len(a):
        return p == a
    return p[:len(a)] == a if not re.fullmatch(r"[A-Za-z]", p[len(a)]) else False

def margin(d):
    v = sorted(d.values(), reverse=True)
    return v[0] - v[1] if len(v) > 1 else 0.0

def replay(path, t1, t2, tauv=0.625):
    stats = {}
    for line in open(path):
        r = json.loads(line)
        base, tc, rv = r['base'], r['tc'], r['rv']
        st = base; fired = False
        if margin(st) <= t1:
            st = tc; fired = True
        if t2 > 0 and margin(st) <= t2:
            cand = {c: st[c] + rv[c] for c in st}
            if margin(cand) - margin(st) >= tauv:
                st = cand; fired = True
        pred = max(st, key=st.get) if fired else r['original_pred']
        d = stats.setdefault(r['dataset'], {'n': 0, 'o': 0, 'f': 0})
        d['n'] += 1; d['o'] += r['original_hit']
        d['f'] += int(open_hit(r['answer'], pred))
    out = {ds: round(100 * (v['f'] - v['o']) / v['n'], 2) for ds, v in stats.items()}
    n = sum(v['n'] for v in stats.values())
    dd = sum(v['f'] - v['o'] for v in stats.values())
    out['Others'] = round(100 * dd / n, 2)
    return out

if __name__ == "__main__":
    tag, val, test = sys.argv[1], sys.argv[2], sys.argv[3]
    best = None
    for t1 in (0.0, 0.25, 0.5, 1.0):
        for t2 in (0.0, 0.25, 0.5):
            r = replay(val, t1, t2)
            if best is None or r['Others'] > best[2]['Others']:
                best = (t1, t2, r)
    t1, t2, rv_ = best
    rt = replay(test, t1, t2)
    print(f"{tag}: selected (t1,t2)=({t1},{t2})  Val {rv_}  ->  Test {rt}")
