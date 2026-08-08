"""Number-word normalization for fair open-ended scoring on new models
(ov15 answers 'four' where gold is '4'; prefix scorer miskilled it)."""
W2N = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
N2W = {v: k for k, v in W2N.items()}

def variants(s):
    s = str(s).strip().lower()
    out = {s}
    first = s.split()[0].rstrip(".,;") if s.split() else s
    if first in W2N:
        out.add(W2N[first] + s[len(first):] if s.startswith(first) else s)
        out.add(W2N[first])
    if first in N2W:
        out.add(N2W[first])
    return out

def norm_hit(answer, prediction, base_hit):
    """base_hit = the raw prefix scorer; accept if ANY variant pair hits."""
    for a in variants(answer):
        for p in variants(prediction):
            if base_hit(a, p):
                return 1
    return 0

import re
ART = re.compile(r"^(the|a|an)\s+")
def canon(s):
    s = str(s).strip().lower()
    s = ART.sub("", s)
    w = s.split()
    if w and w[0] in W2N: w[0] = W2N[w[0]]
    s = " ".join(w)
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"): s = s[:-1]
    return s

def fair_hit(answer, prediction):
    from open_ended_scoring import paper_hit
    a, p = canon(answer), canon(prediction)
    if paper_hit(a, p): return 1
    if len(p.split()) <= 4 and re.search(rf"\b{re.escape(a)}\b", p): return 1
    return 0

def fair_eq(x, y):
    """branch-consistency comparison under the same canonicalization."""
    a, b = canon(x), canon(y)
    L = min(len(a), len(b))
    return a[:L] == b[:L] if L else a == b
