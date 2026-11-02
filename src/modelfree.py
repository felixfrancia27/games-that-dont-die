"""Model-free two-sample test: player j's win rate on host h vs on all other hosts (no skill model, no epsilon)."""
import sys, math, random, json
import numpy as np
src = open('src/joint.py').read().split('w_clean =')[0]
sys.argv = ['x', 'src/srtd.py', 'data/matches.json']
exec(src)

def mf_flags(hosts, y, delta=0.01, min_n=5):
    tests = []
    for h in range(H):
        for j in range(N):
            inj = X[:, j] != 0
            won = np.where(X[:, j] > 0, y, 1 - y)
            a = inj & (hosts == h); b = inj & (hosts != h)
            n1, n2 = a.sum(), b.sum()
            if n1 < min_n: continue
            tests.append((h, j, n1, n2, won[a].mean() - (won[b].mean() if n2 else 0)))
    K = len(tests); flagged = set()
    for h, j, n1, n2, d in tests:
        if n2 == 0: continue                       # no overlap: the test abstains (Proposition 1)
        t = math.sqrt(math.log(K / delta) / 2 * (1 / n1 + 1 / n2))
        if d > t: flagged.add(h)
    return flagged

targets = [int(j) for j in np.argsort(-count)[:12]]
out = []
for f in (0.25, 0.5, 0.75, 0.9, 1.0):
    for lam in (0.0, 0.25, 0.5, 1.0):
        liar = honest = 0; n = 0
        for t in targets:
            for s in range(3):
                rng = random.Random(s * 1000 + t + 7)
                hosts = assign(t, f, rng); y = attack(hosts, t, lam, rng) if lam else Y_true.copy()
                fl = mf_flags(hosts, y); liar += 0 in fl; honest += bool(fl - {0}); n += 1
        row = dict(f=f, lam=lam, caught=round(liar / n, 3), honest_flagged=round(honest / n, 3))
        print(row, flush=True); out.append(row)
json.dump(out, open('results/modelfree.json', 'w'), indent=1)
