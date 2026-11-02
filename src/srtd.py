"""Single-reporter truth discovery: semi-synthetic test on real Warband Premier matches."""
import json, sys, random, math
from statistics import NormalDist
import numpy as np

rows = [r for r in json.load(open(sys.argv[1])) if r['winner'] in (1, 2) and r['team1'] and r['team2']]
rows.sort(key=lambda r: r['order'])
players = sorted({p for r in rows for p in r['team1'] + r['team2']})
idx = {p: i for i, p in enumerate(players)}
N, M = len(players), len(rows)
X = np.zeros((M, N))
for k, r in enumerate(rows):
    for p in r['team1']: X[k, idx[p]] += 1 / len(r['team1'])
    for p in r['team2']: X[k, idx[p]] -= 1 / len(r['team2'])
Y_true = np.array([1.0 if r['winner'] == 1 else 0.0 for r in rows])
count = X.astype(bool).sum(0)

def fit(mask, y, reg=0.3, iters=30):
    A = np.hstack([np.ones((mask.sum(), 1)), X[mask]]); yy = y[mask]
    w = np.zeros(N + 1); R = np.eye(N + 1) * reg; R[0, 0] = 1e-6
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - yy) + R @ w
        Hm = A.T @ (A * (p * (1 - p))[:, None]) + R
        w -= np.linalg.solve(Hm, g)
    return w

def predict(w, m):
    return 1 / (1 + np.exp(-(w[0] + X[m] @ w[1:])))

def host_scan(host, hosts, y, min_n=5, exclude=()):
    """max z over players for one host, skill fitted leaving that host (and excluded hosts) out."""
    keep = hosts != host
    for e in exclude: keep &= hosts != e
    w = fit(keep, y)
    on = np.where(hosts == host)[0]
    p1 = predict(w, on)
    best = -9; tests = 0
    for j in range(N):
        ms = on[X[on, j] != 0]
        if len(ms) < min_n: continue
        side = np.sign(X[ms, j])                       # +1 team1, -1 team2
        yj = np.where(side > 0, y[ms], 1 - y[ms])       # j's team reported win
        pj = np.where(side > 0, p1[X[on, j] != 0], 1 - p1[X[on, j] != 0])
        z = (yj - pj).sum() / math.sqrt((pj * (1 - pj)).sum() + 1e-9)
        best = max(best, z); tests += 1
    return best, tests

def elo_rank(y, target):
    R = np.full(N, 1500.0)
    for k in range(M):
        t1 = X[k] > 0; t2 = X[k] < 0
        e = 1 / (1 + 10 ** ((R[t2].mean() - R[t1].mean()) / 400))
        R[t1] += 32 * (y[k] - e); R[t2] -= 32 * (y[k] - e)
    elig = count >= 10
    return int((R[elig] > R[target]).sum()) + 1

H = 4
def trial(target, f, lam, seed, attack=True):
    rng = random.Random(seed)
    hosts = np.zeros(M, int); y = Y_true.copy()
    for k in range(M):
        if X[k, target] != 0:
            hosts[k] = 0 if rng.random() < f else rng.randint(1, H - 1)
        else:
            hosts[k] = rng.randint(0, H - 1)
        if attack and hosts[k] == 0 and X[k, target] != 0:
            won = (y[k] == 1) == (X[k, target] > 0)
            if not won and rng.random() < lam:
                y[k] = 1 - y[k]
    zs = {h: host_scan(h, hosts, y) for h in range(H)}
    K = sum(t for _, t in zs.values())
    thr = NormalDist().inv_cdf(1 - 0.01 / max(K, 1))
    flag1 = {h for h in range(H) if zs[h][0] > thr}
    flag = set(flag1)
    for _ in range(3):  # iterative exclusion: refit without currently flagged hosts
        z2 = {h: host_scan(h, hosts, y, exclude=flag - {h})[0] for h in range(H)}
        new = {h for h in range(H) if z2[h] > thr}
        if new == flag: break
        flag = new
    return (0 in flag1, bool(flag1 - {0}), 0 in flag, bool(flag - {0}),
            elo_rank(y, target), elo_rank(Y_true, target))

targets = [int(j) for j in np.argsort(-count)[:12]]
print(f"data: {M} matches, {N} players; targets have {sorted(count[targets].tolist())} matches")
res = []
for f in (0.25, 0.5, 0.75, 0.9, 1.0):
    for lam in (0.25, 0.5, 1.0):
        acc = np.zeros(6); gain = []; n = 0
        for t in targets:
            for s in range(3):
                a = trial(t, f, lam, s * 1000 + t)
                b = trial(t, f, lam, s * 1000 + t, attack=False)
                acc += [a[0], a[1], a[2], a[3], b[0], b[2]]; gain.append(a[5] - a[4]); n += 1
        acc /= n
        row = dict(f=f, lam=lam, detect=acc[0], fp_honest=acc[1], detect_iter=acc[2], fp_honest_iter=acc[3],
                   false_flag_no_attack=acc[4], false_flag_no_attack_iter=acc[5], elo_rank_gain=float(np.mean(gain)))
        row = {k: round(float(v), 3) for k, v in row.items()}
        res.append(row); print(row, flush=True)
json.dump(res, open(sys.argv[2], 'w'), indent=1)
