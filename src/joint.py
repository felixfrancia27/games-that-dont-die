"""Joint model: skills + per-(host, player) bias, fitted together (ridge-shrunk bias = honest-majority prior).
Compares against the leave-one-host-out (LOHO) test and against plain Elo, on the real match history."""
import sys, json, math, random
from statistics import NormalDist
import numpy as np
src = open(sys.argv[1]).read().split('H = 4')[0]
sys.argv = [sys.argv[1], sys.argv[2]]
exec(src)                                    # loads rows, X, Y_true, fit, predict, host_scan, elo_rank, count
H = 4

def bias_design(hosts, min_n=5):
    cols, keys = [], []
    for h in range(H):
        on = hosts == h
        for j in range(N):
            sel = on & (X[:, j] != 0)
            if sel.sum() >= min_n:
                c = np.where(sel, np.sign(X[:, j]), 0.0)   # +1 if j on team1, -1 if team2
                cols.append(c); keys.append((h, j))
    return np.array(cols).T, keys

def fit_joint(hosts, y, reg_s=0.3, lam1=1.5, iters=60):
    """L1 penalty on biases (via IRLS): most host-player biases go to ~0, a median-like gauge."""
    B, keys = bias_design(hosts)
    A = np.hstack([np.ones((M, 1)), X, B])
    P = A.shape[1]; nb = B.shape[1]
    w = np.zeros(P)
    for it in range(iters):
        b = w[1 + N:]
        Rb = lam1 / np.sqrt(b * b + 1e-4)
        R = np.concatenate([[1e-6], np.full(N, reg_s), Rb])
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + R * w
        Hm = A.T @ (A * (p * (1 - p))[:, None]) + np.diag(R)
        step = np.linalg.solve(Hm, g); w -= 0.7 * step
        if np.abs(step).max() < 1e-6: break
    p = 1 / (1 + np.exp(-A @ w))
    Hl = A.T @ (A * (p * (1 - p))[:, None]) + np.diag(np.concatenate([[1e-6], np.full(N, reg_s), np.full(nb, 1e-3)]))
    se = np.sqrt(np.diag(np.linalg.inv(Hl))[1 + N:])
    return w[1:1 + N], w[1 + N:] / se, keys

def rank_of(skill, target):
    elig = count >= 10
    return int((skill[elig] > skill[target]).sum()) + 1

def assign(target, f, rng):
    hosts = np.zeros(M, int)
    for k in range(M):
        hosts[k] = (0 if rng.random() < f else rng.randint(1, H - 1)) if X[k, target] != 0 else rng.randint(0, H - 1)
    return hosts

def attack(hosts, target, lam, rng):
    y = Y_true.copy()
    for k in range(M):
        if hosts[k] == 0 and X[k, target] != 0:
            won = (y[k] == 1) == (X[k, target] > 0)
            if not won and rng.random() < lam: y[k] = 1 - y[k]
    return y

w_clean = fit(np.ones(M, bool), Y_true)[1:]
targets = [int(j) for j in np.argsort(-count)[:12]]
out = []
for f in (0.25, 0.5, 1.0):
    for lam in (0.0, 0.5, 1.0):
        st = dict(f=f, lam=lam, n=0, joint_liar=0, joint_honest=0, loho_liar=0, loho_honest=0,
                  rank_gain_elo=[], rank_gain_bt=[], rank_gain_joint=[])
        for t in targets:
            for s in range(3):
                rng = random.Random(s * 1000 + t + 7)
                hosts = assign(t, f, rng); y = attack(hosts, t, lam, rng) if lam > 0 else Y_true.copy()
                skill_j, z, keys = fit_joint(hosts, y)
                thr = NormalDist().inv_cdf(1 - 0.01 / len(keys))
                flagged = {keys[i][0] for i in range(len(keys)) if z[i] > thr}
                zs = {h: host_scan(h, hosts, y) for h in range(H)}
                thr2 = NormalDist().inv_cdf(1 - 0.01 / sum(v[1] for v in zs.values()))
                lf = {h for h in range(H) if zs[h][0] > thr2}
                st['joint_liar'] += 0 in flagged; st['joint_honest'] += bool(flagged - {0})
                st['loho_liar'] += 0 in lf; st['loho_honest'] += bool(lf - {0})
                skill_bt = fit(np.ones(M, bool), y)[1:]
                st['rank_gain_elo'].append(elo_rank(Y_true, t) - elo_rank(y, t))
                st['rank_gain_bt'].append(rank_of(w_clean, t) - rank_of(skill_bt, t))
                st['rank_gain_joint'].append(rank_of(w_clean, t) - rank_of(skill_j, t))
                st['n'] += 1
        n = st['n']
        row = {k: (round(v / n, 3) if isinstance(v, int) and k not in ('n',) else v) for k, v in st.items()}
        for k in ('rank_gain_elo', 'rank_gain_bt', 'rank_gain_joint'): row[k] = round(float(np.mean(st[k])), 2)
        row['f'] = f; row['lam'] = lam
        print(row, flush=True); out.append(row)
json.dump(out, open('results/joint.json', 'w'), indent=1)
