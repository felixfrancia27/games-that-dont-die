"""Sybil farm on top of the real history: trust via personalized PageRank on the player-match graph,
seeded on the real (bootstrap) players. Measures farm token share and rank boost, plus the Prop. 7 bound."""
import json, random, math, sys
import numpy as np
real = [r for r in json.load(open('data/matches.json')) if r['winner'] in (1, 2) and r['team1'] and r['team2']]
real.sort(key=lambda r: r['order'])
honest_players = sorted({p for r in real for p in r['team1'] + r['team2']})
ALPHA = 0.15

def build(rows, players):
    idx = {p: i for i, p in enumerate(players)}
    N, M = len(players), len(rows)
    X = np.zeros((M, N)); y = np.zeros(M)
    for k, r in enumerate(rows):
        for p in r['team1']: X[k, idx[p]] += 1 / len(r['team1'])
        for p in r['team2']: X[k, idx[p]] -= 1 / len(r['team2'])
        y[k] = 1.0 if r['winner'] == 1 else 0.0
    return X, y, idx

def fit_w(X, y, wts, reg=0.3, iters=30):
    M, N = X.shape; A = np.hstack([np.ones((M, 1)), X]); w = np.zeros(N + 1)
    R = np.eye(N + 1) * reg; R[0, 0] = 1e-6
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (wts * (p - y)) + R @ w
        Hm = A.T @ (A * (wts * p * (1 - p))[:, None]) + R
        w -= np.linalg.solve(Hm, g)
    return w

def ppr(X, seeds_mask):
    """PPR on bipartite player-match graph. Returns trust per player node and per match node."""
    B = (X != 0).astype(float)                     # M x N incidence
    dp = B.sum(0); dm = B.sum(1)                   # degrees
    s_p = seeds_mask / seeds_mask.sum(); s_m = np.zeros(B.shape[0])
    p_p, p_m = s_p.copy(), s_m.copy()
    for _ in range(200):
        new_m = (1 - ALPHA) * (B @ (p_p / np.maximum(dp, 1)))
        new_p = ALPHA * s_p + (1 - ALPHA) * (B.T @ (p_m / np.maximum(dm, 1)))
        p_p, p_m = new_p, new_m
    return p_p, p_m, dp, dm, B

def H2(p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def run(attack_matches, farm_matches=300, n_syb=12, mode='tokens', seed=0):
    rng = random.Random(seed)
    syb = [f'syb{i}' for i in range(n_syb)]
    rows = [dict(r) for r in real]
    # attack edges: sybils sneak into real matches (replace one real participant)
    for k in rng.sample(range(len(rows)), attack_matches):
        r = rows[k]; side = rng.choice(['team1', 'team2'])
        team = list(r[side]); team[rng.randrange(len(team))] = rng.choice(syb); r[side] = team
    target = syb[0]
    for _ in range(farm_matches):
        pool = syb[:]; rng.shuffle(pool); t1, t2 = pool[:6], pool[6:12]
        if mode == 'rank':
            if target not in t1 and target not in t2: t1[0] = target
            win = 1 if target in t1 else 2
        else:
            win = rng.choice([1, 2])
        rows.append(dict(team1=t1, team2=t2, winner=win))
    players = honest_players + syb
    X, y, idx = build(rows, players)
    seeds = np.array([1.0 if p in set(honest_players) else 0.0 for p in players])
    p_p, p_m, dp, dm, B = ppr(X, seeds)
    tp = p_p / np.maximum(dp, 1)                                      # degree-normalised player trust
    tau = np.array([tp[B[m] > 0].min() for m in range(len(rows))])
    honest_m = np.arange(len(real)); tau = np.minimum(tau / np.median(tau[honest_m]), 1.0)
    is_farm = np.zeros(len(rows), bool); is_farm[len(real):] = True
    syb_idx = np.array([idx[s] for s in syb])
    # Prop 7 bound on trust mass in the sybil region (sybil players + farm matches)
    in_S_p = np.zeros(len(players), bool); in_S_p[syb_idx] = True
    mass_S = p_p[in_S_p].sum() + p_m[is_farm].sum()
    flow_in = 0.0
    for m in range(len(rows)):                                       # honest match -> sybil player edges
        if not is_farm[m]:
            for i in np.where(B[m] > 0)[0]:
                if in_S_p[i]: flow_in += p_m[m] / dm[m]
    for i in np.where(~in_S_p)[0]:                                   # honest player -> farm match edges
        flow_in += p_p[i] / dp[i] * B[is_farm, i].sum()
    bound = (1 - ALPHA) / ALPHA * flow_in
    out = dict(attack=attack_matches, mass_S=mass_S, bound=bound)
    if mode == 'tokens':
        w = fit_w(X, y, np.ones(len(rows)))
        phat = 1 / (1 + np.exp(-(w[0] + X @ w[1:])))
        for name, wt in (('equal', np.ones(len(rows))), ('entropy', H2(phat)), ('trust_entropy', tau * H2(phat))):
            out['share_' + name] = float(wt[is_farm].sum() / wt.sum())
        out['honest_low_trust'] = float((tau[honest_m] < 0.5).mean())
    else:
        def rank(w):
            sk = w[1:]; cnt = (X != 0).sum(0); elig = (cnt >= 10) & ~in_S_p; elig[idx[target]] = True
            return int((sk[elig] > sk[idx[target]]).sum()) + 1
        R = np.full(len(players), 1500.0)
        for k in range(len(rows)):
            t1 = X[k] > 0; t2 = X[k] < 0
            e = 1 / (1 + 10 ** ((R[t2].mean() - R[t1].mean()) / 400)); R[t1] += 32 * (y[k] - e); R[t2] -= 32 * (y[k] - e)
        cnt = (X != 0).sum(0); elig = (cnt >= 10) & ~in_S_p
        out['rank_elo'] = int((R[elig] > R[idx[target]]).sum()) + 1
        out['rank_bt'] = rank(fit_w(X, y, np.ones(len(rows))))
        out['rank_trust_bt'] = rank(fit_w(X, y, tau))
        rel = tp / np.median(tp[~in_S_p & (dp >= 10)])
        out['target_trust'] = float(rel[idx[target]])
        out['sybils_eligible'] = int((rel[syb_idx] >= 0.5).sum())
        out['honest_ineligible'] = float((rel[~in_S_p & (dp >= 10)] < 0.5).mean())
    return out

res = []
for mode in (sys.argv[1:] or ['tokens', 'rank']):
    for a in (0, 10, 50, 200):
        rs = [run(a, mode=mode, seed=s) for s in range(3)]
        row = {'mode': mode, 'attack_matches': a}
        for k in rs[0]:
            if k != 'attack': row[k] = round(float(np.mean([r[k] for r in rs])), 4)
        print(row, flush=True); res.append(row)
json.dump(res, open('results/sybil_' + '_'.join(sys.argv[1:] or ['all']) + '.json', 'w'), indent=1)
