"""Null calibration: no attack; how often is each honest host flagged at nominal 1%?"""
import sys
src = open(sys.argv[1]).read().split('targets = [')[0]
sys.argv = [sys.argv[1], sys.argv[2]]
exec(src)
for f in (0.25, 1.0):
    per_host = []; any_host = []
    targets = [int(j) for j in np.argsort(-count)[:12]]
    for t in targets:
        for s in range(5):
            rng_seed = 777 + s * 1000 + t
            rng = random.Random(rng_seed); hosts = np.zeros(M, int)
            for k in range(M):
                hosts[k] = (0 if rng.random() < f else rng.randint(1, H - 1)) if X[k, t] != 0 else rng.randint(0, H - 1)
            zs = {h: host_scan(h, hosts, Y_true) for h in range(H)}
            K = sum(v[1] for v in zs.values()); thr = NormalDist().inv_cdf(1 - 0.01 / K)
            fl = [zs[h][0] > thr for h in range(H)]
            per_host += fl; any_host.append(any(fl))
    print(f'f={f}: per-host false flag {np.mean(per_host):.3f}, any host {np.mean(any_host):.3f} (nominal any-host 0.010), n={len(any_host)}', flush=True)
