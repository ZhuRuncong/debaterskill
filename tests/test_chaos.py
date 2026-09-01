"""The p_chaos robust mixture: evidence and update damping."""
import math
import random

from debaterskill import Tab, Debate, Team

rng = random.Random(11)
N = 40
SKILL = {f'p{i}': rng.gauss(0, 2) for i in range(N)}
TEAMS = [(f'p{2*i}', f'p{2*i+1}') for i in range(N // 2)]
TRUE_CHAOS = 0.25


def rooms(n=400, chaos=TRUE_CHAOS):
    out = []
    for k in range(n):
        picks = rng.sample(TEAMS, 4)
        if rng.random() < chaos:
            order = list(range(4))
            rng.shuffle(order)
        else:
            perf = [sum(SKILL[s] + rng.gauss(0, 1) for s in t) for t in picks]
            order = sorted(range(4), key=lambda i: -perf[i])
        pts = [0] * 4
        for rank, i in enumerate(order):
            pts[i] = 3 - rank
        out.append(Debate([Team(speakers=list(t), points=p)
                           for t, p in zip(picks, pts)], day=k // 20))
    return out


def corr(mus):
    ks = list(mus)
    xs = [SKILL[k] for k in ks]
    ys = [mus[k] for k in ks]
    n = len(ks)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def fit(rs, p_chaos):
    tab = Tab(gamma=0.0, within=None, motions=False, p_chaos=p_chaos)
    for d in rs:
        tab.add(d)
    tab.fit(iterations=30, epsilon=1e-8)
    return tab


def test_zero_is_identical():
    rs = rooms(120)
    a = fit(rs, 0.0).speakers()
    b = fit(rs, 0.0).speakers()
    assert all(abs(a[k][1].mu - b[k][1].mu) < 1e-12 for k in a)


def test_chaos_helps_when_data_is_chaotic():
    rs = rooms(400, chaos=TRUE_CHAOS)
    base = corr({k: v[1].mu for k, v in fit(rs, 0.0).speakers().items()})
    got = {p: corr({k: v[1].mu for k, v in fit(rs, p).speakers().items()})
           for p in (0.1, 0.25, 0.4)}
    best = max(got, key=got.get)
    assert got[best] >= base - 1e-9


def test_chaos_costs_little_when_data_is_clean():
    rs = rooms(400, chaos=0.0)
    base = corr({k: v[1].mu for k, v in fit(rs, 0.0).speakers().items()})
    on = corr({k: v[1].mu for k, v in fit(rs, 0.25).speakers().items()})
    assert on > base - 0.05
