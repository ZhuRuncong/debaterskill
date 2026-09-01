"""fit_judges recovers which chairs are noisy and downweights their rooms."""
import math
import random

from debaterskill import Tab, Debate, Team, Judge

rng = random.Random(5)
N_SPK = 32
SKILL = {f'p{i}': rng.gauss(0, 2) for i in range(N_SPK)}
TEAMS = [(f'p{2 * i}', f'p{2 * i + 1}') for i in range(N_SPK // 2)]
CHAIR_U = {'sharp chair': 0.0, 'blind chair': 9.0}


def make_rooms(n=300):
    rooms = []
    for k in range(n):
        picks = rng.sample(TEAMS, 4)
        chair = 'sharp chair' if k % 2 == 0 else 'blind chair'
        u = CHAIR_U[chair]
        perf = [sum(SKILL[s] + rng.gauss(0, 1) for s in t) + rng.gauss(0, math.sqrt(u))
                for t in picks]
        order = sorted(range(4), key=lambda i: -perf[i])
        points = [0] * 4
        for rank, i in enumerate(order):
            points[i] = 3 - rank
        rooms.append(Debate(
            [Team(speakers=list(t), points=p) for t, p in zip(picks, points)],
            day=k // 10, chair=Judge(chair)))
    return rooms


def corr(d):
    xs = [SKILL[k] for k in d]
    ys = [d[k] for k in d]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def mus(tab):
    return {k: v[1].mu for k, v in tab.speakers().items()}


def test_recovery():
    rooms = make_rooms()

    plain = Tab(gamma=0.0, within=None)
    for d in rooms:
        plain.add(d)
    plain.fit()
    c_plain = corr(mus(plain))

    judged = Tab(gamma=0.0, within=None)
    for d in rooms:
        judged.add(d)
    blur = judged.fit_judges(rounds=2, tau=1.0)
    c_judged = corr(mus(judged))

    assert blur['blind chair'] >= 1.0
    assert blur['sharp chair'] <= 0.25
    assert blur['blind chair'] > 4 * max(blur['sharp chair'], 0.05)
    assert c_judged >= c_plain - 0.005
