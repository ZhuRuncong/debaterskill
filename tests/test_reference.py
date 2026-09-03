"""Differential test against trueskillthroughtime under its fixed-5-sweep regime."""
import math
import random

import pytest

ttt = pytest.importorskip('trueskillthroughtime')

from debaterskill import observation
from debaterskill import _core
from debaterskill._core import TabCore

TOL = 1e-9
MARGIN_TOL = 1e-7


@pytest.fixture(autouse=True)
def legacy_sweeps():
    prev = _core.get_sweeps()
    _core.set_sweeps(5, 0.0)
    yield
    _core.set_sweeps(*prev)


def close(a, b, tol=TOL):
    if math.isinf(a) and math.isinf(b):
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_game(teams_ms, result, obs, p_draw, tol=TOL, label=''):
    ref_teams = [[ttt.Player(ttt.Gaussian(mu, sigma), beta, 0.0) for mu, sigma, beta in t]
                 for t in teams_ms]
    g = ttt.Game(ref_teams, list(result), p_draw,
                 obs='Continuous' if obs == 'C' else 'Ordinal')
    ev, lik = observation(teams_ms, list(result), obs == 'C', p_draw)
    assert close(ev, g.evidence, tol), f'{label} evidence {ev} vs {g.evidence}'
    for t in range(len(teams_ms)):
        for i in range(len(teams_ms[t])):
            rmu, rsig = g.likelihoods[t][i].mu, g.likelihoods[t][i].sigma
            mu, sig = lik[t][i]
            assert close(mu, rmu, tol), f'{label} [{t}][{i}] mu {mu} vs {rmu}'
            assert close(sig, rsig, tol), f'{label} [{t}][{i}] sigma {sig} vs {rsig}'


def random_team(rng, nmin=1, nmax=3):
    return [(rng.uniform(-3, 3), rng.uniform(0.5, 4), rng.uniform(0.0, 2))
            for _ in range(rng.randint(nmin, nmax))]


def test_games():
    rng = random.Random(7)
    for trial in range(200):
        nteams = rng.randint(2, 5)
        teams = [random_team(rng) for _ in range(nteams)]
        if rng.random() < 0.3:
            result = [float(rng.randint(0, nteams)) for _ in range(nteams)]
            p_draw = rng.uniform(0.05, 0.4)
            tol = MARGIN_TOL
        else:
            perm = list(range(nteams))
            rng.shuffle(perm)
            result = [float(p) for p in perm]
            p_draw = rng.choice([0.0, rng.uniform(0.05, 0.4)])
            tol = TOL if p_draw == 0.0 else MARGIN_TOL
        check_game(teams, result, 'O', p_draw, tol, f'ordinal-{trial}')
        result_c = [rng.uniform(-3, 3) for _ in range(nteams)]
        if len({round(x, 12) for x in result_c}) == nteams:
            check_game(teams, result_c, 'C', 0.0, TOL, f'continuous-{trial}')
    for trial in range(50):
        teams = [random_team(rng, 1, 1) for _ in range(8)]
        result = [rng.uniform(70, 80) for _ in range(8)]
        check_game(teams, result, 'C', 0.0, TOL, f'room-{trial}')


def random_history(rng, nplayers, ngames, dup_ok=False):
    players = [f'p{i}' for i in range(nplayers)]
    comp, res, times, obs = [], [], [], []
    for _ in range(ngames):
        nteams = rng.randint(2, 4)
        continuous = rng.random() < 0.4
        game, used = [], set()
        for _ in range(nteams):
            size = rng.randint(1, 2)
            team = []
            for _ in range(size):
                p = rng.choice(players)
                if not dup_ok:
                    while p in used:
                        p = rng.choice(players)
                used.add(p)
                team.append(p)
            game.append(team)
        if continuous:
            r = [round(rng.uniform(-2, 2), 6) for _ in range(nteams)]
            while len({x for x in r}) < nteams:
                r = [round(rng.uniform(-2, 2), 6) for _ in range(nteams)]
        else:
            perm = list(range(nteams))
            rng.shuffle(perm)
            r = [float(x) for x in perm]
        comp.append(game)
        res.append(r)
        times.append(rng.randint(0, 40))
        obs.append('Continuous' if continuous else 'Ordinal')
    return comp, res, times, obs


def check_history(comp, res, times, obs, mu, sigma, beta, gamma, priors=None,
                  iterations=6, tol=TOL, label=''):
    ref_priors = {k: ttt.Player(ttt.Gaussian(m, s), b, g)
                  for k, (m, s, b, g) in (priors or {}).items()}
    h = ttt.History(comp, results=res, times=times, mu=mu, sigma=sigma, beta=beta,
                    gamma=gamma, p_draw=0.0, obs=obs, priors=ref_priors or None)
    h.convergence(iterations=iterations, epsilon=1e-30, verbose=False)

    core = TabCore(mu, sigma, beta, gamma, 1)
    for k, (m, s, b, g) in (priors or {}).items():
        core.enroll(k, m, s, b, g)
    for c, r, t, o in zip(comp, res, times, obs):
        core.add_observation(t, c, r, o == 'Continuous', 0.0, 3)
    core.fit(iterations, 1e-30, False)

    assert close(core.log_evidence(-1), h.log_evidence(), tol), \
        f'{label} evidence {core.log_evidence(-1)} vs {h.log_evidence()}'
    ref_lc = h.learning_curves()
    lc = core.curves()
    for name, pts in ref_lc.items():
        mine = lc[name]
        assert len(mine) == len(pts), f'{label} {name} curve length'
        for (rt, rg), (mt, mmu, msig) in zip(pts, mine):
            assert mt == rt, f'{label} {name} time {mt} vs {rt}'
            assert close(mmu, rg.mu, tol), f'{label} {name}@{rt} mu {mmu} vs {rg.mu}'
            assert close(msig, rg.sigma, tol), f'{label} {name}@{rt} sigma {msig} vs {rg.sigma}'


def test_histories():
    rng = random.Random(11)
    for trial in range(12):
        comp, res, times, obs = random_history(rng, 12, 60)
        check_history(comp, res, times, obs, 0.0, 3.0, 1.0, 0.04,
                      label=f'history-{trial}')
    comp, res, times, obs = random_history(rng, 8, 40)
    priors = {'p0': (1.5, 2.0, 0.5, 0.01), 'p1': (-1.0, 1.0, 0.0, 0.0)}
    check_history(comp, res, times, obs, 0.0, 3.0, 1.0, 0.04, priors=priors,
                  label='priors')


def test_duplicate_occurrence():
    rng = random.Random(3)
    comp = [[['a', 'x'], ['b', 'x']], [['x'], ['c']], [['a'], ['b'], ['c', 'x']]]
    res = [[1.0, 0.0], [1.0, 0.0], [2.0, 1.0, 0.0]]
    times = [0, 5, 5]
    obs = ['Ordinal', 'Ordinal', 'Ordinal']
    check_history(comp, res, times, obs, 0.0, 3.0, 1.0, 0.04, label='dup')
    for trial in range(6):
        comp, res, times, obs = random_history(rng, 5, 30, dup_ok=True)
        check_history(comp, res, times, obs, 0.0, 3.0, 1.0, 0.04,
                      label=f'dup-{trial}')


def test_week_period():
    core_d = TabCore(0.0, 3.0, 1.0, 0.04, 1)
    core_w = TabCore(0.0, 3.0, 1.0, 0.04, 7)
    games = [(0, [['a'], ['b']]), (2, [['a'], ['c']]), (9, [['b'], ['c']]),
             (16, [['a'], ['b']])]
    for day, c in games:
        core_d.add_observation(day, c, [1.0, 0.0], False, 0.0, 3)
        core_w.add_observation(day, c, [1.0, 0.0], False, 0.0, 3)
    core_d.fit(8, 1e-8, False)
    core_w.fit(8, 1e-8, False)
    days_d = {t for pts in core_d.curves().values() for t, *_ in pts}
    days_w = {t for pts in core_w.curves().values() for t, *_ in pts}
    assert days_d == {0, 2, 9, 16}
    assert days_w == {0, 7, 14}
