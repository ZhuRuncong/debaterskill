"""Inner-loop convergence of 3+-team rooms under the default sweep budget."""
import pytest

from debaterskill import Debate, Tab, Team, observation
from debaterskill import _core


@pytest.fixture(autouse=True)
def restore_sweeps():
    prev = _core.get_sweeps()
    _core.set_sweeps(64, 1e-12)
    yield
    _core.set_sweeps(*prev)


def four_teams():
    return [[(0.0, 3.0, 1.0), (0.5, 2.0, 1.0)],
            [(1.0, 1.0, 1.0), (-0.5, 3.0, 1.0)],
            [(0.2, 2.5, 1.0), (0.9, 1.5, 1.0)],
            [(-1.0, 2.0, 1.0), (0.3, 2.8, 1.0)]]


def test_five_sweeps_is_not_converged():
    teams, result = four_teams(), [3.0, 2.0, 1.0, 0.0]
    _core.set_sweeps(5, 0.0)
    _, legacy = observation(teams, result, False, 0.0)
    _core.set_sweeps(64, 1e-12)
    _, converged = observation(teams, result, False, 0.0)
    moved = max(abs(a[0] - b[0])
                for ta, tb in zip(legacy, converged) for a, b in zip(ta, tb))
    assert moved > 0.0


def test_converged_result_is_stable():
    teams, result = four_teams(), [3.0, 2.0, 1.0, 0.0]
    _core.set_sweeps(64, 1e-12)
    _, a = observation(teams, result, False, 0.0)
    _core.set_sweeps(200, 1e-15)
    _, b = observation(teams, result, False, 0.0)
    for ta, tb in zip(a, b):
        for x, y in zip(ta, tb):
            assert abs(x[0] - y[0]) <= 1e-11
            assert abs(x[1] - y[1]) <= 1e-11


def test_sweep_histogram_records_convergence():
    _core.reset_sweep_hist()
    observation(four_teams(), [3.0, 2.0, 1.0, 0.0], False, 0.0)
    hist = _core.sweep_hist()
    assert sum(hist) == 1
    used = next(k for k, v in enumerate(hist) if v)
    assert 1 < used < 64


def test_two_team_observation_needs_no_sweeps():
    _core.reset_sweep_hist()
    observation([[(0.0, 3.0, 1.0)], [(1.0, 3.0, 1.0)]], [1.0, 0.0], False, 0.0)
    assert sum(_core.sweep_hist()) == 0


def test_fit_converges_under_default_regime():
    tab = Tab(mu=0.0, sigma=3.0, beta=1.0, gamma=0.0, between='ordinal')
    tab.add(Debate([Team(speakers=[f'{s}1', f'{s}2'], side=s, points=3 - i)
                    for i, s in enumerate(('og', 'oo', 'cg', 'co'))], day=0))
    step, iters = tab.fit(iterations=30, epsilon=1e-9)
    assert max(step) <= 1e-9
