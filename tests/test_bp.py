"""BP shape validation and the room-to-observation decomposition."""
import math

import pytest

from debaterskill import Tab, Debate, Outround, Team, Motion


def team(names, side=None, points=None, speaks=None, advancing=None):
    return Team(speakers=names, side=side, points=points, speaks=speaks,
                advancing=advancing)


def observations(tab, debate):
    return list(tab._decompose(debate))


def test_strictness():
    with pytest.raises(ValueError):
        Team(speakers=['a', 'b', 'c'])
    with pytest.raises(ValueError):
        Team(speakers=['a', 'a'])
    with pytest.raises(ValueError):
        Team(speakers=['a'], speaks=[70, 71])
    with pytest.raises(ValueError):
        Team(speakers=['a'], side='opening')
    with pytest.raises(ValueError):
        Team()
    with pytest.raises(ValueError):
        Debate([team(['a'], points=1)], day=0)
    with pytest.raises(ValueError):
        Debate([team([f'p{i}'], points=i) for i in range(5)], day=0)
    with pytest.raises(ValueError):
        Debate([team(['a'], points=1), team(['a'], points=0)], day=0)
    with pytest.raises(ValueError):
        Debate([team(['a'], points=1), team(['b'])], day=0)
    with pytest.raises(ValueError):
        Outround([team(['a'], advancing=True), team(['b'])], day=0)
    with pytest.raises(ValueError):
        Outround([team(['a'], advancing=True), team(['b'], advancing=True)], day=0)
    Debate([team(['a'], points=1), Team(name='swing team', points=0)], day=0)


def test_outround_partial_order():
    tab = Tab()
    d = Outround([team(['a1', 'a2'], advancing=True),
                  team(['b1', 'b2'], advancing=False),
                  team(['c1', 'c2'], advancing=True),
                  team(['d1', 'd2'], advancing=False)], day=0)
    cs = observations(tab, d)
    assert len(cs) == 4
    assert all(sc == [1.0, 0.0] and not cont for _, sc, cont, _, tag in cs)
    assert all(tag == 'outround' for *_, tag in cs)
    winners = {tuple(c[0][0]) for c in cs}
    assert winners == {('s:a1', 's:a2'), ('s:c1', 's:c2')}


def test_within_gate_and_modes():
    def room(d_speaks):
        return Debate([team(['a1', 'a2'], points=3, speaks=[71, 70]),
                       team(['b1', 'b2'], points=2, speaks=[73, 73]),
                       team(['c1', 'c2'], points=1, speaks=[69, 68]),
                       team(['d1', 'd2'], points=0, speaks=d_speaks)],
                      day=0, scale=2.0)
    gap = Tab(within='gap')
    cs = observations(gap, room([67, 66]))
    assert sum(1 for c in cs if c[4] == 'speaker') == 4
    assert all(c[2] for c in cs if c[4] == 'speaker')
    assert [c for c in cs if c[4] == 'speaker'][0][1] == [35.5, 35.0]
    cs = observations(gap, room([67, None]))
    assert sum(1 for c in cs if c[4] == 'speaker') == 0
    assert sum(1 for c in cs if c[4] == 'team') == 1

    tie = Tab(within='ordinal', within_p_draw=0.0)
    cs = observations(tie, room([67, 66]))
    spk = [c for c in cs if c[4] == 'speaker']
    assert len(spk) == 3 and not any(c[2] for c in spk)
    tie2 = Tab(within='ordinal', within_p_draw=0.3)
    assert sum(1 for c in observations(tie2, room([67, 66])) if c[4] == 'speaker') == 4

    speaks = Tab(between='speaks')
    cs = observations(speaks, room([67, 66]))
    assert len(cs) == 1 and len(cs[0][0]) == 8 and cs[0][2]
    cs = observations(speaks, room([67, None]))
    assert len(cs) == 1 and len(cs[0][0]) == 4 and not cs[0][2]


def test_motion_placement():
    tab = Tab(within='gap', motions=True, motion_sigma=0.7)
    m = Motion('thw-test')
    d = Debate([team(['a1', 'a2'], side='og', points=3, speaks=[71, 70]),
                team(['b1', 'b2'], side='oo', points=2, speaks=[73, 73]),
                team(['c1', 'c2'], side='cg', points=1, speaks=[69, 68]),
                team(['d1', 'd2'], side='co', points=0, speaks=[67, 66])],
               day=0, motion=m)
    cs = observations(tab, d)
    spk = [c for c in cs if c[4] == 'speaker']
    tm = [c for c in cs if c[4] == 'team']
    assert all(all(len(lu) == 1 for lu in c[0]) for c in spk)
    assert len(tm) == 1
    assert all(len(lu) == 3 and lu[2].startswith('m:') for lu in tm[0][0])
    tab.add(d)
    tab.fit()
    bias = tab.motion_bias()
    assert set(bias) == {('thw-test', s) for s in ('og', 'oo', 'cg', 'co')}


def test_fit_and_forecast():
    tab = Tab(gamma=0.02)
    for day in (0, 7, 14):
        tab.add(Debate([team(['a1', 'a2'], points=3), team(['b1', 'b2'], points=2),
                        team(['c1', 'c2'], points=1), team(['d1', 'd2'], points=0)],
                       day=day))
    tab.fit()
    a = tab.speakers()['a1'][1]
    d = tab.speakers()['d1'][1]
    assert a.mu > d.mu
    lp, n = tab.forecast(
        Debate([team(['a1', 'a2'], points=3), team(['b1', 'b2'], points=2),
                team(['c1', 'c2'], points=1), team(['d1', 'd2'], points=0)], day=21))
    assert n == 1 and math.exp(lp) > 1 / 24
    lp2, _ = tab.forecast(
        Debate([team(['a1', 'a2'], points=0), team(['b1', 'b2'], points=1),
                team(['c1', 'c2'], points=2), team(['d1', 'd2'], points=3)], day=21))
    assert lp > lp2
