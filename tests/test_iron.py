"""Iron speakers count twice, not the team at half strength."""
import pytest

from debaterskill import Tab, Debate, Team


def test_iron_needs_exactly_one_speaker():
    with pytest.raises(ValueError):
        Team(speakers=['a', 'b'], iron=True, points=3)
    with pytest.raises(ValueError):
        Team(name='solo team', iron=True, points=3)
    Team(speakers=['a'], iron=True, points=3)


def _room(iron):
    return Debate([
        Team(speakers=['a'], iron=iron, points=3, side='og'),
        Team(speakers=['c', 'd'], points=2, side='oo'),
        Team(speakers=['e', 'f'], points=1, side='cg'),
        Team(speakers=['g', 'h'], points=0, side='co'),
    ], 0)


def test_iron_enters_the_room_at_twice_the_speaker():
    tab = Tab(between='ordinal', within=None, motions=False)
    assert tab._members(_room(True).teams[0]) == ['s:a', 's:a']
    assert tab._members(_room(False).teams[0]) == ['s:a']


def test_iron_is_not_inflated_relative_to_a_pair():
    def fit(iron):
        tab = Tab(mu=0.0, sigma=3.0, beta=1.0, gamma=0.0,
                  between='ordinal', within=None, motions=False)
        for i in range(12):
            tab.add(Debate([
                Team(speakers=['a'], iron=iron, points=3, side='og'),
                Team(speakers=['p%d' % i, 'q%d' % i], points=2, side='oo'),
                Team(speakers=['r%d' % i, 's%d' % i], points=1, side='cg'),
                Team(speakers=['t%d' % i, 'u%d' % i], points=0, side='co'),
            ], 0))
        tab.fit(iterations=40, epsilon=1e-8)
        return {k: g.mu for k, (_d, g) in tab.ratings().items()}['a']

    assert fit(True) < fit(False)


def test_iron_score_covers_both_speeches():
    tab = Tab(between='speaks', within='gap', motions=False)
    d = Debate([
        Team(speakers=['a'], speaks=[76], iron=True, points=3, side='og'),
        Team(speakers=['c', 'd'], speaks=[75, 74], points=2, side='oo'),
        Team(speakers=['e', 'f'], speaks=[73, 72], points=1, side='cg'),
        Team(speakers=['g', 'h'], speaks=[71, 70], points=0, side='co'),
    ], 0)
    parts = list(tab._decompose(d))
    assert any(cont for _l, _s, cont, _p, _t in parts)

    gap = Tab(between='ordinal', within='gap', motions=False)
    tags = [t for _l, _s, _c, _p, t in gap._decompose(d)]
    assert tags.count('speaker') == 3
