"""Skill dynamics between appearances: sigma-capped walk vs the unbounded walk.

The mean is a plain random walk in both modes (an absence never rewrites the
skill estimate); the default caps the forward message's sigma at the prior so a
returning entity is never rated as more uncertain than a newcomer, without
touching the mean.
"""
from debaterskill import Tab, Debate, Team


def reappear(gap, gamma, bound_sigma, sigma=3.0):
    """Fit a player who wins 20 weekly rooms, then one more after `gap` days.
    Returns (settled node, reappearance node) from the player's curve."""
    tab = Tab(mu=0.0, sigma=sigma, beta=1.0, gamma=gamma, within=None,
              motions=False, bound_sigma=bound_sigma)
    day = 0
    for r in range(20):
        tab.add(Debate([Team(speakers=['a', f'p{r}'], points=3),
                        Team(speakers=[f'q{r}', f'w{r}'], points=2),
                        Team(speakers=[f'x{r}', f'y{r}'], points=1),
                        Team(speakers=[f'z{r}', f'u{r}'], points=0)], day=day))
        day += 7
    tab.add(Debate([Team(speakers=['a', 'pN'], points=3),
                    Team(speakers=['qN', 'wN'], points=2),
                    Team(speakers=['xN', 'yN'], points=1),
                    Team(speakers=['zN', 'uN'], points=0)], day=day - 7 + gap))
    tab.fit(iterations=60, epsilon=1e-12)
    curve = tab.curves()['a']
    return curve[-2][1], curve[-1][1]


def test_cap_bounds_sigma_at_the_prior():
    """After a long absence the unbounded walk grows past the newcomer prior; the
    capped walk does not."""
    _, s_cap = reappear(1095, 0.15, bound_sigma=True)
    _, s_walk = reappear(1095, 0.15, bound_sigma=False)
    assert s_cap.sigma <= 3.0 + 1e-9
    assert s_walk.sigma > 3.0
    assert s_cap.sigma < s_walk.sigma


def test_cap_preserves_the_mean_through_a_gap():
    """The cap widens sigma but never pulls the estimate toward the field: a
    strong player is still strong on return, not regressed toward the prior."""
    settled, returned = reappear(1095, 0.15, bound_sigma=True)
    assert settled.mu > 5.0                 # genuinely settled high
    assert returned.mu > 0.8 * settled.mu   # not dragged toward the prior mean 0


def test_cap_matches_the_walk_on_short_gaps():
    """When the cap does not bind, the forward marginal is untouched and the
    backward likelihood is never capped, so the two modes are identical."""
    for gap in (7, 30, 90):
        c_cap = reappear(gap, 0.04, bound_sigma=True)
        c_walk = reappear(gap, 0.04, bound_sigma=False)
        for g_cap, g_walk in zip(c_cap, c_walk):
            assert abs(g_cap.mu - g_walk.mu) < 1e-9
            assert abs(g_cap.sigma - g_walk.sigma) < 1e-9


def test_cap_is_a_noop_without_drift():
    """gamma=0 means no dynamics at all, so the flag cannot matter."""
    cap = reappear(1000, 0.0, bound_sigma=True)
    walk = reappear(1000, 0.0, bound_sigma=False)
    for g_cap, g_walk in zip(cap, walk):
        assert abs(g_cap.mu - g_walk.mu) < 1e-12
        assert abs(g_cap.sigma - g_walk.sigma) < 1e-12
