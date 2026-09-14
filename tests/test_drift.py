"""Skill dynamics between appearances: OU reversion vs the plain random walk."""
from debaterskill import Tab, Debate, Team


def reappear(gap, gamma, revert, sigma=3.0):
    tab = Tab(mu=0.0, sigma=sigma, beta=1.0, gamma=gamma, within=None,
              motions=False, revert=revert)
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
    tab.fit(iterations=40, epsilon=1e-8)
    mu, sigma = tab.curves()['a'][-1][1].mu, tab.curves()['a'][-1][1].sigma
    return mu, sigma


def test_reversion_floors_sigma_at_the_prior():
    """After a long absence the walk grows past the newcomer prior; OU does not."""
    _, s_ou = reappear(1095, 0.15, revert=True)
    _, s_walk = reappear(1095, 0.15, revert=False)
    assert s_ou <= 3.0 + 1e-9
    assert s_walk > 3.0
    assert s_ou < s_walk


def test_reversion_regresses_the_mean_over_a_long_gap():
    """A long-absent entity's estimate decays toward the prior mean (0)."""
    mu_ou, _ = reappear(1095, 0.15, revert=True)
    mu_walk, _ = reappear(1095, 0.15, revert=False)
    assert abs(mu_ou) < abs(mu_walk)


def test_reversion_matches_the_walk_on_short_gaps():
    """Over ordinary gaps the two dynamics are near-identical by construction."""
    for gap in (7, 30, 90):
        _, s_ou = reappear(gap, 0.04, revert=True)
        _, s_walk = reappear(gap, 0.04, revert=False)
        assert abs(s_ou - s_walk) < 0.05


def test_reversion_is_a_noop_without_drift():
    """gamma=0 means no dynamics at all, so the flag cannot matter."""
    mu_ou, s_ou = reappear(1000, 0.0, revert=True)
    mu_walk, s_walk = reappear(1000, 0.0, revert=False)
    assert abs(mu_ou - mu_walk) < 1e-12
    assert abs(s_ou - s_walk) < 1e-12
