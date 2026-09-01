# \x1f cannot occur in a real name or slug, so composite keys never collide.
_SEP = '\x1f'
_SIDES = ('og', 'oo', 'cg', 'co', 'aff', 'neg')


class Speaker:
    """A debater. mu/sigma/beta/gamma override the Tab defaults when set on the
    speaker's first appearance (or via Tab.enroll); later overrides are ignored."""

    __slots__ = ('name', 'mu', 'sigma', 'beta', 'gamma')

    def __init__(self, name, mu=None, sigma=None, beta=None, gamma=None):
        self.name = name
        self.mu, self.sigma, self.beta, self.gamma = mu, sigma, beta, gamma

    @property
    def key(self):
        return 's:' + self.name

    def __repr__(self):
        return f'Speaker({self.name!r})'


class Team:
    """One team's entry in a room.

    With `speakers` the team performs as the sum of its members; without, the
    team itself is rated as a single atomic entity. `iron` marks a lone
    speaker who gave both speeches (downstream they count twice, so the team
    is not entered at half strength). `speaks` aligns with `speakers`;
    `points` is the room result (higher is better); `advancing` is only
    meaningful in an Outround.
    """

    __slots__ = ('name', 'speakers', 'speaks', 'side', 'points', 'advancing',
                 'iron', 'mu', 'sigma', 'beta', 'gamma')

    def __init__(self, name=None, speakers=None, speaks=None, side=None, points=None,
                 advancing=None, iron=False, mu=None, sigma=None, beta=None,
                 gamma=None):
        self.speakers = ([Speaker(s) if isinstance(s, str) else s for s in speakers]
                         if speakers else None)
        if self.speakers is not None:
            if not 1 <= len(self.speakers) <= 2:
                raise ValueError('a BP team fields 1 or 2 speakers')
            if len({s.name for s in self.speakers}) != len(self.speakers):
                raise ValueError('duplicate speaker in team')
        if iron and (self.speakers is None or len(self.speakers) != 1):
            raise ValueError('an iron team fields exactly one speaker')
        if speaks is not None:
            if not self.speakers or len(speaks) != len(self.speakers):
                raise ValueError('speaks must align with speakers')
        if side is not None and side not in _SIDES:
            raise ValueError(f'unknown side {side!r}')
        if name is None and self.speakers:
            name = ' & '.join(s.name for s in self.speakers)
        if name is None:
            raise ValueError('a team needs a name or speakers')
        self.name = name
        self.speaks = speaks
        self.side = side
        self.points = points
        self.advancing = advancing
        self.iron = bool(iron)
        self.mu, self.sigma, self.beta, self.gamma = mu, sigma, beta, gamma

    @property
    def assigned(self):
        return bool(self.speakers)

    @property
    def key(self):
        return 't:' + self.name

    def __repr__(self):
        return f'Team({self.name!r})'


class Motion:
    """A motion's static side offsets: one skill variable per position, with
    no drift and no performance noise."""

    __slots__ = ('slug', 'mu', 'sigma')

    def __init__(self, slug, mu=None, sigma=None):
        self.slug = slug
        self.mu, self.sigma = mu, sigma

    def side_key(self, side):
        return 'm:' + self.slug + _SEP + side

    def __repr__(self):
        return f'Motion({self.slug!r})'


class Judge:
    """A chair identity, for the per-judge blur model (`Tab.fit_judges`)."""

    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'Judge({self.name!r})'


class Debate:
    """An inround: 2-4 teams, each with `points`. `scale` divides continuous
    speak observations; `chair` attaches the room to a judge."""

    __slots__ = ('teams', 'day', 'motion', 'scale', 'chair')
    outround = False

    def __init__(self, teams, day, motion=None, scale=1.0, chair=None):
        if not 2 <= len(teams) <= 4:
            raise ValueError('a BP room seats 2 to 4 teams')
        names = []
        for t in teams:
            names += [s.name for s in t.speakers] if t.assigned else [t.name]
        if len(set(names)) != len(names):
            raise ValueError('the same speaker appears in more than one team')
        if self.outround:
            if any(t.advancing is None for t in teams):
                raise ValueError('every outround team needs an advancing flag')
            n_adv = sum(1 for t in teams if t.advancing)
            # All-advance or all-eliminated yields no comparison at all.
            if n_adv in (0, len(teams)):
                raise ValueError('an outround needs advancing and eliminated teams')
        elif any(t.points is None for t in teams):
            raise ValueError('every inround team needs points')
        self.teams = teams
        self.day = day
        self.motion = motion
        self.scale = scale
        self.chair = chair.name if isinstance(chair, Judge) else chair


class Outround(Debate):
    """An elimination room: each advancing team beats each eliminated team,
    with no comparison inside either group."""

    outround = True
