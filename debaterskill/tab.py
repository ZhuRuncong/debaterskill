import math

from ._core import Gaussian, TabCore
from .entities import _SEP, _SIDES, Motion, Speaker

# Observation tags, so evidence can be split by kind (log_evidence(tag=...)).
_TAGS = {'team': 0, 'speaker': 1, 'outround': 2, 'raw': 3}


def _decode(key):
    kind, rest = key[:2], key[2:]
    if kind == 's:':
        return rest
    if kind == 't:':
        return ('team', rest)
    slug, side = rest.split(_SEP)
    return (slug, side)


class Tab:
    """The whole record: every debate, one TrueSkill Through Time fit.

    `between` picks how a room's team results become observations, `within`
    how credit splits between partners; see the README for the modes.
    """

    def __init__(self, mu=0.0, sigma=3.0, beta=1.0, gamma=0.04,
                 between='ordinal', within='gap', p_draw=0.0, within_p_draw=0.0,
                 motions=True, motion_sigma=1.0, period='date', p_chaos=0.0):
        if between not in ('ordinal', 'speaks'):
            raise ValueError("between must be 'ordinal' or 'speaks'")
        if within not in (None, 'none', 'ordinal', 'gap'):
            raise ValueError("within must be None, 'ordinal' or 'gap'")
        if period not in ('date', 'week'):
            raise ValueError("period must be 'date' or 'week'")
        if not 0.0 <= p_chaos < 1.0:
            raise ValueError('p_chaos must be in [0, 1)')
        self.mu, self.sigma, self.beta, self.gamma = mu, sigma, beta, gamma
        self.between = between
        self.within = None if within == 'none' else within
        self.p_draw = p_draw
        self.within_p_draw = within_p_draw
        self.motions = motions
        self.motion_sigma = motion_sigma
        self.period = period
        self.p_chaos = p_chaos
        self._core = TabCore(mu, sigma, beta, gamma, 7 if period == 'week' else 1,
                             p_chaos)
        self._enrolled = set()
        self.judge_blur = {}
        self._judged = {}

    def enroll(self, *entities):
        """Register entities whose priors override the Tab defaults.

        A Motion is enrolled on every side, since any side may appear later.
        """
        for ent in entities:
            if isinstance(ent, Motion):
                for side in _SIDES:
                    key = ent.side_key(side)
                    self._enrolled.add(key)
                    self._core.enroll(
                        key,
                        0.0 if ent.mu is None else ent.mu,
                        self.motion_sigma if ent.sigma is None else ent.sigma,
                        0.0, 0.0)
            else:
                key = ent.key
                self._enrolled.add(key)
                self._core.enroll(
                    key,
                    self.mu if ent.mu is None else ent.mu,
                    self.sigma if ent.sigma is None else ent.sigma,
                    self.beta if ent.beta is None else ent.beta,
                    self.gamma if ent.gamma is None else ent.gamma)

    def _enroll(self, ent):
        key = ent.key
        if key not in self._enrolled:
            self._enrolled.add(key)
            # The core interns unknown keys with the Tab defaults, so an
            # explicit enroll is only needed when something is overridden.
            if any(v is not None for v in (ent.mu, ent.sigma, ent.beta, ent.gamma)):
                self._core.enroll(
                    key,
                    self.mu if ent.mu is None else ent.mu,
                    self.sigma if ent.sigma is None else ent.sigma,
                    self.beta if ent.beta is None else ent.beta,
                    self.gamma if ent.gamma is None else ent.gamma)
        return key

    def _enroll_motion(self, motion, side):
        key = motion.side_key(side)
        if key not in self._enrolled:
            self._enrolled.add(key)
            # Static side offset: no performance noise, no drift.
            self._core.enroll(
                key,
                0.0 if motion.mu is None else motion.mu,
                self.motion_sigma if motion.sigma is None else motion.sigma,
                0.0, 0.0)
        return key

    def _members(self, team):
        if team.assigned:
            keys = [self._enroll(s) for s in team.speakers]
            # An iron gave both speeches: double the speaker so the team is
            # not entered at half strength against opponents' two.
            if team.iron:
                keys = keys * 2
            return keys
        return [self._enroll(team)]

    def _decompose(self, debate):
        teams = debate.teams
        motion = debate.motion if self.motions else None

        def lineup(team):
            keys = self._members(team)
            if motion is not None and team.side:
                keys = keys + [self._enroll_motion(motion, team.side)]
            return keys

        if debate.outround:
            for a in (t for t in teams if t.advancing):
                for b in (t for t in teams if not t.advancing):
                    yield ([lineup(a), lineup(b)], [1.0, 0.0], False,
                           self.p_draw, 'outround')
            return

        scored = [[(s, v) for s, v in zip(t.speakers or [], t.speaks or []) if v is not None]
                  for t in teams]
        # An iron's one score stands for both speeches.
        scored = [sc * 2 if t.iron and len(sc) == 1 else sc
                  for t, sc in zip(teams, scored)]

        if self.between == 'speaks':
            # One continuous ballot over all scored speakers encodes the team
            # result (ranking follows total speaks; `points` are ignored
            # here), the margins and the partner gaps at once. Complete
            # rooms only; partial scores would bias it, so incomplete rooms
            # fall through to the ordinal team ballot.
            if all(len(sc) == 2 for sc in scored):
                lineups, scores = [], []
                for team, sc in zip(teams, scored):
                    tail = ([self._enroll_motion(motion, team.side)]
                            if motion is not None and team.side else [])
                    for spk, v in sc:
                        lineups.append([self._enroll(spk)] + tail)
                        scores.append(v / debate.scale)
                yield (lineups, scores, True, 0.0, 'team')
                return
        elif self.within is not None and all(len(sc) == 2 for sc in scored):
            for t, sc in zip(teams, scored):
                # No gap between a speaker and themself.
                if t.iron:
                    continue
                (s1, v1), (s2, v2) = sc
                k1, k2 = self._enroll(s1), self._enroll(s2)
                if self.within == 'gap':
                    yield ([[k1], [k2]], [v1 / debate.scale, v2 / debate.scale],
                           True, 0.0, 'speaker')
                # Equal speaks with p_draw=0 would be a zero-probability
                # observation, so the pair is skipped instead.
                elif v1 != v2 or self.within_p_draw > 0.0:
                    yield ([[k1], [k2]], [float(v1), float(v2)],
                           False, self.within_p_draw, 'speaker')

        points = [float(t.points) for t in teams]
        if len(set(points)) == len(points):
            yield ([lineup(t) for t in teams], points, False, self.p_draw, 'team')
        else:
            # A tied full ordering has zero likelihood at p_draw=0, so ties
            # reduce the room to its strict pairwise comparisons. Tied pairs
            # are dropped at any p_draw: a draw is never used as evidence.
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    if points[i] == points[j]:
                        continue
                    hi, lo = ((teams[i], teams[j]) if points[i] > points[j]
                              else (teams[j], teams[i]))
                    yield ([lineup(hi), lineup(lo)], [1.0, 0.0], False,
                           self.p_draw, 'team')

    def add(self, *debates):
        """Add debates to the record. Returns self."""
        for debate in debates:
            for lineups, scores, continuous, p_draw, tag in self._decompose(debate):
                # Chair blur is excess perceptual noise on the team
                # performances the chair ranks (excess over the average judge,
                # which beta already carries); the partner gap gets none,
                # where it would only widen the gap.
                chaired = debate.chair is not None and tag != 'speaker'
                noise = self.judge_blur.get(debate.chair, 0.0) if chaired else 0.0
                self._core.add_ballot(debate.day, lineups, scores, continuous, p_draw,
                                     _TAGS[tag], noise)
                if chaired:
                    self._judged.setdefault(debate.chair, []).append(
                        self._core.size() - 1)
        return self

    def _ballot_member(self, m):
        if isinstance(m, str):
            return self._enroll(Speaker(m))
        if isinstance(m, tuple):
            motion, side = m
            return self._enroll_motion(motion, side)
        return self._enroll(m)

    def add_ballot(self, lineups, scores, day=0, continuous=False, p_draw=0.0, tag='raw'):
        """Add one pre-decomposed observation; members are names, entities, or
        (Motion, side) tuples. Returns self."""
        keys = [[self._ballot_member(m) for m in lineup] for lineup in lineups]
        self._core.add_ballot(day, keys, [float(s) for s in scores], continuous, p_draw,
                             _TAGS[tag])
        return self

    def fit(self, iterations=10, epsilon=1e-4, verbose=False):
        """Run EP smoothing until no rating moves more than epsilon.
        Returns ((max mu step, max sigma step), iterations used)."""
        return self._core.fit(iterations, epsilon, verbose)

    def log_evidence(self, tag=None):
        """Marginal log likelihood of the fitted record, optionally per tag."""
        return self._core.log_evidence(-1 if tag is None else _TAGS[tag])

    def evidence_counts(self, tag=None):
        """(used, dropped) observation counts; dropped means zero evidence."""
        return self._core.evidence_counts(-1 if tag is None else _TAGS[tag])

    def curves(self):
        """key -> [(time, Gaussian skill)] posterior trajectory."""
        return {_decode(k): [(t, Gaussian(mu, sigma)) for t, mu, sigma in pts]
                for k, pts in self._core.curves().items()}

    def ratings(self):
        """key -> (last time, Gaussian skill)."""
        return {k: (pts[-1][0], pts[-1][1]) for k, pts in self.curves().items()}

    def speakers(self):
        """Speaker ratings only (string keys)."""
        return {k: v for k, v in self.ratings().items() if isinstance(k, str)}

    def motion_bias(self):
        """(slug, side) -> Gaussian side offset."""
        return {k: v[1] for k, v in self.ratings().items()
                if isinstance(k, tuple) and k[0] != 'team'}

    def forecast(self, debate, tags=('team', 'outround'), chair_noise=True):
        """Held-out log evidence of a room under the fitted state; no update.
        Returns (total log evidence, observation count)."""
        total = 0.0
        count = 0
        for lineups, scores, continuous, p_draw, tag in self._decompose(debate):
            if tag not in tags:
                continue
            noise = (self.judge_blur.get(debate.chair, 0.0)
                     if chair_noise and debate.chair is not None and tag != 'speaker'
                     else 0.0)
            total += self._core.forecast(debate.day, lineups, scores, continuous,
                                         p_draw, noise)
            count += 1
        return total, count

    def fit_judges(self, rounds=2, tau=0.6,
                   grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5),
                   iterations=10, epsilon=1e-4, verbose=False):
        """Learn per-chair blur by empirical Bayes; returns {chair: variance}.

        Each round: with skills fixed, every chair's blur maximizes the
        leave-one-out (cavity) evidence of their judged observations under a
        shrinkage prior concentrated at zero, half-normal(tau) on the blur
        SD (the -u/(2 tau^2) term); room noises are then updated and skills
        refitted: an EM loop with EP evidence as the inner objective.
        Leave-one-out runs per observation, which equals per room except for
        outrounds and tied rooms.
        """
        self.fit(iterations, epsilon, verbose)
        for r in range(rounds):
            for chair, idxs in self._judged.items():
                best_u, best_score = 0.0, None
                for s in grid:
                    u = s * s
                    score = (sum(self._core.evidence_at(i, u) for i in idxs)
                             - u / (2.0 * tau * tau))
                    if best_score is None or score > best_score:
                        best_u, best_score = u, score
                self.judge_blur[chair] = best_u
            for chair, idxs in self._judged.items():
                u = self.judge_blur[chair]
                for i in idxs:
                    self._core.set_noise(i, u)
            self.fit(iterations, epsilon, verbose)
            if verbose:
                blurred = sum(1 for u in self.judge_blur.values() if u > 0)
                print(f'judge round {r + 1}: {blurred}/{len(self.judge_blur)} '
                      f'chairs with excess blur', flush=True)
        return self.judge_blur

    def judges(self):
        """chair -> (observations judged, blur SD)."""
        return {chair: (len(idxs), math.sqrt(self.judge_blur.get(chair, 0.0)))
                for chair, idxs in self._judged.items()}

    @property
    def size(self):
        """Number of observations added so far."""
        return self._core.size()
