"""
Confirmation statistics: a Beta-Binomial posterior over a regularity's success
probability p, updated only from *independent execution confirmations* (never from
the model's own confidence).

This is the single place where a mathematical threshold lives. Two thresholds with
a deliberate asymmetry:

  * ACTING (production side, looser): the regularity may replay autonomously.
  * LEARNING (learning side, stricter): a confirmed observation may reshape the
    induced structure (promote a slot / fold an example).

The asymmetry production << learning keeps the online feedback-loop gain below 1
and is the lever that suppresses self-poisoning (demonstrated in Phase 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from scipy.stats import beta as _beta


@dataclass
class Posterior:
    """Beta(alpha0 + successes, beta0 + failures).

    An optional recency ``decay`` in (0, 1] down-weights old evidence before each
    update, giving an effective window ~1/(1-decay). decay=1.0 keeps all evidence
    (no forgetting). Decay is what lets a long-matured regularity actually lose
    confidence after distribution drift instead of being anchored by ancient
    successes."""
    successes: float = 0.0
    failures: float = 0.0
    alpha0: float = 1.0
    beta0: float = 1.0
    decay: float = 1.0

    def update(self, success: bool) -> None:
        if self.decay < 1.0:
            self.successes *= self.decay
            self.failures *= self.decay
        if success:
            self.successes += 1.0
        else:
            self.failures += 1.0

    @property
    def alpha(self) -> float:
        return self.alpha0 + self.successes

    @property
    def beta(self) -> float:
        return self.beta0 + self.failures

    @property
    def n(self) -> int:
        return self.successes + self.failures

    def prob_exceeds(self, p_target: float) -> float:
        """P(p > p_target) under the current Beta posterior."""
        # survival function = 1 - CDF
        return float(_beta.sf(p_target, self.alpha, self.beta))

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass(frozen=True)
class Thresholds:
    # Acting (production): may replay when P(p > p_act) >= c_act.
    p_act: float = 0.85
    c_act: float = 0.80
    # Learning (structural): may reshape structure when P(p > p_learn) >= c_learn.
    # Strictly higher bar than acting.
    p_learn: float = 0.95
    c_learn: float = 0.95

    def may_act(self, post: Posterior) -> bool:
        return post.prob_exceeds(self.p_act) >= self.c_act

    def may_learn_structure(self, post: Posterior) -> bool:
        return post.prob_exceeds(self.p_learn) >= self.c_learn


def confirmations_for_acting(thr: Thresholds, *, clean: bool = True) -> int:
    """
    Smallest number of consecutive clean successes (0 prior failures) needed to
    cross the ACTING bar from a fresh Beta(1,1) prior. Reported as the
    sample-efficiency of autonomous replay.
    """
    post = Posterior()
    k = 0
    while not thr.may_act(post) and k < 10_000:
        post.update(clean)
        k += 1
    return k
