"""Pattern-selection policies. They see only (pattern, reward) feedback from tasks they actually ran
(bandit feedback). `ctx` is the coarse task type; only PerTypeThompson uses it."""

import math

import numpy as np


class Policy:
    name = "base"

    def __init__(self, arms, rng: np.random.Generator, r_lo: float, r_hi: float):
        self.arms = list(arms)
        self.k = len(arms)
        self.rng = rng
        self.r_lo, self.r_hi = r_lo, r_hi
        self.n = np.zeros(self.k, int)
        self.s = np.zeros(self.k)  # sum of raw rewards
        self.frozen = False

    def unit(self, r):  # rescale to [0, 1] for UCB1 / Beta-Bernoulli TS
        return float(np.clip((r - self.r_lo) / (self.r_hi - self.r_lo), 0.0, 1.0))

    def means(self):
        return np.where(self.n > 0, self.s / np.maximum(self.n, 1), 0.0)

    def _argmax(self, v):
        v = np.asarray(v, float)
        best = np.flatnonzero(v == v.max())
        return int(self.rng.choice(best))

    def select(self, t, ctx=None) -> int:
        raise NotImplementedError

    def update(self, a, r, ctx=None):
        assert not self.frozen, "policy is frozen: held-out feedback must never reach update()"
        self.n[a] += 1
        self.s[a] += r

    def deploy(self, ctx=None) -> int:
        """Pattern used on the held-out set after online learning (greedy w.r.t. the learned estimate)."""
        tried = self.n > 0
        if not tried.any():
            return int(self.rng.integers(self.k))
        return self._argmax(np.where(tried, self.means(), -np.inf))

    def state(self) -> dict:
        return {"n": self.n.tolist(), "mean": np.round(self.means(), 4).tolist()}


class RandomPolicy(Policy):
    name = "random"

    def select(self, t, ctx=None):
        return int(self.rng.integers(self.k))

    def deploy(self, ctx=None):  # random selection stays random at test time
        return int(self.rng.integers(self.k))


class BestFixed(Policy):
    """Budget-fair 'global best fixed pattern': spend the online budget round-robin (equal observations per
    pattern), then commit to the pattern with the best mean reward for deployment."""
    name = "best_fixed"

    def select(self, t, ctx=None):
        return t % self.k


class EpsGreedy(Policy):
    name = "eps_greedy"

    def __init__(self, *a, eps=0.1, **kw):
        super().__init__(*a, **kw)
        self.eps = eps

    def select(self, t, ctx=None):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))
        return self._argmax(self.means())  # untried arms have estimate 0 (optimistic vs. negative rewards)


class UCB1(Policy):
    name = "ucb1"

    def __init__(self, *a, c=2.0, **kw):
        super().__init__(*a, **kw)
        self.c = c
        self.su = np.zeros(self.k)

    def update(self, a, r, ctx=None):
        super().update(a, r, ctx)
        self.su[a] += self.unit(r)

    def ucb(self, t):
        tot = max(int(self.n.sum()), 1)
        mu = self.su / np.maximum(self.n, 1)
        return np.where(self.n > 0, mu + np.sqrt(self.c * math.log(tot) / np.maximum(self.n, 1)), np.inf)

    def select(self, t, ctx=None):
        return self._argmax(self.ucb(t))

    def state(self):
        u = self.ucb(0)
        return {**super().state(), "ucb": [round(float(x), 4) if np.isfinite(x) else None for x in u]}


class Thompson(Policy):
    """Beta-Bernoulli TS with the Bernoulli trick (Agrawal & Goyal 2012): reward rescaled to [0,1], then a
    Bernoulli(r01) pseudo-outcome updates Beta(1,1) priors. Deploy = argmax posterior mean."""
    name = "thompson"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.alpha = np.ones(self.k)
        self.beta = np.ones(self.k)

    def select(self, t, ctx=None):
        return self._argmax(self.rng.beta(self.alpha, self.beta))

    def update(self, a, r, ctx=None):
        super().update(a, r, ctx)
        b = self.rng.random() < self.unit(r)
        self.alpha[a] += b
        self.beta[a] += 1 - b

    def deploy(self, ctx=None):
        return self._argmax(self.alpha / (self.alpha + self.beta))

    def state(self):
        return {**super().state(), "alpha": self.alpha.tolist(), "beta": self.beta.tolist()}


class PerTypeThompson(Policy):
    """Contextual-lite extension (not one of the required baselines): an independent Thompson sampler per
    coarse task type, with the global sampler as fallback for types not yet seen."""
    name = "ts_per_type"

    def __init__(self, arms, rng, r_lo, r_hi):
        super().__init__(arms, rng, r_lo, r_hi)
        self.glob = Thompson(arms, rng, r_lo=r_lo, r_hi=r_hi)
        self.per = {}

    def _get(self, ctx):
        if ctx not in self.per:
            self.per[ctx] = Thompson(self.arms, self.rng, r_lo=self.r_lo, r_hi=self.r_hi)
        return self.per[ctx]

    def select(self, t, ctx=None):
        return self._get(ctx).select(t)

    def update(self, a, r, ctx=None):
        super().update(a, r, ctx)
        self.glob.update(a, r)
        self._get(ctx).update(a, r)

    def deploy(self, ctx=None):
        p = self.per.get(ctx)
        return p.deploy() if p is not None and p.n.sum() >= self.k else self.glob.deploy()

    def state(self):
        return {**super().state(), "per_type": {k: v.state() for k, v in self.per.items()}}


REQUIRED = [RandomPolicy, BestFixed, EpsGreedy, UCB1, Thompson]
EXTRA = [PerTypeThompson]
