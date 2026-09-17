import json
from collections import deque
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np

from . import features as F
from . import game_utils as g


@dataclass
class Hyper:
    feature_set: str = F.DEFAULT_SET
    model: str = "linear"

    gamma: float = 0.97
    n_step: int = 5  # a bomb pays off 4 steps after it is dropped

    lr: float = 5e-4
    batch_size: int = 64
    train_every: int = 4
    updates_per_step: int = 1
    buffer_size: int = 150_000
    warmup: int = 2_000
    grad_clip: float = 5.0
    target_sync: int = 250
    double: bool = True
    ridge_lambda: float = 1.0
    refit_every: int = 0

    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_end: float = 1.0
    per_eps: float = 1e-3

    policy: str = "epsilon_greedy"
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_rounds: float = 3_000.0
    temperature: float = 1.0
    temperature_end: float = 0.05
    safe_exploration: bool = True

    forest_trees: int = 40
    forest_max_depth: int = 16
    forest_min_leaf: int = 20
    forest_max_features: float = 1.0
    fqi_every: int = 300
    fqi_iterations: int = 5
    fqi_max_samples: int = 150_000

    augment: int = 8

    reward_scheme: str = "potential+events"
    step_penalty: float = 0.0
    death_penalty: float = -8.0
    invalid_penalty: float = -0.4
    crate_reward: float = 0.5
    coin_found_reward: float = 0.5
    psi_coin: float = 0.5
    psi_crate: float = 0.2
    psi_enemy: float = 0.1
    psi_safe: float = 0.15
    psi_cleared: float = 1.0
    psi_escape: float = 1.0

    bomb_bonus: float = 0.0
    bomb_bonus_rounds: float = 1500.0

    preload: str = ""

    seed: int = 0
    total_rounds: int = 10_000
    checkpoint_every: int = 500

    @classmethod
    def from_json(cls, path):
        known = {f.name for f in fields(cls)}
        data = json.loads(Path(path).read_text())
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown hyperparameters in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)

    @property
    def forest_depth(self):
        return None if not self.forest_max_depth else int(self.forest_max_depth)


class PrioritisedReplay:
    def __init__(self, capacity, dim, rng, alpha=0.6, eps=1e-3, protected=0):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.rng = rng
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.protected = 0
        self.x = np.zeros((self.capacity, dim), dtype=np.float32)
        self.xn = np.zeros((self.capacity, dim), dtype=np.float32)
        self.a = np.zeros(self.capacity, dtype=np.int8)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=bool)
        self.steps = np.ones(self.capacity, dtype=np.int8)
        self.prio = np.zeros(self.capacity, dtype=np.float64)
        self.size = 0
        self.next = 0
        self.max_prio = 1.0
        self.reserve(protected)

    def reserve(self, count):
        self.protected = min(int(count), self.capacity - 1)
        self.next = max(self.next, self.protected)

    def __len__(self):
        return self.size

    def add(self, x, a, r, xn, done, steps):
        i = self.next
        if i < self.protected:
            i = self.next = self.protected
        self.x[i] = x
        self.xn[i] = 0.0 if xn is None else xn
        self.a[i] = a
        self.r[i] = r
        self.done[i] = done
        self.steps[i] = steps
        self.prio[i] = self.max_prio
        nxt = i + 1
        self.next = self.protected if nxt >= self.capacity else nxt
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        n = self.size
        p = self.prio[:n]
        total = p.sum()
        if total <= 0:
            idx = self.rng.integers(0, n, size=batch_size)
            weights = np.ones(batch_size)
        else:
            cdf = np.cumsum(p)
            idx = np.searchsorted(cdf, self.rng.random(batch_size) * total)
            idx = np.clip(idx, 0, n - 1)
            probs = p[idx] / total
            weights = (n * probs) ** (-beta)
            weights /= weights.max()
        return idx, weights

    def update_priorities(self, idx, td_errors):
        p = (np.abs(td_errors) + self.eps) ** self.alpha
        self.prio[idx] = p
        self.max_prio = max(self.max_prio, float(p.max()))


class NStepAccumulator:
    def __init__(self, n, gamma):
        self.n = int(n)
        self.gamma = float(gamma)
        self.buf = deque()

    def reset(self):
        self.buf.clear()

    def push(self, x, a, r):
        self.buf.append((x, a, r))
        if len(self.buf) < self.n + 1:
            return None
        return self._emit(len(self.buf) - 1, done=False)

    def flush(self, done=True):
        out = []
        while self.buf:
            out.append(self._emit(len(self.buf) - 1, done=done, drain=True))
            self.buf.popleft()
        return out

    def _emit(self, horizon, done, drain=False):
        x0, a0, _ = self.buf[0]
        ret, discount = 0.0, 1.0
        for i in range(horizon):
            ret += discount * self.buf[i][2]
            discount *= self.gamma
        if done:
            ret += discount * self.buf[horizon][2]
            bootstrap, is_terminal, steps = None, True, horizon + 1
        else:
            bootstrap, is_terminal, steps = self.buf[horizon][0], False, horizon
        if not drain:
            self.buf.popleft()
        return x0, a0, ret, bootstrap, is_terminal, steps


class Learner:
    def __init__(self, model, hyper: Hyper, rng=None):
        self.model = model
        self.h = hyper
        self.rng = np.random.default_rng(hyper.seed if rng is None else rng)
        self.buffer = PrioritisedReplay(hyper.buffer_size, model.dim, self.rng,
                                        alpha=hyper.per_alpha, eps=hyper.per_eps)
        self.accumulator = NStepAccumulator(hyper.n_step, hyper.gamma)

        self.target = model.copy() if hyper.target_sync else model

        self.is_gradient = model.kind == "linear"
        self._m = np.zeros_like(model.theta) if self.is_gradient else None
        self._v = np.zeros_like(model.theta) if self.is_gradient else None
        self._t = 0
        self.updates = 0
        self.rounds = 0
        self.env_steps = 0
        self.last_loss = float("nan")

        k = max(1, min(8, int(hyper.augment)))
        self._transforms = list(range(8))[:k]
        self._perms = [F.feature_permutation(t, hyper.feature_set) for t in self._transforms]
        self._action_perms = [g.D4_ACTION_PERM[t] for t in self._transforms]

    @property
    def epsilon(self):
        h = self.h
        decay = np.exp(-self.rounds / max(h.eps_decay_rounds, 1e-9))
        return h.eps_end + (h.eps_start - h.eps_end) * decay

    @property
    def temperature(self):
        h = self.h
        decay = np.exp(-self.rounds / max(h.eps_decay_rounds, 1e-9))
        return h.temperature_end + (h.temperature - h.temperature_end) * decay

    @property
    def bomb_bonus(self):
        h = self.h
        if not h.bomb_bonus:
            return 0.0
        return h.bomb_bonus * float(np.exp(-self.rounds / max(h.bomb_bonus_rounds, 1e-9)))

    @property
    def beta(self):
        h = self.h
        frac = min(1.0, self.rounds / max(h.total_rounds, 1))
        return h.per_beta + (h.per_beta_end - h.per_beta) * frac

    def select(self, q_values, rng, info=None):
        from . import model as M

        if self.h.policy == "boltzmann":
            return M.boltzmann(q_values, self.temperature, rng)
        if not (self.h.safe_exploration and info is not None):
            return M.epsilon_greedy(q_values, self.epsilon, rng)
        if rng.random() >= self.epsilon:
            return M.greedy(q_values, rng)

        threat_on_the_board = bool((info.danger < g.UNREACHABLE).any())
        if threat_on_the_board:
            allowed = [a for a in range(5) if info.escape_mask & (1 << a)]
        else:
            allowed = list(range(5))
        if info.bombs_left and info.escape_bomb:
            allowed.append(5)
        if not allowed:
            return M.greedy(q_values, rng)
        return int(allowed[rng.integers(len(allowed))])

    def preload(self, path, logger=None):
        with np.load(Path(path), allow_pickle=False) as handle:
            stored = str(handle["feature_set"]) if "feature_set" in handle else self.h.feature_set
            if stored != self.h.feature_set:
                raise ValueError(f"demonstrations use feature set {stored!r}, "
                                 f"but this run uses {self.h.feature_set!r}")
            data = {key: handle[key] for key in ("x", "a", "r", "xn", "done", "steps")}

        n = len(data["x"])
        room = self.buffer.capacity // 2

        take = min(n, room)
        rows = self.rng.choice(n, size=take, replace=False) if take < n else np.arange(n)

        self.buffer.protected = 0
        self.buffer.next = 0
        for i in rows:
            self.buffer.add(data["x"][i], int(data["a"][i]), float(data["r"][i]),
                            None if data["done"][i] else data["xn"][i],
                            bool(data["done"][i]), int(data["steps"][i]))
        self.buffer.reserve(take)
        if logger:
            logger.info(f"preloaded {take} distinct demonstration transitions from "
                        f"{Path(path).name} (of {n} recorded, unaugmented for coverage)")
        return take

    def observe(self, x, a, r):
        self.env_steps += 1
        emitted = self.accumulator.push(x, a, r)
        if emitted is not None:
            self._store(*emitted)

    def end_episode(self):
        for tr in self.accumulator.flush(done=True):
            if tr[0] is not None:
                self._store(*tr)
        self.accumulator.reset()
        self.rounds += 1

    def _store(self, x, a, r, xn, terminal, steps):
        for perm, aperm in zip(self._perms, self._action_perms):
            self.buffer.add(x[perm], aperm[a], r,
                            None if xn is None else xn[perm], terminal, steps)

    def maybe_learn(self):
        h = self.h
        if not self.is_gradient:
            return
        if len(self.buffer) < max(h.warmup, h.batch_size):
            return
        if self.env_steps % h.train_every:
            return
        for _ in range(h.updates_per_step):
            self.learn_step()
        if h.refit_every and self.updates % h.refit_every == 0:
            self.ridge_refit()

    def learn_step(self):
        h = self.h
        buf = self.buffer
        idx, weights = buf.sample(h.batch_size, self.beta)

        X = buf.x[idx]
        A = buf.a[idx].astype(np.intp)
        R = buf.r[idx].astype(np.float64)
        XN = buf.xn[idx]
        done = buf.done[idx]
        steps = buf.steps[idx].astype(np.float64)

        if h.double and h.target_sync:
            best = self.model.q_batch(XN).argmax(axis=1)
            q_next = np.take_along_axis(self.target.q_batch(XN), best[:, None], axis=1)[:, 0]
        else:
            q_next = self.target.q_batch(XN).max(axis=1)
        target = R + np.where(done, 0.0, (h.gamma ** steps) * q_next)

        pred = np.einsum("ij,ij->i", self.model.theta[A], X)
        td = target - pred

        grad = np.zeros_like(self.model.theta)
        np.add.at(grad, A, (-(weights * td))[:, None] * X)
        grad /= len(idx)
        norm = np.linalg.norm(grad)
        if h.grad_clip and norm > h.grad_clip:
            grad *= h.grad_clip / norm

        self._adam(grad, h.lr)
        buf.update_priorities(idx, td)
        self.last_loss = float(np.mean(td ** 2))
        self.updates += 1
        if h.target_sync and self.updates % h.target_sync == 0:
            self.target.theta = self.model.theta.copy()

    def _adam(self, grad, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        self._m = b1 * self._m + (1 - b1) * grad
        self._v = b2 * self._v + (1 - b2) * grad ** 2
        mhat = self._m / (1 - b1 ** self._t)
        vhat = self._v / (1 - b2 ** self._t)
        self.model.theta -= lr * mhat / (np.sqrt(vhat) + eps)

    def due_for_fqi(self):
        h = self.h
        return (not self.is_gradient) and h.fqi_every and self.rounds % h.fqi_every == 0 \
            and len(self.buffer) >= max(h.warmup, h.batch_size)

    def fit_batch(self, logger=None):
        h = self.h
        n = len(self.buffer)
        buf = self.buffer
        rows = np.arange(n)
        if h.fqi_max_samples and n > h.fqi_max_samples:
            rows = self.rng.choice(n, size=h.fqi_max_samples, replace=False)

        X = buf.x[rows]
        XN = buf.xn[rows]
        A = buf.a[rows].astype(np.intp)
        R = buf.r[rows].astype(np.float64)
        done = buf.done[rows]
        discount = h.gamma ** buf.steps[rows].astype(np.float64)

        previous = self.model.copy()
        for sweep in range(max(1, h.fqi_iterations)):
            q_next = previous.q_batch(XN).max(axis=1) if previous.is_fitted else np.zeros(len(rows))
            targets = R + np.where(done, 0.0, discount * q_next)
            counts = self.model.fit_heads(X, A, targets)
            previous = self.model.copy()
            if logger:
                logger.info(f"FQI sweep {sweep + 1}/{h.fqi_iterations} on {len(rows)} rows, "
                            f"per-action {counts}, target mean {targets.mean():.3f}")
        self.updates += 1
        self.last_loss = float(np.mean((self.model.q_batch(X)[np.arange(len(rows)), A] - targets) ** 2))

    def ridge_refit(self):
        h = self.h
        n = len(self.buffer)
        if n < h.batch_size:
            return
        buf = self.buffer
        X, A = buf.x[:n], buf.a[:n].astype(np.intp)
        q_next = self.target.q_batch(buf.xn[:n]).max(axis=1)
        target = buf.r[:n] + np.where(buf.done[:n], 0.0,
                                      (h.gamma ** buf.steps[:n].astype(np.float64)) * q_next)
        eye = np.eye(self.model.dim)
        for a in range(self.model.n_actions):
            rows = A == a
            if rows.sum() < self.model.dim:
                continue
            Xa = X[rows].astype(np.float64)
            self.model.theta[a] = np.linalg.solve(Xa.T @ Xa + h.ridge_lambda * eye, Xa.T @ target[rows])
