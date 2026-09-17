from pathlib import Path

import numpy as np

from . import features as F
from . import game_utils as g

MODEL_DIR = Path(__file__).resolve().parent / "models"


class LinearQ:
    kind = "linear"

    def __init__(self, feature_set=F.DEFAULT_SET, n_actions=g.N_ACTIONS, init_scale=0.0, rng=None):
        self.feature_set = feature_set
        self.n_actions = n_actions
        self.dim = F.feature_dim(feature_set)
        rng = np.random.default_rng(rng)
        self.theta = rng.normal(0.0, init_scale, size=(n_actions, self.dim)) if init_scale else \
            np.zeros((n_actions, self.dim))

    def q(self, x):
        return self.theta @ x

    def q_batch(self, X):
        return X @ self.theta.T

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, theta=self.theta, feature_set=self.feature_set,
                            feature_version=F.FEATURE_VERSION, kind=self.kind)

    @classmethod
    def load(cls, path):
        data = np.load(Path(path), allow_pickle=False)
        feature_set = str(data["feature_set"])
        model = cls(feature_set=feature_set, n_actions=data["theta"].shape[0])
        model.theta = data["theta"]
        if model.theta.shape[1] != F.feature_dim(feature_set):
            raise ValueError(f"model was trained on a different feature map "
                             f"({model.theta.shape[1]} != {F.feature_dim(feature_set)})")
        return model

    def copy(self):
        clone = LinearQ(self.feature_set, self.n_actions)
        clone.theta = self.theta.copy()
        return clone

    @property
    def is_fitted(self):
        return True


def greedy(q_values, rng=None):
    best = np.flatnonzero(q_values == q_values.max())
    if len(best) == 1:
        return int(best[0])
    rng = rng if rng is not None else np.random
    return int(rng.choice(best))


def epsilon_greedy(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return greedy(q_values, rng)


def boltzmann(q_values, temperature, rng):
    if temperature <= 1e-8:
        return greedy(q_values, rng)
    z = (q_values - q_values.max()) / temperature
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(q_values), p=p))


def build(hyper):
    if hyper.model == "linear":
        return LinearQ(feature_set=hyper.feature_set)
    if hyper.model == "forest":
        from .model_forest import ForestQ

        return ForestQ(feature_set=hyper.feature_set, n_estimators=hyper.forest_trees,
                       max_depth=hyper.forest_depth, min_samples_leaf=hyper.forest_min_leaf,
                       max_features=hyper.forest_max_features, random_state=hyper.seed)
    raise ValueError(f"unknown model family {hyper.model!r}")


def load_any(path):
    path = Path(path)
    if path.suffix == ".joblib":
        from .model_forest import ForestQ

        return ForestQ.load(path)
    data = np.load(path, allow_pickle=False)
    kind = str(data["kind"]) if "kind" in data else "linear"
    if kind == "linear":
        return LinearQ.load(path)
    raise ValueError(f"unknown model kind {kind!r} in {path}")


def default_checkpoint(directory):
    for name in ("model.npz", "model.joblib"):
        candidate = Path(directory) / name
        if candidate.is_file():
            return candidate
    return Path(directory) / "model.npz"
