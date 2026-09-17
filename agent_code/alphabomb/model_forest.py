from pathlib import Path

import numpy as np

from . import features as F
from . import game_utils as g


class ForestQ:
    kind = "forest"

    def __init__(self, feature_set=F.DEFAULT_SET, n_actions=g.N_ACTIONS,
                 n_estimators=40, max_depth=16, min_samples_leaf=20,
                 max_features=1.0, n_jobs=1, random_state=0):
        self.feature_set = feature_set
        self.n_actions = n_actions
        self.dim = F.feature_dim(feature_set)
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf, max_features=max_features,
                           n_jobs=n_jobs, random_state=random_state)
        self.forests = [None] * n_actions
        self.fits = 0
        self._trees = None

    def _build_fast_path(self):
        self._trees = [[e.tree_ for e in f.estimators_] if f is not None else None
                       for f in self.forests]

    def q(self, x):
        if self._trees is None:
            self._build_fast_path()
        row = np.ascontiguousarray(x.reshape(1, -1), dtype=np.float32)
        out = np.zeros(self.n_actions)
        for a, trees in enumerate(self._trees):
            if trees:
                out[a] = sum(float(t.predict(row)[0, 0]) for t in trees) / len(trees)
        return out

    def q_batch(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros((len(X), self.n_actions))
        if len(X) == 0:
            return out
        for a, forest in enumerate(self.forests):
            if forest is not None:
                out[:, a] = forest.predict(X)
        return out

    @property
    def is_fitted(self):
        return any(f is not None for f in self.forests)

    def fit_heads(self, X, actions, targets, min_rows=64):
        from sklearn.ensemble import ExtraTreesRegressor

        fitted = []
        for a in range(self.n_actions):
            rows = actions == a
            count = int(rows.sum())
            if count < min_rows:
                fitted.append(count)
                continue
            forest = ExtraTreesRegressor(**self.params)
            forest.fit(X[rows], targets[rows])
            self.forests[a] = forest
            fitted.append(count)
        self.fits += 1
        self._trees = None
        return fitted

    def feature_importances(self):
        fitted = [f for f in self.forests if f is not None]
        if not fitted:
            return np.zeros(self.dim)
        return np.mean([f.feature_importances_ for f in fitted], axis=0)

    def save(self, path):
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": self.kind, "feature_set": self.feature_set,
                     "feature_version": F.FEATURE_VERSION, "n_actions": self.n_actions,
                     "params": self.params, "forests": self.forests, "fits": self.fits},
                    path, compress=3)

    @classmethod
    def load(cls, path):
        import joblib

        data = joblib.load(Path(path))
        model = cls(feature_set=data["feature_set"], n_actions=data["n_actions"], **data["params"])
        model.forests = data["forests"]
        model.fits = data.get("fits", 0)
        model._trees = None
        if model.dim != F.feature_dim(model.feature_set):
            raise ValueError("model was trained on a different feature map")
        return model

    def copy(self):
        clone = ForestQ(self.feature_set, self.n_actions, **self.params)
        clone.forests = list(self.forests)
        clone.fits = self.fits
        return clone

    @property
    def theta(self):
        raise AttributeError("ForestQ has no weight matrix; it is fitted in batch")
