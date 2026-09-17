import os
from pathlib import Path

import numpy as np

from agent_code.alphabomb import features as F
from agent_code.alphabomb import game_utils as g
from agent_code.alphabomb import model as M

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIONS = list(g.ACTIONS)


def _pool_dir():
    raw = os.environ.get("ALPHABOMB_POOL", "runs/pool")
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def setup(self):
    self.pool_dir = _pool_dir()
    self.rng = np.random.default_rng()
    self.current_round = -1
    self.model = None
    self.epsilon = float(os.environ.get("ALPHABOMB_POOL_EPSILON", "0.05"))
    self._cache = {}
    _resample(self)


def _resample(self):
    snapshots = sorted(self.pool_dir.glob("*.npz")) if self.pool_dir.is_dir() else []
    if not snapshots:
        self.model = None
        self.logger.warning(f"no snapshots in {self.pool_dir}; playing randomly")
        return
    choice = snapshots[self.rng.integers(len(snapshots))]
    try:
        self.model = M.load_any(choice)
        self.logger.info(f"sparring as {choice.name}")
    except Exception:
        self.logger.exception(f"could not load {choice}")
        self.model = None


def act(self, game_state: dict) -> str:
    if game_state["round"] != self.current_round:
        self.current_round = game_state["round"]
        _resample(self)
    if self.model is None:
        return ACTIONS[self.rng.integers(len(ACTIONS))]
    try:
        if self.rng.random() < self.epsilon:
            return ACTIONS[self.rng.integers(len(ACTIONS))]
        x = F.state_to_features(game_state, self.model.feature_set)
        return ACTIONS[M.greedy(self.model.q(x), self.rng)]
    except Exception:
        self.logger.exception("frozen agent failed; waiting")
        return "WAIT"
