import json
import os
from pathlib import Path

import numpy as np

from . import features as F
from . import game_utils as g
from . import model as M

ACTIONS = list(g.ACTIONS)

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parents[1]
DEFAULT_MODEL = M.default_checkpoint(AGENT_DIR / "models")
DEFAULT_CONFIG = AGENT_DIR / "models" / "config.json"


def resolve_path(value, default=None):
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _resolve(env_var, default):
    return resolve_path(os.environ.get(env_var), default)


def setup(self):
    self.agent_dir = AGENT_DIR
    self.config_path = _resolve("ALPHABOMB_CONFIG", DEFAULT_CONFIG)
    self.model_path = _resolve("ALPHABOMB_MODEL", DEFAULT_MODEL)
    self.config = json.loads(self.config_path.read_text()) if self.config_path.is_file() else {}
    self.feature_set = self.config.get("feature_set", F.DEFAULT_SET)
    self.rng = np.random.default_rng()

    if self.model_path.is_file():
        self.model = M.load_any(self.model_path)
        self.feature_set = self.model.feature_set
        self.logger.info(f"Loaded {self.model.kind} model from {self.model_path.name} "
                         f"(features {self.feature_set}, dim {self.model.dim})")
    else:
        self.model = M.LinearQ(feature_set=self.feature_set)
        level = self.logger.info if self.train else self.logger.warning
        level(f"No model at {self.model_path}; starting from zero weights.")

    self._analysis_cache = {}


def analysed(self, game_state):
    if game_state is None:
        return None, None
    # identity, not (round, step): pre- and post-move states of a step share both
    key = id(game_state)
    cached = self._analysis_cache.get(key)
    if cached is not None and cached[0] is game_state:
        return cached[1], cached[2]
    info = F.analyse(game_state)
    x = F.features_from(info, self.feature_set)
    if len(self._analysis_cache) > 4:
        self._analysis_cache.clear()
    self._analysis_cache[key] = (game_state, info, x)
    return info, x


def act(self, game_state: dict) -> str:
    try:
        info, x = analysed(self, game_state)
        q = self.model.q(x)
        if getattr(self, "train", False) and hasattr(self, "learner"):
            action_index = self.learner.select(q, self.rng, info)
        else:
            action_index = M.greedy(q, self.rng)
        action = ACTIONS[action_index]
        if getattr(self, "train", False):
            self.logger.debug(f"step {game_state['step']}: {action} "
                              f"Q={np.array2string(q, precision=2)}")
        else:
            self.logger.debug(f"step {game_state['step']}: {action}")
        return action
    except Exception:
        self.logger.exception("act() failed; falling back to a safe move")
        return fallback_action(self, game_state)


def fallback_action(self, game_state):
    try:
        free = g.free_map(game_state)
        lethal = g.lethal_schedule(game_state)
        mask = int(g.escape_masks(free, lethal, game_state["self"][3]))
        for a in range(5):
            if mask & (1 << a):
                return ACTIONS[a]
    except Exception:
        self.logger.exception("fallback_action() failed as well")
    return "WAIT"


def state_to_features(game_state: dict):
    return F.state_to_features(game_state)
