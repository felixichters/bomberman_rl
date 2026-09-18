import csv
import json
import os
import re
import shutil
from collections import deque
from pathlib import Path
from typing import List

import numpy as np

import events as e

from . import model as M
from . import rewards as R
from .callbacks import analysed, resolve_path
from .learning import Hyper, Learner

METRIC_COLUMNS = [
    "round", "steps", "score", "coins", "crates", "kills", "suicide", "got_killed",
    "survived", "invalid", "bombs", "waited", "epsilon", "loss", "buffer",
    "reward_sum", "updates",
] + list(R.CUSTOM_EVENTS)


def _run_dir():
    default = Path(__file__).resolve().parents[2] / "runs" / "manual"
    return resolve_path(os.environ.get("ALPHABOMB_RUN_DIR"), default)


def _is_primary(name):
    match = re.fullmatch(r".*_(\d+)", name or "")
    return match is None or int(match.group(1)) == 0


def setup_training(self):
    self.hyper = Hyper.from_json(self.config_path) if self.config_path.is_file() else Hyper()
    if self.model.feature_set != self.hyper.feature_set or self.model.kind != self.hyper.model:
        self.logger.warning(f"checkpoint is {self.model.kind}/{self.model.feature_set!r} but the config "
                            f"asks for {self.hyper.model}/{self.hyper.feature_set!r}; starting fresh")
        self.model = M.build(self.hyper)
        self.feature_set = self.hyper.feature_set

    self.learner = Learner(self.model, self.hyper)
    self.event_table = R.event_table(self.hyper.reward_scheme, self.hyper)
    self.shaped = R.scheme_is_shaped(self.hyper.reward_scheme)

    self.run_dir = _run_dir()
    self.run_dir.mkdir(parents=True, exist_ok=True)
    (self.run_dir / "config.json").write_text(json.dumps(self.hyper.to_dict(), indent=2))
    self.metrics_path = self.run_dir / "metrics.csv"
    self.primary = True
    if not self.metrics_path.exists():
        with self.metrics_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(METRIC_COLUMNS)

    preload = self.hyper.preload or os.environ.get("ALPHABOMB_PRELOAD")
    if preload:
        path = resolve_path(preload)
        if path.is_file():
            self.learner.preload(path, logger=self.logger)
        else:
            self.logger.warning(f"ALPHABOMB_PRELOAD={preload} does not exist; ignoring")

    self.history = deque(maxlen=8)
    self.round_events = []
    self.reward_sum = 0.0
    self.logger.info(f"Training with {self.hyper.to_dict()}")


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action is None:
        return

    old_info, x_old = analysed(self, old_game_state)
    new_info, _ = analysed(self, new_game_state)

    events = list(events) + R.derive_events(old_info, self_action, new_info, events, self.history)
    self.round_events.extend(events)
    self.history.append(old_info.pos)

    reward = R.reward_from(events, self.event_table, self.hyper,
                           old_info=old_info, new_info=new_info, shaped=self.shaped,
                           bomb_bonus=self.learner.bomb_bonus)
    self.reward_sum += reward

    self.learner.observe(x_old, learner_action_index(self, self_action), reward)
    self.learner.maybe_learn()


def learner_action_index(self, action):
    from . import game_utils as g

    return g.ACTION_INDEX.get(action, g.ACTION_INDEX["WAIT"])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state is not None and last_action is not None:
        old_info, x_old = analysed(self, last_game_state)
        events = list(events) + R.derive_events(old_info, last_action, None, events, self.history)
        self.round_events.extend(events)
        reward = R.reward_from(events, self.event_table, self.hyper,
                               old_info=old_info, new_info=None, shaped=self.shaped,
                               bomb_bonus=self.learner.bomb_bonus)
        self.reward_sum += reward
        self.learner.observe(x_old, learner_action_index(self, last_action), reward)

    self.learner.end_episode()
    self.primary = _is_primary(last_game_state["self"][0] if last_game_state else None)

    if self.learner.due_for_fqi():
        self.learner.fit_batch(logger=self.logger)

    _write_metrics(self, last_game_state)

    rounds = self.learner.rounds
    if self.primary and self.hyper.checkpoint_every and rounds % self.hyper.checkpoint_every == 0:
        ckpt = self.run_dir / "checkpoints" / f"round_{rounds:06d}{_suffix(self)}"
        self.model.save(ckpt)
        self.logger.info(f"checkpoint {ckpt.name}")
    if self.primary:
        live = self.run_dir / f"model{_suffix(self)}"
        self.model.save(live)
        target = self.model_path.with_suffix(_suffix(self))
        if target.resolve() != live.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(live, target)

    self.history.clear()
    self.round_events = []
    self.reward_sum = 0.0


def _suffix(self):
    return ".joblib" if self.model.kind == "forest" else ".npz"


def _write_metrics(self, last_game_state):
    counted = self.round_events
    row = {
        "round": self.learner.rounds,
        "steps": last_game_state["step"] if last_game_state else 0,
        "score": last_game_state["self"][1] if last_game_state else 0,
        "coins": counted.count(e.COIN_COLLECTED),
        "crates": counted.count(e.CRATE_DESTROYED),
        "kills": counted.count(e.KILLED_OPPONENT),
        "suicide": int(e.KILLED_SELF in counted),
        "got_killed": int(e.GOT_KILLED in counted and e.KILLED_SELF not in counted),
        "survived": int(e.SURVIVED_ROUND in counted),
        "invalid": counted.count(e.INVALID_ACTION),
        "bombs": counted.count(e.BOMB_DROPPED),
        "waited": counted.count(e.WAITED),
        "epsilon": round(self.learner.epsilon, 4),
        "loss": round(self.learner.last_loss, 5) if np.isfinite(self.learner.last_loss) else "",
        "buffer": len(self.learner.buffer),
        "reward_sum": round(self.reward_sum, 3),
        "updates": self.learner.updates,
    }
    for ev in R.CUSTOM_EVENTS:
        row[ev] = counted.count(ev)
    if not self.primary:
        return
    with self.metrics_path.open("a", newline="") as fh:
        csv.writer(fh).writerow([row[c] for c in METRIC_COLUMNS])
