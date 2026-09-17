import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import settings as s
from environment import BombeRLeWorld, WorldArgs
from training import scenarios

DEFAULT_LOG_DIR = REPO_ROOT / "logs"

STAT_KEYS = ("score", "coins", "kills", "suicides", "crates", "bombs", "moves", "invalid", "steps", "time")


def make_args(scenario="classic", seed=None, log_dir=None, match_name=None,
              continue_without_training=False, silence_errors=False):
    log_dir = Path(log_dir or DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    return WorldArgs(
        no_gui=True, fps=None, turn_based=False, update_interval=None,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=continue_without_training,
        log_dir=str(log_dir), save_stats=False, match_name=match_name,
        seed=seed, silence_errors=silence_errors, scenario=scenario,
    )


@contextmanager
def quiet_logging(level=logging.WARNING):
    saved = {}
    for name in ("BombeRLeWorld",):
        lg = logging.getLogger(name)
        saved[name] = lg.level
        lg.setLevel(level)
    prev_game, prev_wrapper, prev_code = s.LOG_GAME, s.LOG_AGENT_WRAPPER, s.LOG_AGENT_CODE
    s.LOG_GAME = s.LOG_AGENT_WRAPPER = s.LOG_AGENT_CODE = level
    try:
        yield
    finally:
        s.LOG_GAME, s.LOG_AGENT_WRAPPER, s.LOG_AGENT_CODE = prev_game, prev_wrapper, prev_code
        for name, lvl in saved.items():
            logging.getLogger(name).setLevel(lvl)


def make_world(agent_specs, scenario="classic", seed=None, **kwargs):
    specs = [(a, False) if isinstance(a, str) else tuple(a) for a in agent_specs]
    args = make_args(scenario=scenario, seed=seed, **kwargs)
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        world = BombeRLeWorld(args, specs)
    finally:
        os.chdir(cwd)
    world.user_input = None  # only do_step sets it, but get_state_for_agent reads it
    return world


def round_record(world):
    record = {"steps": world.step}
    for a in world.agents:
        stats = {k: a.statistics.get(k, 0) for k in STAT_KEYS}
        stats["survived"] = not a.dead
        record[a.name] = stats
    return record


def play_round(world, seed=None):
    if seed is not None:
        world.rng = np.random.default_rng(seed)
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        world.new_round()
        while world.running:
            world.do_step()
    finally:
        os.chdir(cwd)
    return round_record(world)


def run_rounds(agent_specs, n_rounds, scenario="classic", seeds=None, quiet=True, **kwargs):
    ctx = quiet_logging() if quiet else _null_context()
    with ctx:
        world = make_world(agent_specs, scenario=scenario,
                           seed=None if seeds else kwargs.pop("seed", None), **kwargs)
        records = []
        for i in range(n_rounds):
            records.append(play_round(world, seed=None if seeds is None else seeds[i % len(seeds)]))
        cwd = os.getcwd()
        os.chdir(REPO_ROOT)
        try:
            world.end()
        finally:
            os.chdir(cwd)
    return records


@contextmanager
def _null_context():
    yield
