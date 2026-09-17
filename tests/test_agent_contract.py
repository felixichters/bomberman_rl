import ast
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = REPO_ROOT / "agent_code" / "alphabomb"
PYTHON = sys.executable

FRAMEWORK = ("main.py", "environment.py", "agents.py", "items.py", "events.py",
             "settings.py", "fallbacks.py", "replay.py")
UPSTREAM = "cea0005"

ALLOWED_TOP_LEVEL = {
    "numpy", "sklearn", "joblib", "scipy",
    "settings", "events",
    "os", "sys", "json", "csv", "re", "math", "time", "pathlib", "collections",
    "dataclasses", "typing", "itertools", "functools", "random", "pickle", "shutil",
    "copy", "logging", "warnings", "contextlib", "abc",
}


def agent_sources():
    return sorted(AGENT_DIR.glob("*.py"))


def test_framework_is_unmodified():
    diff = subprocess.run(["git", "diff", "--name-only", UPSTREAM, "--", *FRAMEWORK],
                          cwd=REPO_ROOT, text=True, capture_output=True).stdout.strip()
    assert diff == "", f"framework files differ from upstream: {diff}"


def test_agent_imports_only_what_it_ships_with():
    offenders = []
    for path in agent_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in ALLOWED_TOP_LEVEL:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                top = (node.module or "").split(".")[0]
                if top not in ALLOWED_TOP_LEVEL:
                    offenders.append(f"{path.name}: from {node.module} import ...")
    assert not offenders, "\n".join(offenders)


def test_no_multiprocessing_in_the_agent():
    for path in agent_sources():
        text = path.read_text()
        for banned in ("multiprocessing", "concurrent.futures", "threading"):
            assert banned not in text, f"{path.name} mentions {banned}"


def test_no_absolute_paths():
    for path in agent_sources():
        text = path.read_text()
        assert "/Users/" not in text and "C:\\" not in text, path.name
        assert "os.getcwd" not in text, f"{path.name} depends on the callback cwd"


def test_requirements_are_declared():
    listed = {line.strip().split("=")[0].split(">")[0].lower()
              for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")}
    assert "numpy" in listed


def build_clean_framework(tmp_path):
    root = tmp_path / "clean"
    (root / "agent_code").mkdir(parents=True)
    for name in FRAMEWORK:
        blob = subprocess.run(["git", "show", f"{UPSTREAM}:{name}"], cwd=REPO_ROOT,
                              capture_output=True, check=True).stdout
        (root / name).write_bytes(blob)
    shutil.copytree(REPO_ROOT / "assets", root / "assets")
    for extra in ("random_agent", "rule_based_agent"):
        shutil.copytree(REPO_ROOT / "agent_code" / extra, root / "agent_code" / extra)
    shutil.copytree(AGENT_DIR, root / "agent_code" / "alphabomb",
                    ignore=shutil.ignore_patterns("__pycache__", "logs", "*.log"))
    (root / "logs").mkdir(exist_ok=True)
    return root


@pytest.mark.slow
def test_plays_in_a_clean_framework(tmp_path):
    root = build_clean_framework(tmp_path)
    result = subprocess.run(
        [str(PYTHON), "main.py", "play", "--no-gui", "--n-rounds", "3",
         "--agents", "alphabomb", "random_agent", "random_agent", "random_agent"],
        cwd=root, capture_output=True, text=True, timeout=600,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr, result.stderr

    game_log = (root / "logs" / "game.log").read_text()
    assert "STARTING ROUND #3" in game_log
    agent_log = (root / "agent_code" / "alphabomb" / "logs" / "alphabomb.log").read_text()
    for bad in ("Traceback", "falling back to a safe move", "ERROR"):
        assert bad not in agent_log, f"agent log contains {bad!r}"


def test_act_is_well_inside_the_time_budget():
    from training.env_runner import make_world, quiet_logging

    world = make_world([("alphabomb", False), "rule_based_agent",
                        "rule_based_agent", "rule_based_agent"], scenario="classic", seed=5)
    agent = world.agents[0]
    fake_self = agent.backend.runner.fake_self
    callbacks = agent.backend.runner.callbacks

    states = []
    with quiet_logging():
        for _ in range(3):
            world.new_round()
            while world.running:
                world.do_step()
                for a in world.active_agents:
                    states.append(world.get_state_for_agent(a))
        world.end()
    states = [s for s in states if s is not None][:2000]
    assert len(states) > 500

    timings = np.empty(len(states))
    for i, state in enumerate(states):
        fake_self._analysis_cache.clear()
        start = time.perf_counter()
        callbacks.act(fake_self, state)
        timings[i] = time.perf_counter() - start

    p99, worst = np.quantile(timings, 0.99), timings.max()
    print(f"\nact(): mean {timings.mean() * 1e3:.2f} ms, p99 {p99 * 1e3:.2f} ms, max {worst * 1e3:.2f} ms")
    assert p99 < 0.05, f"p99 {p99 * 1e3:.1f} ms leaves too little headroom"
    assert worst < 0.2, f"worst case {worst * 1e3:.1f} ms"


def test_forest_fast_path_matches_sklearn():
    import numpy as np

    from agent_code.alphabomb.model_forest import ForestQ

    rng = np.random.default_rng(0)
    X = rng.random((4000, ForestQ().dim)).astype(np.float32)
    A = rng.integers(0, 6, size=4000)
    y = rng.normal(size=4000)
    model = ForestQ(n_estimators=12, max_depth=8, min_samples_leaf=20)
    model.fit_heads(X, A, y)

    for row in X[:40]:
        fast = model.q(row)
        slow = np.array([f.predict(row.reshape(1, -1))[0] if f is not None else 0.0
                         for f in model.forests])
        assert np.allclose(fast, slow, atol=1e-9)


def test_shipped_config_matches_the_schema():
    import json

    from agent_code.alphabomb.learning import Hyper

    shipped = json.loads((AGENT_DIR / "models" / "config.json").read_text())
    expected = set(Hyper().to_dict())
    missing = sorted(expected - set(shipped))
    extra = sorted(set(shipped) - expected)
    assert not missing, f"config.json is missing {missing}"
    assert not extra, f"config.json has unknown keys {extra}"
    Hyper(**shipped)


def test_shipped_model_matches_the_feature_map():
    from agent_code.alphabomb import features as F
    from agent_code.alphabomb import model as M

    checkpoint = M.default_checkpoint(AGENT_DIR / "models")
    if not checkpoint.is_file():
        pytest.skip("no trained model in the agent directory")
    model = M.load_any(checkpoint)
    assert model.dim == F.feature_dim(model.feature_set)
    assert model.feature_set in F.FEATURE_SETS
