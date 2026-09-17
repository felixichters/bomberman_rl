import shutil

import numpy as np

from training.env_runner import make_world, quiet_logging

from agent_code.alphabomb import features as F


def test_old_and_new_states_share_a_step_number():
    world = make_world([("alphabomb", False)], scenario="classic", seed=3)
    with quiet_logging():
        world.new_round()
        world.do_step()
    agent = world.agents[0]
    assert agent.last_game_state["step"] == world.get_state_for_agent(agent)["step"]


def test_analysis_cache_distinguishes_pre_and_post_move_states():
    world = make_world([("alphabomb", False)], scenario="classic", seed=3)
    with quiet_logging():
        world.new_round()
        world.do_step()
    agent = world.agents[0]
    fake_self = agent.backend.runner.fake_self
    callbacks = agent.backend.runner.callbacks

    pre = agent.last_game_state
    post = world.get_state_for_agent(agent)
    assert pre is not post

    _, x_pre = callbacks.analysed(fake_self, pre)
    _, x_post = callbacks.analysed(fake_self, post)
    _, x_pre_again = callbacks.analysed(fake_self, pre)
    assert np.array_equal(x_pre, x_pre_again)
    assert np.array_equal(x_pre, F.state_to_features(pre))
    assert np.array_equal(x_post, F.state_to_features(post))


def test_training_round_fills_the_buffer_and_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHABOMB_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("ALPHABOMB_MODEL", str(tmp_path / "run" / "model.npz"))
    world = make_world([("alphabomb", True)], scenario="coin-heaven", seed=1)
    agent = world.agents[0]
    fake_self = agent.backend.runner.fake_self
    fake_self.hyper.warmup = 50
    fake_self.hyper.checkpoint_every = 2

    with quiet_logging():
        for _ in range(6):
            world.new_round()
            while world.running:
                world.do_step()
        world.end()

    learner = fake_self.learner
    assert learner.rounds == 6
    assert len(learner.buffer) > 100
    assert learner.updates > 0
    assert np.isfinite(fake_self.model.theta).all()
    assert not np.allclose(fake_self.model.theta, 0.0), "the model never moved"

    metrics = (tmp_path / "run" / "metrics.csv").read_text().strip().splitlines()
    assert len(metrics) == 7
    assert (tmp_path / "run" / "model.npz").is_file()
    assert list((tmp_path / "run" / "checkpoints").glob("*.npz"))


def test_a_rerun_does_not_append_to_the_previous_run(tmp_path, monkeypatch):
    import subprocess
    import sys as _sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / f"pytest_rerun_{tmp_path.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        (run_dir / "eval.csv").write_text("round,coins\n1,99\n")
        (run_dir / "metrics.csv").write_text("round,steps\n1,99\n")
        subprocess.run(
            [_sys.executable, str(repo / "training" / "run_training.py"),
             "--stage", "t1", "--name", run_dir.name, "--rounds", "1",
             "--eval-every", "0", "--eval-rounds", "5", "--eval-workers", "1"],
            cwd=repo, capture_output=True, timeout=600)
        archives = list(run_dir.glob("superseded_*"))
        assert archives, "the previous run's records were not archived"
        assert (archives[0] / "eval.csv").read_text().strip().endswith("1,99")
        fresh = (run_dir / "eval.csv").read_text()
        assert "99" not in fresh, "the new run appended to the old records"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
