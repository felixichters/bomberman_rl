import numpy as np
import pytest

import settings as s
from items import Bomb
from training.env_runner import make_world, quiet_logging, play_round

from agent_code.alphabomb import features as F
from agent_code.alphabomb import game_utils as g


def sample_states(n_rounds=3, scenario="classic", seed=1, agents=("rule_based_agent", "coin_collector_agent",
                                                                 "peaceful_agent", "random_agent")):
    world = make_world(list(agents), scenario=scenario, seed=seed)
    states = []
    with quiet_logging():
        for r in range(n_rounds):
            world.new_round()
            while world.running:
                world.do_step()
                for a in world.active_agents:
                    states.append(world.get_state_for_agent(a))
        world.end()
    return [st for st in states if st is not None]


@pytest.fixture(scope="module")
def states():
    out = sample_states()
    assert len(out) > 500
    return out


def test_dimension_and_names_agree():
    for name in F.FEATURE_SETS:
        assert len(F.feature_names(name)) == F.feature_dim(name)
    assert F.feature_dim("v2") > F.feature_dim("base")


def test_features_are_finite_and_bounded(states):
    for st in states[:1500]:
        x = F.state_to_features(st)
        assert x.dtype == np.float32
        assert np.isfinite(x).all()
        assert (x >= 0.0).all() and (x <= 1.0).all()


def test_none_state_returns_none():
    assert F.state_to_features(None) is None


def test_features_are_deterministic(states):
    for st in states[:200]:
        assert np.array_equal(F.state_to_features(st), F.state_to_features(st))


@pytest.mark.parametrize("k", range(8))
def test_feature_map_is_d4_equivariant(k, states):
    P = F.feature_permutation(k)
    for st in states[:300]:
        x = F.state_to_features(st)
        xt = F.state_to_features(g.transform_state(st, k))
        assert np.allclose(xt, x[P]), f"symmetry {k} broken"


def test_escape_bomb_feature_is_off_without_a_bomb(states):
    idx = F.feature_names().index("escape_BOMB")
    left = F.feature_names().index("bombs_left")
    for st in states[:400]:
        x = F.state_to_features(st)
        if x[left] == 0.0:
            assert x[idx] == 0.0


def test_suicidal_bomb_is_flagged():
    world = make_world(["peaceful_agent"], scenario="empty", seed=0)
    with quiet_logging():
        world.new_round()
    world.user_input = None
    agent = world.agents[0]
    agent.x, agent.y = 1, 1
    world.arena[2, 1] = 1
    world.arena[1, 2] = 1
    x = F.state_to_features(world.get_state_for_agent(agent))
    assert x[F.feature_names().index("escape_BOMB")] == 0.0

    world.arena[1, 2] = 0
    x = F.state_to_features(world.get_state_for_agent(agent))
    assert x[F.feature_names().index("escape_BOMB")] == 1.0


def test_danger_features_fire_when_a_bomb_is_ticking():
    world = make_world(["peaceful_agent"], scenario="empty", seed=0)
    with quiet_logging():
        world.new_round()
    world.user_input = None
    agent = world.agents[0]
    agent.x, agent.y = 1, 1
    names = F.feature_names()
    assert F.state_to_features(world.get_state_for_agent(agent))[names.index("urgency_HERE")] == 0.0

    world.bombs.append(Bomb((1, 1), agent, 0, s.BOMB_POWER, agent.bomb_sprite))
    x = F.state_to_features(world.get_state_for_agent(agent))
    assert x[names.index("lethal_next_HERE")] == 1.0
    assert x[names.index("urgency_HERE")] == 1.0
    assert x[names.index("escape_WAIT")] == 0.0


def test_feature_map_is_fast(states):
    import time

    sample = states[:400]
    start = time.perf_counter()
    for st in sample:
        F.state_to_features(st)
    per_call = (time.perf_counter() - start) / len(sample)
    assert per_call < 2e-3, f"{per_call * 1e3:.2f} ms per call is too slow"
