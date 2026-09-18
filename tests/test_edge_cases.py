import numpy as np
import pytest

import settings as s
from items import Bomb
from training.env_runner import make_world, quiet_logging

from agent_code.alphabomb import features as F
from agent_code.alphabomb import game_utils as g


def blank_world(scenario="empty", seed=0):
    world = make_world(["peaceful_agent"], scenario=scenario, seed=seed)
    with quiet_logging():
        world.new_round()
    world.user_input = None
    return world


def state_of(world):
    return world.get_state_for_agent(world.agents[0])


def test_empty_board_no_coins_no_crates_no_opponents():
    world = blank_world()
    x = F.state_to_features(state_of(world))
    assert np.isfinite(x).all() and (x <= 1.0).all() and (x >= 0.0).all()
    names = F.feature_names()
    assert x[names.index("coin_none")] == 1.0
    assert x[names.index("crate_none")] == 1.0
    assert x[names.index("enemy_none")] == 1.0


def test_board_with_every_crate_removed_mid_round():
    world = blank_world(scenario="classic", seed=2)
    world.arena[world.arena == 1] = 0
    for coin in world.coins:
        coin.collectable = True
    x = F.state_to_features(state_of(world))
    assert np.isfinite(x).all()
    assert x[F.feature_names().index("crate_none")] == 1.0


def test_agent_standing_on_its_own_bomb():
    world = blank_world()
    agent = world.agents[0]
    agent.x, agent.y = 1, 1
    agent.bombs_left = False
    world.bombs.append(Bomb((1, 1), agent, s.BOMB_TIMER - 1, s.BOMB_POWER, agent.bomb_sprite))
    info = F.analyse(state_of(world))
    assert info.escape_bomb is False, "no second bomb is available"
    assert info.danger[1, 1] < g.UNREACHABLE, "the tile the agent stands on is doomed"
    assert np.isfinite(F.features_from(info)).all()


def test_completely_enclosed_agent():
    world = blank_world()
    agent = world.agents[0]
    agent.x, agent.y = 1, 1
    world.arena[2, 1] = 1
    world.arena[1, 2] = 1
    info = F.analyse(state_of(world))
    x = F.features_from(info)
    assert np.isfinite(x).all()
    assert x[F.feature_names().index("nb_free_RIGHT")] == 0.0
    assert x[F.feature_names().index("nb_free_DOWN")] == 0.0


def test_three_opponents_adjacent():
    world = make_world(["peaceful_agent"] * 4, scenario="empty", seed=0)
    with quiet_logging():
        world.new_round()
    world.user_input = None
    agent = world.agents[0]
    agent.x, agent.y = 3, 3
    for other, (dx, dy) in zip(world.agents[1:], g.DELTAS):
        other.x, other.y = 3 + dx, 3 + dy
    info = F.analyse(state_of(world))
    x = F.features_from(info)
    assert np.isfinite(x).all()
    assert info.enemy_dist == 1
    assert x[F.feature_names().index("bomb_hits_enemy")] == 1.0


@pytest.mark.parametrize("countdown", range(s.BOMB_TIMER + 1))
def test_every_bomb_countdown_is_handled(countdown):
    world = blank_world()
    agent = world.agents[0]
    agent.x, agent.y = 5, 5
    world.bombs.append(Bomb((5, 7), agent, countdown, s.BOMB_POWER, agent.bomb_sprite))
    x = F.state_to_features(state_of(world))
    assert np.isfinite(x).all() and (x <= 1.0).all()


def test_many_simultaneous_bombs():
    world = blank_world(scenario="classic", seed=4)
    agent = world.agents[0]
    free = np.argwhere(world.arena == 0)
    for (bx, by) in free[:12]:
        world.bombs.append(Bomb((int(bx), int(by)), agent, 1, s.BOMB_POWER, agent.bomb_sprite))
    info = F.analyse(state_of(world))
    assert np.isfinite(F.features_from(info)).all()
    assert info.escape_mask == 0 or info.escape_mask > 0


def test_act_never_raises_on_harvested_states():
    world = make_world([("alphabomb", False), "rule_based_agent", "rule_based_agent",
                        "rule_based_agent"], scenario="classic", seed=9)
    runner = world.agents[0].backend.runner
    fake_self, callbacks = runner.fake_self, runner.callbacks
    seen = 0
    with quiet_logging():
        for _ in range(3):
            world.new_round()
            while world.running:
                world.do_step()
                for a in world.active_agents:
                    state = world.get_state_for_agent(a)
                    if state is None:
                        continue
                    assert callbacks.act(fake_self, state) in g.ACTIONS
                    seen += 1
        world.end()
    assert seen > 300
    log = (world.agents[0].code_name and
           (world.agents[0].backend.runner.fake_self.agent_dir / "logs" / "alphabomb.log"))
    assert "falling back to a safe move" not in log.read_text()


def test_a_fresh_world_can_be_inspected_before_the_first_step():
    world = make_world(["peaceful_agent"], scenario="classic", seed=1)
    with quiet_logging():
        world.new_round()
    state = world.get_state_for_agent(world.agents[0])
    assert state is not None and "user_input" in state
