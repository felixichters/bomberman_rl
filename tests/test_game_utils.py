import numpy as np
import pytest

import settings as s
from items import Bomb
from training.env_runner import make_world, quiet_logging

from agent_code.alphabomb import game_utils as g


def fresh_world(scenario="empty", seed=0):
    world = make_world(["peaceful_agent"], scenario=scenario, seed=seed)
    with quiet_logging():
        world.new_round()
    world.user_input = None
    return world


def advance_physics(world):
    world.collect_coins()
    world.update_explosions()
    world.update_bombs()
    world.evaluate_explosions()
    world.step += 1


def test_blast_coords_matches_environment():
    world = fresh_world(scenario="classic", seed=3)
    arena = world.arena
    tested = 0
    for x in range(1, s.COLS - 1):
        for y in range(1, s.ROWS - 1):
            if arena[x, y] == -1:
                continue
            reference = Bomb((x, y), None, s.BOMB_TIMER, s.BOMB_POWER, None)
            assert sorted(g.blast_coords(arena, x, y)) == sorted(reference.get_blast_coords(arena))
            tested += 1
    assert tested > 100


@pytest.mark.parametrize("arrival_offset,should_die", [(0, True), (1, True), (2, False)])
def test_lethal_window_matches_environment(arrival_offset, should_die):
    world = fresh_world()
    agent = world.agents[0]
    bomb_tile = (1, 1)
    safe_tile = (1, 5) if s.BOMB_POWER < 4 else (1, 9)

    agent.x, agent.y = safe_tile
    world.bombs.append(Bomb(bomb_tile, agent, s.BOMB_TIMER, s.BOMB_POWER, agent.bomb_sprite))
    advance_physics(world)

    state = world.get_state_for_agent(agent)
    lethal = g.lethal_schedule(state)
    danger = g.danger_map(lethal)
    countdown = state["bombs"][0][1]
    assert danger[bomb_tile] == countdown + 1, "a blast is first fatal one move after the countdown"

    target_moves = countdown + 1 + arrival_offset
    assert bool(lethal[target_moves][bomb_tile]) is should_die, "schedule disagrees with the rule"

    for move in range(1, target_moves + 1):
        agent.x, agent.y = bomb_tile if move == target_moves else safe_tile
        advance_physics(world)
        if agent.dead:
            break
    assert agent.dead is should_die


def test_standing_still_dies_at_danger_map_value():
    world = fresh_world()
    agent = world.agents[0]
    agent.x, agent.y = (1, 1)
    world.bombs.append(Bomb((1, 1), agent, s.BOMB_TIMER, s.BOMB_POWER, agent.bomb_sprite))
    advance_physics(world)

    state = world.get_state_for_agent(agent)
    predicted = int(g.danger_map(g.lethal_schedule(state))[1, 1])

    moves = 0
    for _ in range(g.HORIZON + 2):
        advance_physics(world)
        moves += 1
        if agent.dead:
            break
    assert moves == predicted, "danger_map counts the moves until the tile kills"


def test_explosion_map_is_lethal_now_and_then_clears():
    world = fresh_world()
    agent = world.agents[0]
    agent.x, agent.y = (1, 5)
    world.bombs.append(Bomb((1, 1), agent, s.BOMB_TIMER, s.BOMB_POWER, agent.bomb_sprite))
    advance_physics(world)

    seen_hot, seen_cold = False, False
    for _ in range(g.HORIZON + 3):
        state = world.get_state_for_agent(agent)
        exploding = state["explosion_map"][1, 1] >= 1
        lethal_next = g.lethal_schedule(state)[1][1, 1]
        if exploding:
            assert lethal_next, "a live explosion must be fatal after the next move"
            seen_hot = True
        elif not state["bombs"]:
            assert not lethal_next
            seen_cold = True
        advance_physics(world)
    assert seen_hot and seen_cold


def brute_force_escape(free, lethal, start, depth):
    frontier = {(start, 0)}
    for _ in range(depth):
        nxt = set()
        for (x, y), t in frontier:
            for dx, dy in g.DELTAS + ((0, 0),):
                nx, ny = x + dx, y + dy
                if (dx or dy) and not free[nx, ny]:
                    continue
                if lethal[t + 1][nx, ny]:
                    continue
                nxt.add(((nx, ny), t + 1))
        frontier = nxt
        if not frontier:
            return False
    return len(frontier) > 0


def test_escape_search_matches_brute_force():
    world = fresh_world(scenario="classic", seed=7)
    agent = world.agents[0]
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(60):
        free_tiles = np.argwhere(world.arena == 0)
        x, y = free_tiles[rng.integers(len(free_tiles))]
        agent.x, agent.y = int(x), int(y)
        world.bombs = []
        for _ in range(int(rng.integers(1, 4))):
            bx, by = free_tiles[rng.integers(len(free_tiles))]
            world.bombs.append(Bomb((int(bx), int(by)), agent, int(rng.integers(0, s.BOMB_TIMER)),
                                    s.BOMB_POWER, agent.bomb_sprite))
        state = world.get_state_for_agent(agent)
        lethal = g.lethal_schedule(state)
        free = g.free_map(state)
        if lethal[0][agent.x, agent.y]:
            continue
        mine, _, _ = g.escape_search(free, lethal, (agent.x, agent.y))
        theirs = brute_force_escape(free, lethal, (agent.x, agent.y), g.HORIZON)
        assert mine == theirs, f"escape disagreement at {(agent.x, agent.y)}"
        checked += 1
    assert checked > 20


def test_can_escape_after_bomb_in_a_corridor():
    world = fresh_world(scenario="empty", seed=1)
    agent = world.agents[0]
    agent.x, agent.y = 1, 1
    state = world.get_state_for_agent(agent)
    assert g.can_escape_after_bomb(state), "an open corner has an escape route"

    world.arena[2, 1] = 1
    world.arena[1, 3] = 1
    state = world.get_state_for_agent(agent)
    assert not g.can_escape_after_bomb(state), "a two-tile pocket inside the blast is a death trap"


def test_d4_transforms_are_consistent():
    for k, (coord, perm) in enumerate(zip(g.D4_COORD, g.D4_ACTION_PERM)):
        for a, (dx, dy) in enumerate(g.DELTAS):
            x, y = 5, 7
            moved = coord(x + dx, y + dy)
            tx, ty = coord(x, y)
            mdx, mdy = g.DELTAS[perm[a]]
            assert moved == (tx + mdx, ty + mdy), f"transform {k} breaks action {g.ACTIONS[a]}"


def test_d4_transforms_are_distinct_and_bijective():
    seen = set()
    for coord in g.D4_COORD:
        mapping = tuple(sorted(coord(x, y) for x in range(s.COLS) for y in range(s.ROWS)))
        assert len(set(mapping)) == s.COLS * s.ROWS
        seen.add(tuple(coord(x, y) for x in range(3) for y in range(3)))
    assert len(seen) == 8


def brute_force_escape_per_action(free, lethal, start, depth):
    walkable = free.copy()
    walkable[start] = True
    mask = 0
    for a in range(5):
        dx, dy = g.DELTAS[a] if a < 4 else (0, 0)
        nx, ny = start[0] + dx, start[1] + dy
        if a < 4 and not free[nx, ny]:
            continue
        if lethal[1][nx, ny]:
            continue
        frontier = {(nx, ny)}
        for t in range(1, depth):
            nxt = set()
            for (x, y) in frontier:
                for ddx, ddy in g.DELTAS + ((0, 0),):
                    mx, my = x + ddx, y + ddy
                    if (ddx or ddy) and not walkable[mx, my]:
                        continue
                    if not lethal[t + 1][mx, my]:
                        nxt.add((mx, my))
            frontier = nxt
            if not frontier:
                break
        if frontier:
            mask |= 1 << a
    return mask


def test_escape_masks_match_brute_force_per_action():
    world = fresh_world(scenario="classic", seed=11)
    agent = world.agents[0]
    rng = np.random.default_rng(5)
    free_tiles = np.argwhere(world.arena == 0)
    checked = 0
    for _ in range(200):
        x, y = free_tiles[rng.integers(len(free_tiles))]
        agent.x, agent.y = int(x), int(y)
        world.bombs = []
        for _ in range(int(rng.integers(1, 5))):
            bx, by = free_tiles[rng.integers(len(free_tiles))]
            world.bombs.append(Bomb((int(bx), int(by)), agent, int(rng.integers(0, s.BOMB_TIMER)),
                                    s.BOMB_POWER, agent.bomb_sprite))
        state = world.get_state_for_agent(agent)
        lethal = g.lethal_schedule(state)
        free = g.free_map(state)
        start = (agent.x, agent.y)
        mine = int(g.escape_masks(free, lethal, start))
        theirs = brute_force_escape_per_action(free, lethal, start, g.HORIZON)
        assert mine == theirs, f"at {start}: {mine:05b} != {theirs:05b}"
        checked += 1
    assert checked == 200


def step_physics(world):
    world.collect_coins()
    world.update_explosions()
    world.update_bombs()
    world.evaluate_explosions()
    world.step += 1


def test_lethal_index_is_the_move_count():
    for target in range(s.BOMB_TIMER + 1):
        world = fresh_world()
        agent = world.agents[0]
        world.arena[:] = -1
        for x in range(1, 10):
            world.arena[x, 1] = 0
        agent.x, agent.y = 1, 1
        agent.bombs_left = True
        world.perform_agent_action(agent, "BOMB")

        while True:
            state = world.get_state_for_agent(agent)
            if not state["bombs"] or state["bombs"][0][1] <= target:
                break
            step_physics(world)
        state = world.get_state_for_agent(agent)
        if not state["bombs"]:
            continue

        predicted = int(g.danger_map(g.lethal_schedule(state))[1, 1])
        moves = 0
        while not agent.dead and moves < 12:
            world.perform_agent_action(agent, "WAIT")
            step_physics(world)
            moves += 1
        assert moves == predicted, (
            f"countdown {state['bombs'][0][1]}: died after {moves} moves, "
            f"schedule predicted {predicted}")


@pytest.mark.parametrize("free_tiles", range(4, 9))
def test_escape_prediction_matches_running_the_escape(free_tiles):
    world = fresh_world()
    agent = world.agents[0]
    world.arena[:] = -1
    for x in range(1, free_tiles + 1):
        world.arena[x, 1] = 0
    agent.x, agent.y = 1, 1
    agent.bombs_left = True

    predicted = bool(g.can_escape_after_bomb(world.get_state_for_agent(agent)))
    world.perform_agent_action(agent, "BOMB")
    step_physics(world)
    for _ in range(12):
        if agent.dead:
            break
        world.perform_agent_action(agent, "RIGHT" if agent.x < free_tiles else "WAIT")
        step_physics(world)

    survived = not agent.dead
    assert predicted == survived, (
        f"corridor of {free_tiles}: predicted escapable={predicted}, actually survived={survived}")


def test_escape_mask_is_never_optimistic():
    rng = np.random.default_rng(11)
    checked = 0
    for trial in range(120):
        world = fresh_world(scenario="classic", seed=int(rng.integers(10_000)))
        agent = world.agents[0]
        free_tiles = np.argwhere(world.arena == 0)
        x, y = free_tiles[rng.integers(len(free_tiles))]
        agent.x, agent.y = int(x), int(y)
        agent.bombs_left = True
        world.bombs = []
        world.explosions = []

        if not g.can_escape_after_bomb(world.get_state_for_agent(agent)):
            continue
        world.perform_agent_action(agent, "BOMB")
        step_physics(world)

        for _ in range(g.HORIZON + 2):
            if agent.dead:
                break
            state = world.get_state_for_agent(agent)
            if state is None:
                break
            mask = int(g.escape_masks(g.free_map(state), g.lethal_schedule(state),
                                      (agent.x, agent.y)))
            if mask == 0:
                break
            move = next(i for i in range(5) if mask & (1 << i))
            world.perform_agent_action(agent, g.ACTIONS[move])
            step_physics(world)
        assert not agent.dead, (
            f"trial {trial}: the analysis promised an escape from {(int(x), int(y))} "
            f"and following its own plan still died")
        checked += 1
    assert checked > 30, f"only {checked} escapable situations tested"
