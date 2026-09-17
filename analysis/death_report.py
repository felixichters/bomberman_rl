import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import events as e
from agent_code.alphabomb import features as F
from agent_code.alphabomb import game_utils as g
from training.env_runner import make_world, quiet_logging


def local_shape(info):
    x, y = info.pos
    free = sum(1 for dx, dy in g.DELTAS if info.free_move[x + dx, y + dy])
    return {0: "sealed in", 1: "dead end", 2: "corridor"}.get(free, "junction or open")


def analyse_round(world, agent, runner, max_steps=400):
    fake_self = runner.fake_self
    if not hasattr(world, "user_input"):
        world.user_input = None
    trail = []
    while world.running and world.step < max_steps:
        state = world.get_state_for_agent(agent)
        if state is None:
            break
        info = F.analyse(state)
        fake_self._analysis_cache.clear()
        world.do_step()
        trail.append({
            "pos": info.pos,
            "mask": int(info.escape_mask),
            "escape_bomb": bool(info.escape_bomb),
            "shape": local_shape(info),
            "action": agent.last_action,
            "dropped": e.BOMB_DROPPED in agent.events,
            "invalid": e.INVALID_ACTION in agent.events,
            "danger": int(info.danger[info.pos]),
            "bomb_crates": int(info.bomb_crates),
        })
        if agent.dead:
            return trail
    return None


def attribute(trail):
    for step in reversed(trail):
        if step["mask"] == 0:
            continue
        index = g.ACTION_INDEX.get(step["action"], 4)
        if step["dropped"]:
            cause = ("dropped a bomb with no escape" if not step["escape_bomb"]
                     else "dropped an escapable bomb")
        elif step["invalid"]:
            cause = "illegal move wasted the escape"
        elif index < 5 and not (step["mask"] >> index) & 1:
            cause = "moved somewhere with no survival plan"
        else:
            cause = "took a surviving move, died later anyway"
        return cause, step
    return "no escape existed at any point", trail[-1] if trail else None


def run(model, rounds, scenario, opponents, seed_start):
    os.environ["ALPHABOMB_MODEL"] = str(model)
    agents = [("alphabomb", False)] + [(o, False) for o in opponents]
    world = make_world(agents, scenario=scenario, seed=seed_start)
    agent = world.agents[0]
    runner = agent.backend.runner

    causes = collections.Counter()
    shapes = collections.Counter()
    steps_at_death, deaths = [], 0
    with quiet_logging():
        for r in range(rounds):
            world.rng = np.random.default_rng(seed_start + r)
            world.new_round()
            trail = analyse_round(world, agent, runner)
            if trail is None:
                continue
            deaths += 1
            cause, step = attribute(trail)
            causes[cause] += 1
            if step:
                shapes[step["shape"]] += 1
            steps_at_death.append(len(trail))
        world.end()

    return {
        "rounds": rounds, "deaths": deaths,
        "death_rate": deaths / max(rounds, 1),
        "causes": dict(causes), "geometry_at_the_decision": dict(shapes),
        "mean_steps_survived_when_dying": float(np.mean(steps_at_death)) if steps_at_death else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    report = run(args.model, args.rounds, args.scenario, args.opponents, args.seed_start)
    print(f"{report['deaths']}/{report['rounds']} rounds ended in death "
          f"({report['death_rate']:.1%}); when dying, survived "
          f"{report['mean_steps_survived_when_dying']:.0f} steps on average\n")
    print("at the last decision where an escape still existed, the agent:")
    for cause, n in sorted(report["causes"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4} ({n / max(report['deaths'], 1):5.1%})  {cause}")
    print("\nand the board there was:")
    for shape, n in sorted(report["geometry_at_the_decision"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4} ({n / max(report['deaths'], 1):5.1%})  {shape}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
