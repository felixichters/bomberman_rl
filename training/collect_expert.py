import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_code.alphabomb import features as F
from agent_code.alphabomb import game_utils as g
from agent_code.alphabomb import rewards as R
from agent_code.alphabomb.learning import Hyper, NStepAccumulator
from training.env_runner import make_world, quiet_logging


def collect(expert="rule_based_agent", opponents=(), scenario="crates-sparse",
            rounds=400, hyper=None, seed=None, progress=True):
    hyper = hyper or Hyper()
    table = R.event_table(hyper.reward_scheme, hyper)
    shaped = R.scheme_is_shaped(hyper.reward_scheme)

    world = make_world([expert, *opponents], scenario=scenario, seed=seed)
    agent = world.agents[0]
    accumulator = NStepAccumulator(hyper.n_step, hyper.gamma)

    xs, acts, rews, xns, dones, steps_ = [], [], [], [], [], []
    stats = {"rounds": 0, "steps": 0, "coins": 0, "crates": 0, "suicides": 0, "survived": 0}

    def emit(transition):
        x0, a0, ret, bootstrap, terminal, n = transition
        xs.append(x0)
        acts.append(a0)
        rews.append(ret)
        xns.append(np.zeros_like(x0) if bootstrap is None else bootstrap)
        dones.append(terminal)
        steps_.append(n)

    with quiet_logging():
        for r in range(rounds):
            world.new_round()
            world.user_input = None
            accumulator.reset()
            history = []
            while world.running:
                old_state = world.get_state_for_agent(agent)
                if old_state is None:
                    world.do_step()
                    continue
                old_info = F.analyse(old_state)
                x_old = F.features_from(old_info, hyper.feature_set)

                world.do_step()
                action = agent.last_action
                events = list(agent.events)
                new_state = world.get_state_for_agent(agent)
                new_info = F.analyse(new_state) if new_state is not None else None

                events += R.derive_events(old_info, action, new_info, events, history)
                history.append(old_info.pos)
                reward = R.reward_from(events, table, hyper, old_info=old_info,
                                       new_info=new_info, shaped=shaped)

                emitted = accumulator.push(x_old, g.ACTION_INDEX.get(action, 4), reward)
                if emitted is not None:
                    emit(emitted)

                stats["steps"] += 1
                stats["coins"] += events.count("COIN_COLLECTED")
                stats["crates"] += events.count("CRATE_DESTROYED")

            for transition in accumulator.flush(done=True):
                if transition[0] is not None:
                    emit(transition)
            stats["rounds"] += 1
            stats["suicides"] += int(agent.dead and "KILLED_SELF" in agent.events)
            stats["survived"] += int(not agent.dead)
            if progress and (r + 1) % 100 == 0:
                print(f"  {r + 1}/{rounds} rounds, {len(xs)} transitions", flush=True)
        world.end()

    return {
        "x": np.asarray(xs, dtype=np.float32),
        "a": np.asarray(acts, dtype=np.int8),
        "r": np.asarray(rews, dtype=np.float32),
        "xn": np.asarray(xns, dtype=np.float32),
        "done": np.asarray(dones, dtype=bool),
        "steps": np.asarray(steps_, dtype=np.int8),
    }, stats


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", default="rule_based_agent")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--scenario", default="crates-sparse")
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--config", default="agent_code/alphabomb/models/config.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = Path(args.config)
    if not config.is_absolute():
        config = REPO_ROOT / config
    hyper = Hyper.from_json(config) if config.is_file() else Hyper()

    print(f"collecting {args.rounds} rounds of {args.expert} on '{args.scenario}'")
    data, stats = collect(args.expert, args.opponents, args.scenario, args.rounds, hyper, args.seed)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feature_set=hyper.feature_set, **data)

    per_round = {k: round(v / max(stats["rounds"], 1), 2) for k, v in stats.items() if k != "rounds"}
    print(f"\n{len(data['x'])} transitions -> {out}")
    print(f"expert per round: {json.dumps(per_round)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
