import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import events as e
from agent_code.alphabomb import features as F
from agent_code.alphabomb import game_utils as g
from agent_code.alphabomb import rewards as R
from agent_code.alphabomb.learning import Hyper
from training.env_runner import make_world, quiet_logging

VOCAB = tuple(sorted({v for k, v in vars(e).items() if k.isupper() and isinstance(v, str)}
                     | set(R.CUSTOM_EVENTS)))
VOCAB_INDEX = {name: i for i, name in enumerate(VOCAB)}
DEFAULT_TRACE = REPO_ROOT / "runs" / "expert" / "trace.npz"


def psi_parts(info):
    if info is None:
        return np.zeros(5, dtype=np.float32)
    danger = int(info.danger[info.pos])
    safety = 1.0 if danger >= int(g.UNREACHABLE) else min(1.0, (danger - 1) / float(g.HORIZON))
    return np.array([
        R._proximity(info.coin_dist), R._proximity(info.crate_dist), R._proximity(info.enemy_dist),
        safety, max(0.0, 1.0 - info.n_crates / R.CRATE_REFERENCE),
        1.0 if info.escape_mask else 0.0,
    ], dtype=np.float32)


def encode(events):
    mask = 0
    for name in events:
        bit = VOCAB_INDEX.get(name)
        if bit is not None:
            mask |= 1 << bit
    return np.uint32(mask)


def record(expert="rule_based_agent", opponents=(), scenario="crates-sparse",
           rounds=200, seed=None):
    hyper = Hyper()
    world = make_world([expert, *opponents], scenario=scenario, seed=seed)
    agent = world.agents[0]

    actions, masks, psi, psi_next, terminal, episode = [], [], [], [], [], []
    with quiet_logging():
        for r in range(rounds):
            world.new_round()
            world.user_input = None
            history = []
            while world.running:
                old_state = world.get_state_for_agent(agent)
                if old_state is None:
                    world.do_step()
                    continue
                old_info = F.analyse(old_state)
                world.do_step()
                events = list(agent.events)
                new_state = world.get_state_for_agent(agent)
                new_info = F.analyse(new_state) if new_state is not None else None
                events += R.derive_events(old_info, agent.last_action, new_info, events, history)
                history.append(old_info.pos)

                actions.append(g.ACTION_INDEX.get(agent.last_action, 4))
                masks.append(encode(events))
                psi.append(psi_parts(old_info))
                psi_next.append(psi_parts(new_info))
                terminal.append(new_info is None or not world.running)
                episode.append(r)
            if (r + 1) % 50 == 0:
                print(f"  {r + 1}/{rounds} rounds, {len(actions)} steps", flush=True)
        world.end()

    return {"action": np.asarray(actions, dtype=np.int8),
            "events": np.asarray(masks, dtype=np.uint32),
            "psi": np.asarray(psi, dtype=np.float32),
            "psi_next": np.asarray(psi_next, dtype=np.float32),
            "terminal": np.asarray(terminal, dtype=bool),
            "episode": np.asarray(episode, dtype=np.int32),
            "vocab": np.asarray(VOCAB)}


def returns_for(trace, hyper):
    table = R.event_table(hyper.reward_scheme, hyper)
    shaped = R.scheme_is_shaped(hyper.reward_scheme)
    weights = np.array([hyper.psi_coin, hyper.psi_crate, hyper.psi_enemy,
                        hyper.psi_safe, hyper.psi_cleared, hyper.psi_escape], dtype=np.float64)
    stored = trace["psi"].shape[1]
    if stored != len(weights):
        raise ValueError(f"trace stores {stored} potential terms but this version uses "
                         f"{len(weights)}; re-record it with --collect")

    value = np.zeros(len(VOCAB))
    for name, amount in table.items():
        if name in VOCAB_INDEX:
            value[VOCAB_INDEX[name]] = amount
    bits = ((trace["events"][:, None] >> np.arange(len(VOCAB))[None, :]) & 1).astype(np.float64)
    both = (bits[:, VOCAB_INDEX[e.KILLED_SELF]] > 0) & (bits[:, VOCAB_INDEX[e.GOT_KILLED]] > 0)
    bits[both, VOCAB_INDEX[e.GOT_KILLED]] = 0.0
    reward = bits @ value + hyper.step_penalty

    if shaped:
        psi = trace["psi"] @ weights
        psi_next = np.where(trace["terminal"], 0.0, trace["psi_next"] @ weights)
        reward = reward + hyper.gamma * psi_next - psi

    n, gamma = hyper.n_step, hyper.gamma
    episode = trace["episode"]
    out = np.zeros(len(reward))
    for i in range(len(reward)):
        total, discount = 0.0, 1.0
        for k in range(n):
            j = i + k
            if j >= len(reward) or episode[j] != episode[i]:
                break
            total += discount * reward[j]
            discount *= gamma
        out[i] = total
    return reward, out


def report(trace, hyper, label=""):
    _, returns = returns_for(trace, hyper)
    action = trace["action"]
    means = {a: float(returns[action == a].mean()) for a in range(6) if (action == a).any()}
    move = np.mean([means[a] for a in range(4) if a in means])
    bomb = means.get(5, float("nan"))
    print(f"{label:<26} " + "  ".join(f"{g.ACTIONS[a]} {means[a]:+.3f}" for a in sorted(means))
          + f"   | BOMB - move = {bomb - move:+.3f}")
    return bomb - move


def sweep(trace):
    print(f"\n{'configuration':<26} mean n-step return by action, in the expert's own games"
          f"\n" + "-" * 122)
    candidates = {
        "n=3 (first design)": Hyper(n_step=3, crate_reward=0.1, coin_found_reward=0.1, psi_safe=0.4),
        "n=4": Hyper(n_step=4),
        "n=5": Hyper(n_step=5),
        "n=6 (default)": Hyper(),
        "n=8": Hyper(n_step=8),
        "n=6, psi_safe=0": Hyper(psi_safe=0.0),
        "n=6, psi_safe=0.15": Hyper(psi_safe=0.15),
        "n=6, crate 0.1": Hyper(crate_reward=0.1, coin_found_reward=0.1),
        "n=6, crate 0.5": Hyper(crate_reward=0.5, coin_found_reward=0.5),
        "n=6, potential only": Hyper(reward_scheme="potential"),
        "n=6, events scheme": Hyper(reward_scheme="events"),
        "n=6, task only": Hyper(reward_scheme="task"),
    }
    results = {}
    for label, hyper in candidates.items():
        results[label] = report(trace, hyper, label)
    print("\nranked by (BOMB - move), i.e. how strongly the reward endorses the expert's bombs:")
    for label, gap in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {gap:+.3f}  {label}")
    return results


def hyper_n(trace):
    return Hyper().n_step


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", type=int, default=0, help="record this many expert rounds")
    parser.add_argument("--scenario", default="crates-sparse")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.trace)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if args.collect:
        print(f"recording {args.collect} expert rounds on '{args.scenario}'")
        trace = record(scenario=args.scenario, opponents=args.opponents, rounds=args.collect)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **trace)
        print(f"{len(trace['action'])} steps -> {path}")

    if args.sweep:
        with np.load(path, allow_pickle=False) as handle:
            trace = {k: handle[k] for k in handle.files}
        results = sweep(trace)
        (path.parent / "reward_sweep.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
