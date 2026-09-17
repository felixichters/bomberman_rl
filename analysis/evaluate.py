import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.env_runner import run_rounds

DEFAULT_SEEDS = list(range(1000, 1000 + 200))


def _worker(payload):
    agents, scenario, seeds, env = payload
    os.environ.update({k: str(v) for k, v in env.items() if v is not None})
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
    return run_rounds(agents, len(seeds), scenario=scenario, seeds=seeds)


def evaluate(agents, n_rounds=200, scenario="classic", seeds=None, env=None, workers=1):
    seeds = list(seeds if seeds is not None else DEFAULT_SEEDS)[:n_rounds]
    while len(seeds) < n_rounds:
        seeds.append(seeds[len(seeds) % max(len(seeds), 1)] + 100000)
    agents = [(a, False) if isinstance(a, str) else tuple(a) for a in agents]
    env = dict(env or {})

    if workers <= 1:
        return _worker((agents, scenario, seeds, env))

    chunks = [seeds[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    payloads = [(agents, scenario, c, env) for c in chunks]
    records = []
    with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
        for part in pool.map(_worker, payloads):
            records.extend(part)
    return records


def expand_agents(spec):
    out = []
    for item in spec:
        if "x" in item and item.rsplit("x", 1)[-1].strip().isdigit():
            name, count = item.rsplit("x", 1)
            out.extend([name.strip()] * int(count))
        else:
            out.append(item.strip())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate agents over a fixed set of boards.")
    parser.add_argument("--agents", nargs="+", required=True,
                        help="agent directories; 'rule_based_agent x3' repeats one")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default=None, help="checkpoint for the AlphaBomb agent")
    parser.add_argument("--config", default=None, help="config json for the AlphaBomb agent")
    parser.add_argument("--out", default=None, help="where to write the per-round records as JSON")
    parser.add_argument("--label", default=None)
    args = parser.parse_args(argv)

    agents = expand_agents(args.agents)
    seeds = list(range(args.seed_start, args.seed_start + args.rounds))
    env = {"ALPHABOMB_MODEL": args.model, "ALPHABOMB_CONFIG": args.config}
    records = evaluate(agents, args.rounds, args.scenario, seeds, env=env, workers=args.workers)

    payload = {
        "label": args.label or "+".join(agents),
        "agents": agents, "scenario": args.scenario, "rounds": args.rounds,
        "seed_start": args.seed_start, "model": args.model, "config": args.config,
        "records": records,
    }
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        print(f"wrote {out}")

    from analysis.aggregate import summarise, format_table

    print(format_table(summarise(records)))
    return payload


if __name__ == "__main__":
    main()
