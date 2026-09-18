import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.curriculum import STAGES, check_gate
from training.env_runner import make_world, quiet_logging

EVAL_COLUMNS = ("round", "rounds_seen", "score", "coins", "crates", "kills",
                "suicide_rate", "survival_rate", "steps", "invalid_rate", "win_rate")

STAGE_OBJECTIVE = {"t1": "coins", "t2a": "crates", "t2b": "crates", "t2": "score",
                   "t3a": "win_rate", "t3": "win_rate", "t4": "score", "t4sp": "score"}


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


SELECTION_SEEDS = 90_000
REPORT_SEEDS = 70_000


def evaluate_snapshot(model_path, config_path, stage, rounds, workers, seed_start=SELECTION_SEEDS):
    from analysis.evaluate import evaluate
    from analysis.aggregate import summarise

    spec = STAGES[stage]
    agents = ["alphabomb"] + list(spec["eval_opponents"])
    records = evaluate(agents, rounds, spec["eval_scenario"],
                       seeds=list(range(seed_start, seed_start + rounds)),
                       env={"ALPHABOMB_MODEL": str(model_path), "ALPHABOMB_CONFIG": str(config_path)},
                       workers=workers)
    return summarise(records), records


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--name", required=True, help="run directory under runs/")
    parser.add_argument("--rounds", type=int, default=None, help="override the stage default")
    parser.add_argument("--config", default="agent_code/alphabomb/models/config.json")
    parser.add_argument("--init", default=None, help="checkpoint to start from")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-rounds", type=int, default=60)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None, help="pin the training worlds too")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    spec = STAGES[args.stage]
    rounds = args.rounds if args.rounds is not None else spec["rounds"]
    run_dir = REPO_ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    shutil.copyfile(config_path, run_dir / "config_input.json")

    family = json.loads(config_path.read_text()).get("model", "linear")
    suffix = ".joblib" if family == "forest" else ".npz"
    model_path = run_dir / f"model{suffix}"
    if args.init:
        init = Path(args.init)
        if not init.is_absolute():
            init = REPO_ROOT / init
        shutil.copyfile(init, model_path)
        print(f"warm start from {init}")
    elif model_path.exists():
        model_path.unlink()

    stale = [f for f in ("eval.csv", "metrics.csv") if (run_dir / f).exists()]
    if stale:
        archive = run_dir / f"superseded_{time.strftime('%Y%m%d_%H%M%S')}"
        archive.mkdir(parents=True, exist_ok=True)
        for name in stale:
            shutil.move(str(run_dir / name), str(archive / name))
        print(f"archived {', '.join(stale)} from a previous run into {archive.name}/")

    os.environ["ALPHABOMB_RUN_DIR"] = str(run_dir)
    os.environ["ALPHABOMB_MODEL"] = str(model_path)
    os.environ["ALPHABOMB_CONFIG"] = str(config_path)

    (run_dir / "run.json").write_text(json.dumps({
        "stage": args.stage, "description": spec["description"], "rounds": rounds,
        "opponents": spec["opponents"], "scenario": spec["scenario"],
        "init": args.init, "git": git_sha(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

    eval_path = run_dir / "eval.csv"
    if not eval_path.exists():
        with eval_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(EVAL_COLUMNS)

    agents = [("alphabomb", True)] + [(o, False) for o in spec["opponents"]]
    print(f"[{args.stage}] {spec['description']}")
    print(f"         {rounds} rounds vs {spec['opponents'] or 'nobody'} on '{spec['scenario']}'")

    world = make_world(agents, scenario=spec["scenario"], seed=args.seed)
    started = time.time()
    best = {"value": -float("inf"), "round": 0, "last": None}
    with quiet_logging():
        for r in range(1, rounds + 1):
            world.new_round()
            while world.running:
                world.do_step()
            if args.eval_every and r % args.eval_every == 0:
                _snapshot(run_dir, model_path, config_path, args, r, rounds, started, best)
        world.end()

    best_path = run_dir / f"best{model_path.suffix}"
    final_model = best_path if best_path.exists() else model_path
    print(f"\nBest checkpoint: round {best['round']} "
          f"({STAGE_OBJECTIVE[args.stage]} = {best['value']:.3f})")
    summary, records = evaluate_snapshot(final_model, config_path, args.stage,
                                         max(args.eval_rounds, 200), args.eval_workers,
                                         seed_start=REPORT_SEEDS)
    (run_dir / "final_eval.json").write_text(json.dumps({"summary": {
        a: {m: list(v) for m, v in stats.items()} for a, stats in summary.items()},
        "records": records}, indent=2))

    from analysis.aggregate import format_table
    print("\nFinal greedy evaluation:")
    print(format_table(summary))
    passed, lines = check_gate(args.stage, summary)
    print(f"\nPromotion gate for {args.stage}: {'PASSED' if passed else 'not met'}")
    print("\n".join(lines))
    return 0 if passed else 1


def _snapshot(run_dir, model_path, config_path, args, r, rounds, started, best):
    summary, _ = evaluate_snapshot(model_path, config_path, args.stage,
                                   args.eval_rounds, args.eval_workers)
    stats = summary["alphabomb"]
    row = [r, rounds] + [round(stats[m][0], 4) for m in EVAL_COLUMNS[2:]]
    with (run_dir / "eval.csv").open("a", newline="") as fh:
        csv.writer(fh).writerow(row)

    objective = stats[STAGE_OBJECTIVE[args.stage]][0]
    smoothed = objective if best["last"] is None else 0.5 * (objective + best["last"])
    best["last"] = objective
    objective = smoothed
    marker = ""
    if objective > best["value"]:
        best.update(value=objective, round=r)
        shutil.copyfile(model_path, run_dir / f"best{model_path.suffix}")
        (run_dir / "best.json").write_text(json.dumps(
            {"round": r, "metric": STAGE_OBJECTIVE[args.stage], "value": objective,
             "summary": {m: list(v) for m, v in stats.items()}}, indent=2))
        marker = "  <- best"

    rate = r / max(time.time() - started, 1e-9)
    print(f"  round {r:>6}/{rounds}  score={stats['score'][0]:6.2f} coins={stats['coins'][0]:5.2f} "
          f"suicide={stats['suicide_rate'][0]:4.2f} steps={stats['steps'][0]:6.1f} "
          f"win={stats['win_rate'][0]:4.2f}  [{rate:.1f} rounds/s]{marker}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
