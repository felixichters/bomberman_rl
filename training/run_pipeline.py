import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_code.alphabomb.learning import Hyper
from training.curriculum import ORDER

STOP_FILE = REPO_ROOT / "runs" / "STOP"


def _interrupted():
    return STOP_FILE.exists()

PYTHON = sys.executable

PLAN = {
    "t1":  (2000, dict(eps_start=1.0, eps_decay_rounds=400.0)),
    "t2a": (4000, dict(eps_start=0.8, eps_decay_rounds=900.0, crate_reward=0.5, coin_found_reward=0.5,
                       preload="runs/expert/crates-sparse.npz")),
    "t2b": (5000, dict(eps_start=0.5, eps_decay_rounds=1200.0, crate_reward=0.4, coin_found_reward=0.4,
                       preload="runs/expert/crates-medium.npz")),
    "t2":  (6000, dict(eps_start=0.4, eps_decay_rounds=1500.0, crate_reward=0.3, coin_found_reward=0.3,
                       preload="runs/expert/classic.npz")),
    "t3a": (4000, dict(eps_start=0.3, eps_decay_rounds=1000.0, crate_reward=0.2, coin_found_reward=0.2,
                       invalid_penalty=-2.0, preload="runs/expert/classic.npz")),
    "t3":  (5000, dict(eps_start=0.3, eps_decay_rounds=1200.0, crate_reward=0.15, coin_found_reward=0.15,
                       invalid_penalty=-2.0, preload="runs/expert/classic-vs.npz")),
    "t4":  (8000, dict(eps_start=0.25, eps_decay_rounds=2000.0, crate_reward=0.1, coin_found_reward=0.1,
                       invalid_penalty=-2.0, preload="runs/expert/classic-vs.npz")),
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="prefix for the run directories")
    parser.add_argument("--model", default="linear", choices=["linear", "forest"])
    parser.add_argument("--from", dest="start", default=ORDER[0], choices=ORDER)
    parser.add_argument("--to", dest="stop", default=ORDER[-1], choices=ORDER)
    parser.add_argument("--init", default=None, help="checkpoint for the first stage")
    parser.add_argument("--rounds-scale", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--eval-rounds", type=int, default=60)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--base-config", default="agent_code/alphabomb/models/config.json")
    parser.add_argument("--override", default="{}", help="extra hyperparameters as JSON")
    args = parser.parse_args(argv)

    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path
    base = Hyper.from_json(base_path).to_dict()
    base["model"] = args.model
    base.update(json.loads(args.override))

    stages = ORDER[ORDER.index(args.start):ORDER.index(args.stop) + 1]
    suffix = ".joblib" if args.model == "forest" else ".npz"
    init = args.init
    print(f"pipeline '{args.tag}' ({args.model}): {' -> '.join(stages)}")

    for stage in stages:
        if _interrupted():
            print("stop file present; ending the pipeline")
            return 1
        rounds, overrides = PLAN[stage]
        rounds = max(1, int(rounds * args.rounds_scale))
        name = f"{args.tag}_{stage}"
        run_dir = REPO_ROOT / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=True)

        config = dict(base)
        config.update(overrides)
        if config.get("preload"):
            demo = REPO_ROOT / config["preload"]
            if not demo.is_file():
                print(f"note: {demo.relative_to(REPO_ROOT)} not found; {stage} runs without demonstrations")
                config["preload"] = ""
        config["total_rounds"] = rounds
        Hyper(**config)
        config_path = run_dir / "config_stage.json"
        config_path.write_text(json.dumps(config, indent=2))

        cmd = [PYTHON, str(REPO_ROOT / "training" / "run_training.py"),
               "--stage", stage, "--name", name, "--config", str(config_path),
               "--rounds", str(rounds), "--eval-every", str(args.eval_every),
               "--eval-rounds", str(args.eval_rounds), "--eval-workers", str(args.eval_workers)]
        if init:
            cmd += ["--init", str(init)]

        print(f"\n{'=' * 70}\n{stage}: {rounds} rounds"
              + (f", warm start from {init}" if init else ", from scratch") + f"\n{'=' * 70}", flush=True)
        try:
            code = subprocess.call(cmd, cwd=REPO_ROOT)
        except KeyboardInterrupt:
            print("interrupted")
            return 1
        if code < 0:
            print(f"{stage} was terminated by signal {-code}; stopping the pipeline")
            return 1

        best = run_dir / f"best{suffix}"
        if not best.exists():
            best = run_dir / f"model{suffix}"
        if not best.exists():
            print(f"{stage} produced no checkpoint; stopping")
            return 1
        init = best
        if code != 0:
            print(f"note: {stage} did not meet its promotion criterion; continuing anyway")

    print(f"\npipeline finished; final checkpoint: {init}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
