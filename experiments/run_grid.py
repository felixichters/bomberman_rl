import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = sys.executable


def materialise_config(base, overrides, path):
    from agent_code.alphabomb.learning import Hyper

    merged = dict(base)
    merged.update(overrides)
    hyper = Hyper(**merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hyper.to_dict(), indent=2))
    return hyper


def run_variant(spec):
    name, config_path, run_name, stage, rounds, init, eval_every, eval_rounds, log_path = spec
    cmd = [PYTHON, str(REPO_ROOT / "training" / "run_training.py"),
           "--stage", stage, "--name", run_name, "--config", str(config_path),
           "--rounds", str(rounds), "--eval-every", str(eval_every),
           "--eval-rounds", str(eval_rounds), "--eval-workers", "1"]
    if init:
        cmd += ["--init", str(init)]
    started = time.time()
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "OMP_NUM_THREADS": "1"})
    return name, result.returncode, time.time() - started


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="path to the experiment JSON")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None, help="run just these variants")
    args = parser.parse_args(argv)

    spec_path = Path(args.experiment)
    if not spec_path.is_absolute():
        spec_path = REPO_ROOT / spec_path
    spec = json.loads(spec_path.read_text())

    experiment = spec.get("name", spec_path.stem)
    out_root = REPO_ROOT / "runs" / experiment
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "experiment.json").write_text(json.dumps(spec, indent=2))

    jobs = []
    for name, overrides in spec["variants"].items():
        if args.only and name not in args.only:
            continue
        config_path = out_root / name / "config.json"
        materialise_config(spec.get("base", {}), overrides, config_path)
        jobs.append((name, config_path, f"{experiment}/{name}", spec["stage"],
                     spec.get("rounds", 3000), spec.get("init"),
                     spec.get("eval_every", 500), spec.get("eval_rounds", 60),
                     out_root / name / "train.log"))

    print(f"{experiment}: {len(jobs)} variants, {args.workers} at a time")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, code, seconds in pool.map(run_variant, jobs):
            print(f"  {name:<28} exit={code} {seconds / 60:5.1f} min", flush=True)

    collect(experiment)
    return 0


def collect(experiment):
    from analysis.aggregate import METRICS

    out_root = REPO_ROOT / "runs" / experiment
    rows = {}
    for final in sorted(out_root.glob("*/final_eval.json")):
        payload = json.loads(final.read_text())
        stats = payload["summary"].get("alphabomb")
        if stats:
            rows[final.parent.name] = stats

    if not rows:
        print("no finished variants yet")
        return rows

    header = "variant".ljust(28) + "".join(m.rjust(20) for m in METRICS)
    lines = [header, "-" * len(header)]
    for name, stats in rows.items():
        cells = [f"{stats[m][0]:.2f} [{stats[m][1]:.2f},{stats[m][2]:.2f}]".rjust(20) for m in METRICS]
        lines.append(name.ljust(28) + "".join(cells))
    table = "\n".join(lines)
    (out_root / "summary.txt").write_text(table + "\n")
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n" + table)
    return rows


if __name__ == "__main__":
    sys.exit(main())
