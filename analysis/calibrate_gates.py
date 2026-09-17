import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.curriculum import STAGES

METRICS = ("score", "coins", "crates", "kills", "suicide_rate", "survival_rate", "win_rate")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--reference", default="rule_based_agent")
    parser.add_argument("--seed-start", type=int, default=90000)
    parser.add_argument("--out", default="runs/report/gate_calibration.json")
    args = parser.parse_args(argv)

    from analysis.aggregate import summarise

    results = {}
    for stage, spec in STAGES.items():
        if stage == "t4sp":
            continue
        agents = [args.reference] + list(spec["eval_opponents"])
        out = REPO_ROOT / "runs" / "report" / f"gate_{stage}.json"
        cmd = [sys.executable, str(REPO_ROOT / "analysis" / "evaluate.py"),
               "--agents", *agents, "--rounds", str(args.rounds),
               "--scenario", spec["eval_scenario"], "--workers", "3",
               "--seed-start", str(args.seed_start), "--out", str(out)]
        subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True)
        if not out.exists():
            continue
        records = json.loads(out.read_text())["records"]
        summary = summarise(records)
        key = args.reference if args.reference in summary else f"{args.reference}_0"
        if key not in summary:
            print(f"{stage}: could not find {args.reference} in {list(summary)}")
            continue
        stats = summary[key]
        results[stage] = {m: stats[m][0] for m in METRICS}

        print(f"\n{stage}: {args.reference} vs {spec['eval_opponents'] or 'nobody'} "
              f"on '{spec['eval_scenario']}'")
        print("   " + "  ".join(f"{m}={results[stage][m]:.2f}" for m in METRICS))
        for metric, requirement in spec["gate"].items():
            op, threshold = requirement if isinstance(requirement, tuple) else (">=", requirement)
            got = results[stage].get(metric)
            if got is None:
                continue
            meets = got >= threshold if op == ">=" else got <= threshold
            verdict = "reference meets it" if meets else "REFERENCE FAILS OUR OWN GATE"
            print(f"   gate {metric} {op} {threshold}: reference gets {got:.3f} -- {verdict}")

    path = REPO_ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
