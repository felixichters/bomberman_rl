import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.aggregate import bootstrap_ci, per_round, summarise

TABLES = REPO_ROOT / "docs" / "report" / "tables"

HEADINGS = {
    "score": "Score", "coins": "Coins", "crates": "Crates", "kills": "Kills",
    "suicide_rate": "Suicide rate", "survival_rate": "Survival", "steps": "Steps",
    "invalid_rate": "Invalid", "win_rate": "Win rate", "ms_per_step": "ms/step",
}


def _escape(text):
    return str(text).replace("_", r"\_")


def _cell(stats, metric, precision=2, with_ci=True):
    mean, lo, hi = stats[metric]
    if not with_ci:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f}~{{\\tiny [{lo:.{precision}f}, {hi:.{precision}f}]}}"


def latex_table(rows, metrics, caption, label, first_column="Agent",
                precision=2, with_ci=True, note=None):
    header = " & ".join([first_column] + [HEADINGS.get(m, m) for m in metrics])
    lines = [r"\begin{table}[h]", r"\centering\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\begin{tabular}{l" + "r" * len(metrics) + "}", r"\toprule",
             header + r" \\", r"\midrule"]
    for name, stats in rows.items():
        cells = [_cell(stats, m, precision, with_ci) for m in metrics]
        lines.append(" & ".join([_escape(name)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(f"\\\\[0.3em]{{\\small {note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def from_final_eval(path, agent="alphabomb"):
    payload = json.loads(Path(path).read_text())
    return payload["summary"].get(agent), payload.get("records")


def grid_table(experiment, metrics=("coins", "crates", "suicide_rate", "steps", "score"),
               caption=None, label=None, order=None, rename=None, out=None):
    root = REPO_ROOT / "runs" / experiment
    rows = {}
    for final in sorted(root.glob("*/final_eval.json")):
        stats, _ = from_final_eval(final)
        if stats:
            rows[final.parent.name] = stats
    if order:
        rows = {k: rows[k] for k in order if k in rows}
    if rename:
        rows = {rename.get(k, k): v for k, v in rows.items()}
    text = latex_table(rows, metrics, caption or f"Experiment {_escape(experiment)}.",
                       label or f"tab:{experiment}", first_column="Variant")
    out = Path(out) if out else TABLES / f"{experiment}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(rows)} rows)")
    return text


def evaluation_table(records_path, metrics=("score", "coins", "crates", "kills",
                                            "suicide_rate", "win_rate"),
                     caption=None, label=None, out=None):
    payload = json.loads(Path(records_path).read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    rows = summarise(records)
    text = latex_table(rows, metrics, caption or "Evaluation.", label or "tab:evaluation")
    out = Path(out) if out else TABLES / (Path(records_path).stem + ".tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(rows)} rows)")
    return text


def paired_table(baseline_records, variant_records, agent="alphabomb",
                 metrics=("score", "coins", "kills", "suicide_rate"),
                 caption=None, label=None, out=None):
    a = json.loads(Path(variant_records).read_text())["records"]
    b = json.loads(Path(baseline_records).read_text())["records"]
    rows = {}
    for metric in metrics:
        va, vb = per_round(a, agent)[metric], per_round(b, agent)[metric]
        n = min(len(va), len(vb))
        rows[HEADINGS.get(metric, metric)] = {"difference": bootstrap_ci(va[:n] - vb[:n])}
    text = latex_table(rows, ("difference",), caption or "Paired difference.",
                       label or "tab:paired", first_column="Metric")
    out = Path(out) if out else TABLES / "paired.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")
    return text


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="what", required=True)

    p = sub.add_parser("grid")
    p.add_argument("experiment")
    p.add_argument("--caption", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None)

    p = sub.add_parser("eval")
    p.add_argument("records")
    p.add_argument("--caption", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.what == "grid":
        print(grid_table(args.experiment, caption=args.caption, label=args.label, out=args.out))
    else:
        print(evaluation_table(args.records, caption=args.caption, label=args.label, out=args.out))


if __name__ == "__main__":
    main()
