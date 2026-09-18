import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import style
from analysis.aggregate import METRICS, bootstrap_ci, per_round

FIGURES = REPO_ROOT / "docs" / "report" / "figures"

PRETTY = {
    "score": "Score", "coins": "Coins collected", "crates": "Crates destroyed",
    "kills": "Opponents killed", "suicide_rate": "Suicide rate",
    "survival_rate": "Survival rate", "steps": "Steps survived",
    "invalid_rate": "Invalid-action rate", "win_rate": "Win rate",
    "bombs": "Bombs dropped", "ms_per_step": "ms per step",
}


def _smooth(values, window):
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=float)
    return pd.Series(values).rolling(window, min_periods=max(1, window // 4)).mean().to_numpy()


RATE_METRICS = {"suicide_rate", "survival_rate", "win_rate", "invalid_rate"}


def _label_ends(ax, placements):
    placements = sorted(placements, key=lambda item: -item[1])
    lo, hi = ax.get_ylim()
    gap = 0.085 * (hi - lo)
    previous = None
    for x, y, text, color in placements:
        if previous is not None and previous - y < gap:
            y = previous - gap
        previous = y
        style.label_line(ax, x, y, text, color)


def learning_curves(runs, metrics=("coins", "suicide_rate", "steps", "score"),
                    out=None, title=None):
    style.use_report_style()
    labels = list(runs)
    frames = {label: pd.read_csv(path) for label, path in runs.items()}

    ncols = min(len(metrics), 2)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows), squeeze=False)
    handles = []

    for ax, metric in zip(axes.ravel(), metrics):
        placements = []
        for i, label in enumerate(labels):
            df = frames[label]
            if metric not in df or not len(df):
                continue
            color = style.SERIES[i % len(style.SERIES)]
            line, = ax.plot(df["round"], df[metric], color=color, label=label)
            if ax is axes.ravel()[0]:
                handles.append(line)
            placements.append((df["round"].iloc[-1], df[metric].iloc[-1], label, color))
        if metric in RATE_METRICS:
            ax.set_ylim(-0.03, 1.03)
        ax.set_title(PRETTY.get(metric, metric))
        ax.set_xlabel("training rounds")
        ax.margins(x=0.02)
        ax.set_xlim(left=0)
        right = max(df["round"].iloc[-1] for df in frames.values() if len(df))
        ax.set_xlim(0, right * 1.42)
        _label_ends(ax, placements)

    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    if len(labels) > 1:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(labels), 4),
                   bbox_to_anchor=(0.5, -0.02), frameon=False)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05 if len(labels) > 1 else 0, 1, 1))
    return style.finish(fig, out or FIGURES / "learning_curves.pdf")


def training_diagnostics(metrics_csv, out=None, window=100, title=None):
    style.use_report_style()
    df = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), squeeze=False)

    panels = [
        ("reward and survival", [("reward_sum", "shaped reward"), ("steps", "steps survived")]),
        ("bomb quality", [("GOOD_BOMB", "useful"), ("USELESS_BOMB", "useless"), ("SUICIDAL_BOMB", "inescapable")]),
        ("mistakes per round", [("invalid", "invalid actions"), ("suicide", "suicides")]),
        ("outcome", [("coins", "coins"), ("crates", "crates"), ("kills", "kills")]),
    ]
    for ax, (name, series) in zip(axes.ravel(), panels):
        for i, (column, label) in enumerate(series):
            if column not in df:
                continue
            color = style.SERIES[i % len(style.SERIES)]
            y = _smooth(df[column].to_numpy(dtype=float), window)
            ax.plot(df["round"], y, color=color, label=label)
            style.label_line(ax, df["round"].iloc[-1], y[-1], label, color)
        ax.set_title(name)
        ax.set_xlabel("training rounds")
        ax.margins(x=0.22)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return style.finish(fig, out or FIGURES / "training_diagnostics.pdf")


def weight_map(model_path, out=None, top=28, title=None):
    from agent_code.alphabomb import features as F
    from agent_code.alphabomb import game_utils as g
    from agent_code.alphabomb import model as M

    style.use_report_style()
    model = M.load_any(model_path)
    names = np.array(F.feature_names(model.feature_set))
    theta = model.theta - model.theta.mean(axis=0, keepdims=True)

    order = np.argsort(-theta.std(axis=0))[:top]
    order = order[np.argsort(names[order])]
    data = theta[:, order]
    limit = float(np.abs(data).max()) or 1.0

    fig, ax = plt.subplots(figsize=(0.34 * len(order) + 1.6, 2.9))
    im = ax.imshow(data, cmap=style.DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(model.n_actions), list(g.ACTIONS))
    ax.set_xticks(range(len(order)), names[order], rotation=60, ha="right", fontsize=7)
    ax.grid(False)
    ax.set_title(title or "Action preference per feature")
    bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    bar.set_label("weight, centred across actions", fontsize=7.5, color=style.INK_SECONDARY)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return style.finish(fig, out or FIGURES / "weight_map.pdf")


def comparison(evaluations, metrics=("score", "coins", "kills", "suicide_rate"),
               agent_of=None, out=None, title=None):
    style.use_report_style()
    labels = list(evaluations)
    agent_of = agent_of or {label: "alphabomb" for label in labels}

    fig, axes = plt.subplots(1, len(metrics), figsize=(2.35 * len(metrics), 2.8), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        means, los, his = [], [], []
        for label in labels:
            values = per_round(evaluations[label], agent_of[label])[metric]
            mean, lo, hi = bootstrap_ci(values)
            means.append(mean), los.append(lo), his.append(hi)
        y = np.arange(len(labels))
        colors = [style.SERIES[i % len(style.SERIES)] for i in range(len(labels))]
        ax.barh(y, means, color=colors, height=0.62)
        ax.errorbar(means, y, xerr=[np.subtract(means, los), np.subtract(his, means)],
                    fmt="none", ecolor=style.INK_SECONDARY, elinewidth=1.1, capsize=2.5)
        for yi, mean in zip(y, means):
            ax.text(mean, yi, f" {mean:.2f}", va="center", ha="left",
                    fontsize=7.5, color=style.INK_SECONDARY)
        ax.set_yticks(y, labels if ax is axes[0][0] else [""] * len(labels))
        ax.invert_yaxis()
        ax.set_title(PRETTY.get(metric, metric))
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.22)
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return style.finish(fig, out or FIGURES / "comparison.pdf")


def load_records(path):
    payload = json.loads(Path(path).read_text())
    return payload["records"] if isinstance(payload, dict) else payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate report figures.")
    sub = parser.add_subparsers(dest="what", required=True)

    p = sub.add_parser("curves")
    p.add_argument("runs", nargs="+", help="label=runs/<name>/eval.csv")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)

    p = sub.add_parser("diagnostics")
    p.add_argument("metrics_csv")
    p.add_argument("--out", default=None)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--title", default=None)

    p = sub.add_parser("weights")
    p.add_argument("model")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)

    args = parser.parse_args(argv)
    if args.what == "curves":
        runs = dict(item.split("=", 1) for item in args.runs)
        print(learning_curves(runs, out=args.out, title=args.title))
    elif args.what == "diagnostics":
        print(training_diagnostics(args.metrics_csv, out=args.out, window=args.window, title=args.title))
    else:
        print(weight_map(args.model, out=args.out, title=args.title))


if __name__ == "__main__":
    main()
