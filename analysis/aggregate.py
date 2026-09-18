import numpy as np

METRICS = ("score", "coins", "crates", "kills", "suicide_rate", "survival_rate",
           "steps", "invalid_rate", "bombs", "win_rate", "ms_per_step")


def per_round(records, agent):
    out = {m: [] for m in METRICS}
    for rec in records:
        stats = rec.get(agent)
        if stats is None:
            continue
        steps = max(stats["steps"], 1)
        scores = {name: s["score"] for name, s in rec.items() if isinstance(s, dict)}
        best = max(scores.values())
        winners = [n for n, v in scores.items() if v == best]
        out["score"].append(stats["score"])
        out["coins"].append(stats["coins"])
        out["crates"].append(stats["crates"])
        out["kills"].append(stats["kills"])
        out["suicide_rate"].append(float(stats["suicides"] > 0))
        out["survival_rate"].append(float(stats["survived"]))
        out["steps"].append(stats["steps"])
        out["invalid_rate"].append(stats["invalid"] / steps)
        out["bombs"].append(stats["bombs"])
        out["win_rate"].append((1.0 / len(winners)) if agent in winners else 0.0)
        out["ms_per_step"].append(1000.0 * stats["time"] / steps)
    return {m: np.asarray(v, dtype=float) for m, v in out.items()}


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0 if rng is None else rng)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def agent_names(records):
    names = []
    for rec in records:
        for key, value in rec.items():
            if isinstance(value, dict) and key not in names:
                names.append(key)
    return names


def summarise(records, agents=None):
    agents = agents or agent_names(records)
    return {a: {m: bootstrap_ci(v) for m, v in per_round(records, a).items()} for a in agents}


def paired_difference(records_a, records_b, agent, metric, n_boot=2000):
    a = per_round(records_a, agent)[metric]
    b = per_round(records_b, agent)[metric]
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    return bootstrap_ci(diff, n_boot=n_boot)


def format_table(summary, metrics=METRICS, precision=2):
    agents = list(summary)
    width = max(len(a) for a in agents) + 2
    header = "agent".ljust(width) + "".join(m.rjust(20) for m in metrics)
    lines = [header, "-" * len(header)]
    for a in agents:
        cells = []
        for m in metrics:
            mean, lo, hi = summary[a][m]
            cells.append(f"{mean:.{precision}f} [{lo:.{precision}f},{hi:.{precision}f}]".rjust(20))
        lines.append(a.ljust(width) + "".join(cells))
    return "\n".join(lines)


def to_markdown(summary, metrics=METRICS, precision=2):
    agents = list(summary)
    lines = ["| agent | " + " | ".join(metrics) + " |",
             "|" + "---|" * (len(metrics) + 1)]
    for a in agents:
        cells = [f"{summary[a][m][0]:.{precision}f} ({summary[a][m][1]:.{precision}f}–{summary[a][m][2]:.{precision}f})"
                 for m in metrics]
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
