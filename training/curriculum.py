STAGES = {
    "t1": dict(
        description="Task 1 -- collect revealed coins on an empty board.",
        scenario="coin-heaven", opponents=[],
        eval_scenario="coin-heaven", eval_opponents=[],
        gate={"coins": 40.0, "suicide_rate": ("<=", 0.02)},
        rounds=4000,
    ),
    "t2a": dict(
        description="Task 2a -- sparse crates: learn 'drop a bomb, then run'.",
        scenario="crates-sparse", opponents=[],
        eval_scenario="crates-sparse", eval_opponents=[],
        gate={"crates": 20.0, "suicide_rate": ("<=", 0.15)},
        rounds=4000,
    ),
    "t2b": dict(
        description="Task 2b -- medium crate density: tighter escapes.",
        scenario="crates-medium", opponents=[],
        eval_scenario="crates-medium", eval_opponents=[],
        gate={"crates": 45.0, "suicide_rate": ("<=", 0.10)},
        rounds=6000,
    ),
    "t2": dict(
        description="Task 2 -- the tournament board, alone: bomb crates and survive.",
        scenario="classic", opponents=[],
        eval_scenario="classic", eval_opponents=[],
        gate={"suicide_rate": ("<=", 0.05), "coins": 6.0},
        rounds=8000,
    ),
    "t3a": dict(
        description="Task 3a -- against the peaceful agent (easy prey).",
        scenario="classic", opponents=["peaceful_agent", "peaceful_agent"],
        eval_scenario="classic", eval_opponents=["peaceful_agent", "peaceful_agent"],
        gate={"win_rate": 0.8},
        rounds=6000,
    ),
    "t3": dict(
        description="Task 3 -- against the coin collector (hard prey).",
        scenario="classic", opponents=["coin_collector_agent", "coin_collector_agent"],
        eval_scenario="classic", eval_opponents=["coin_collector_agent", "coin_collector_agent"],
        gate={"win_rate": 0.35},
        rounds=8000,
    ),
    "t4": dict(
        description="Task 4 -- against three full-strength rule-based agents.",
        scenario="classic", opponents=["rule_based_agent"] * 3,
        eval_scenario="classic", eval_opponents=["rule_based_agent"] * 3,
        gate={"win_rate": 0.27},
        rounds=12000,
    ),
    "t4sp": dict(
        description="Task 4 -- self-play against frozen snapshots of ourselves.",
        scenario="classic", opponents=["alphabomb_frozen"] * 3,
        eval_scenario="classic", eval_opponents=["rule_based_agent"] * 3,
        gate={"win_rate": 0.4},
        rounds=12000,
    ),
}

ORDER = ("t1", "t2a", "t2b", "t2", "t3a", "t3", "t4")


def check_gate(stage, summary, agent="alphabomb"):
    gate = STAGES[stage]["gate"]
    stats = summary[agent]
    passed, lines = True, []
    for metric, requirement in gate.items():
        op, threshold = requirement if isinstance(requirement, tuple) else (">=", requirement)
        value = stats[metric][0]
        ok = value >= threshold if op == ">=" else value <= threshold
        passed &= ok
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {metric} = {value:.3f} {op} {threshold}")
    return passed, lines
