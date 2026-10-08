# sim/config.py
import os
import yaml
from dataclasses import dataclass

_CFG = yaml.safe_load(open("config.yaml"))


@dataclass
class HarnessLimits:
    command_timeout_seconds: int = 300
    max_stdout_chars: int = 5000
    max_file_lines: int = 150
    max_file_chars: int = 10000


@dataclass
class EvaluationBudget:
    time_minutes: float = 60.0
    tool_calls: int = 100
    turns: int = 500
    cost_usd: float | None = None


def load_budget_from_dict(d: dict) -> EvaluationBudget:
    return EvaluationBudget(
        time_minutes=float(d["max_time_minutes"]),
        tool_calls=int(d["max_tool_calls"]),
        turns=int(d["max_turns"]),
    )


def load_harness_from_dict(d: dict) -> HarnessLimits:
    return HarnessLimits(
        command_timeout_seconds=int(d["command_timeout_seconds"]),
        max_stdout_chars=int(d["max_stdout_chars"]),
        max_file_lines=int(d["max_file_lines"]),
        max_file_chars=int(d["max_file_chars"]),
    )


def default_budget_harness():
    """Дефолты харнесса (щедрые: 60 мин / 100 вызовов) -- берём из config.yaml:env_budget."""
    return load_budget_from_dict(_CFG["env_budget"]), load_harness_from_dict(_CFG["env_budget"])


def tight_budget_harness(path="eval_config.yaml"):
    """Реалистичный "боевой" бюджет из eval_config.yaml (для curriculum.tight_budget_fraction)."""
    budget, harness = default_budget_harness()
    if os.path.exists(path):
        ev = (yaml.safe_load(open(path)) or {}).get("evaluation", {})
        if "max_time_minutes" in ev:
            budget.time_minutes = float(ev["max_time_minutes"])
        if "max_tool_calls" in ev:
            budget.tool_calls = int(ev["max_tool_calls"])
        if "max_turns" in ev:
            budget.turns = int(ev["max_turns"])
        if "timeout_seconds" in ev:
            harness.command_timeout_seconds = int(ev["timeout_seconds"])
    budget.time_minutes = float(os.environ.get("SWE_MAX_TIME_MINUTES", budget.time_minutes))
    budget.tool_calls = int(os.environ.get("SWE_MAX_TOOL_CALLS", budget.tool_calls))
    budget.turns = int(os.environ.get(
        "SWE_MAX_TURNS", os.environ.get("SWE_MAX_LOOP_ITERATIONS", budget.turns)))
    harness.command_timeout_seconds = int(
        os.environ.get("SWE_TIMEOUT_SECONDS", harness.command_timeout_seconds))
    return budget, harness
