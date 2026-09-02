"""聚合多次运行，并计算实验组相对 Baseline 的描述性差异。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean, median, stdev
from typing import Any

from .metric_definitions import safe_ratio


EXPERIMENT_METRICS_FORMAT_VERSION = 1

# 提前破产本身是结果，但不能把破产日经营状态当作完整期限终态。
# 新增经营指标时默认只使用跑满期限的运行，避免因漏维护名单而混入口径。
ALL_RUN_METRICS = {
    "survival_days",
    "max_cash_drawdown_absolute",
    "max_cash_drawdown_rate",
}


def aggregate_experiments(
    runs_by_group: Mapping[str, Sequence[dict[str, Any]]],
    *,
    baseline_group: str,
) -> dict[str, Any]:
    """汇总已终结运行；组间只报告描述统计，不擅自选择显著性检验。"""
    if baseline_group not in runs_by_group:
        raise ValueError(f"Unknown baseline group: {baseline_group}")
    if any(not runs for runs in runs_by_group.values()):
        raise ValueError("Every experiment group must contain at least one run")

    for group, runs in runs_by_group.items():
        for run in runs:
            if run["run"].get("status") != "completed":
                run_id = run["run"].get("run_id")
                raise ValueError(f"Run {run_id!r} in {group!r} is not finalized")
            if run["summary"].get("outcome") not in {"completed", "bankrupt"}:
                run_id = run["run"].get("run_id")
                raise ValueError(f"Run {run_id!r} in {group!r} has invalid outcome")

    configured_days = {
        run["run"].get("configured_days")
        for runs in runs_by_group.values()
        for run in runs
    }
    if len(configured_days) != 1:
        raise ValueError("All experiment runs must use the same configured duration")

    groups = {
        group: _aggregate_group(group, list(runs))
        for group, runs in sorted(runs_by_group.items())
    }
    baseline = groups[baseline_group]
    comparisons = [
        _compare_group(baseline, groups[group])
        for group in sorted(groups)
        if group != baseline_group
    ]
    return {
        "format_version": EXPERIMENT_METRICS_FORMAT_VERSION,
        "baseline_group": baseline_group,
        "groups": list(groups.values()),
        "comparisons": comparisons,
        "inference_status": "not_configured",
    }


def _aggregate_group(
    group: str, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    outcomes = [str(run["summary"]["outcome"]) for run in runs]
    scalar_names = sorted(
        {
            name
            for run in runs
            for name, value in _run_scalars(run).items()
            if _is_number(value)
        }
    )
    scalar_metrics = []
    for metric in scalar_names:
        eligible_runs = _eligible_runs(metric, runs)
        values = [
            float(value)
            for run in eligible_runs
            if _is_number(value := _run_scalars(run).get(metric))
        ]
        if values:
            scalar_metrics.append({"metric": metric, **_describe(values)})

    return {
        "group": group,
        "run_count": len(runs),
        "completed_run_count": outcomes.count("completed"),
        "bankrupt_run_count": outcomes.count("bankrupt"),
        "bankruptcy_rate": outcomes.count("bankrupt") / len(runs),
        "run_ids": [run["run"].get("run_id") for run in runs],
        "scalar_metrics": scalar_metrics,
        "series": _aggregate_series(runs),
        # 账本累计额只比较跑满相同期限的运行，破产轨迹由生存指标单独表达。
        "ledger_by_category": _aggregate_ledger(
            [run for run in runs if run["summary"]["outcome"] == "completed"]
        ),
    }


def _run_scalars(run: dict[str, Any]) -> dict[str, Any]:
    values = dict(run["summary"])
    for scope in ("agent", "environment"):
        costs = run["breakdowns"].get(f"{scope}_api_cost_by_currency", {})
        for currency, amount in costs.items():
            values[f"{scope}_api_cost_{currency}"] = amount
    return values


def _eligible_runs(
    metric: str, runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if metric in ALL_RUN_METRICS or metric.startswith(
        ("agent_api_cost_", "environment_api_cost_")
    ):
        return runs
    return [run for run in runs if run["summary"]["outcome"] == "completed"]


def _aggregate_series(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: defaultdict[tuple[str, str, int], list[float]] = defaultdict(list)
    for run in runs:
        for series_name, points in run["series"].items():
            for point in points:
                day = int(point["day"])
                for metric, value in point.items():
                    if metric != "day" and _is_number(value):
                        values[(series_name, metric, day)].append(float(value))
    return [
        {
            "series": series,
            "metric": metric,
            "day": day,
            **_describe(observations),
        }
        for (series, metric, day), observations in sorted(values.items())
    ]


def _aggregate_ledger(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not runs:
        return []
    categories = sorted(
        {
            category
            for run in runs
            for category in run["breakdowns"]["ledger_by_category"]
        }
    )
    return [
        {
            "category": category,
            **_describe([
                float(run["breakdowns"]["ledger_by_category"].get(category, 0.0))
                for run in runs
            ]),
        }
        for category in categories
    ]


def _compare_group(
    baseline: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, Any]:
    baseline_metrics = {
        metric["metric"]: metric for metric in baseline["scalar_metrics"]
    }
    treatment_metrics = {
        metric["metric"]: metric for metric in treatment["scalar_metrics"]
    }
    scalar_differences = []
    for metric in sorted(baseline_metrics.keys() & treatment_metrics.keys()):
        baseline_metric = baseline_metrics[metric]
        treatment_metric = treatment_metrics[metric]
        difference = treatment_metric["mean"] - baseline_metric["mean"]
        scalar_differences.append(
            {
                "metric": metric,
                "baseline_n": baseline_metric["n"],
                "treatment_n": treatment_metric["n"],
                "baseline_mean": baseline_metric["mean"],
                "treatment_mean": treatment_metric["mean"],
                "absolute_difference": difference,
                "relative_difference": safe_ratio(
                    difference, abs(baseline_metric["mean"])
                ),
            }
        )

    baseline_ledger = {
        item["category"]: item for item in baseline["ledger_by_category"]
    }
    treatment_ledger = {
        item["category"]: item for item in treatment["ledger_by_category"]
    }
    categories = sorted(baseline_ledger.keys() | treatment_ledger.keys())
    ledger_differences = []
    for category in categories:
        baseline_mean = baseline_ledger.get(category, {}).get("mean", 0.0)
        treatment_mean = treatment_ledger.get(category, {}).get("mean", 0.0)
        ledger_differences.append(
            {
                "category": category,
                "baseline_n": baseline_ledger.get(category, {}).get("n", 0),
                "treatment_n": treatment_ledger.get(category, {}).get("n", 0),
                "baseline_mean": baseline_mean,
                "treatment_mean": treatment_mean,
                "cash_difference_contribution": treatment_mean - baseline_mean,
            }
        )

    return {
        "baseline_group": baseline["group"],
        "treatment_group": treatment["group"],
        "bankruptcy_rate_difference": (
            treatment["bankruptcy_rate"] - baseline["bankruptcy_rate"]
        ),
        "scalar_differences": scalar_differences,
        "ledger_differences": ledger_differences,
    }


def _describe(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else None,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _is_number(value: Any) -> bool:
    # bool 是 int 的子类，但破产标记不应被当作 0/1 标量重复聚合。
    return isinstance(value, (int, float)) and not isinstance(value, bool)
