"""构造开题绘图链路使用的五周模拟指标。"""

from __future__ import annotations

import copy
from typing import Any


MOCK_LEDGERS = {
    "baseline": {
        "advertising": -7000.0,
        "capacity": -2380.0,
        "compute": -80.0,
        "development": 0.0,
        "initial_funding": 1_000_000.0,
        "lead_acquisition_cost": -250.0,
        "operations": -2500.0,
        "subscription_payment": 450.0,
    },
    "analysis": {
        "advertising": -5200.0,
        "capacity": -2380.0,
        "compute": -70.0,
        "development": 0.0,
        "initial_funding": 1_000_000.0,
        "lead_acquisition_cost": -180.0,
        "operations": -2100.0,
        "subscription_payment": 720.0,
    },
}

MOCK_CASH_MILESTONES = {
    "baseline": {
        0: 1_000_000.0,
        7: 996_400.0,
        14: 995_495.0,
        21: 993_400.0,
        28: 990_800.0,
        35: 988_240.0,
    },
    "analysis": {
        0: 1_000_000.0,
        7: 996_637.84,
        14: 996_026.26,
        21: 994_500.0,
        28: 992_600.0,
        35: 990_790.0,
    },
}

MOCK_SUBSCRIPTION_MILESTONES = {
    "baseline": {0: 0, 7: 5, 14: 5, 21: 7, 28: 9, 35: 11},
    "analysis": {0: 0, 7: 6, 14: 6, 21: 9, 28: 12, 35: 15},
}


def build_mock_pair(
    reference_analysis: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """构造格式兼容且明确标识为模拟数据的五周对照组。"""
    return (
        _build_mock_run(reference_analysis, "baseline"),
        _build_mock_run(reference_analysis, "analysis"),
    )


def _build_mock_run(reference: dict[str, Any], group: str) -> dict[str, Any]:
    metrics = copy.deepcopy(reference)
    metrics["preview"] = {
        "mock": True,
        "purpose": "visualization_pipeline_only",
        "reference_run": reference["run"].get("run_id"),
        "mock_group": group,
    }
    metrics["run"].update(
        run_id=f"MOCK-{group}-35d",
        experiment_name=f"{group}_mock",
        configured_days=35,
        latest_fact_day=35,
    )
    ledger = MOCK_LEDGERS[group]
    final_cash = sum(ledger.values())
    metrics["series"]["cash_daily"] = _interpolate_series(
        MOCK_CASH_MILESTONES[group], "value"
    )
    metrics["series"]["subscription_daily"] = _mock_subscription_series(group)
    final_accounts = MOCK_SUBSCRIPTION_MILESTONES[group][35]
    terminal_weekly_cash_flow = round(
        (MOCK_CASH_MILESTONES[group][35] - MOCK_CASH_MILESTONES[group][7]) / 4,
        2,
    )
    metrics["summary"].update(
        final_cash=final_cash,
        max_cash_drawdown_absolute=1_000_000.0 - final_cash,
        max_cash_drawdown_rate=(1_000_000.0 - final_cash) / 1_000_000.0,
        final_mrr=float(final_accounts * 15),
        active_individual_subscriptions=final_accounts,
        enterprise_subscription_seats=0,
        terminal_28d_average_weekly_net_cash_flow=terminal_weekly_cash_flow,
    )
    metrics["breakdowns"]["ledger_by_category"] = dict(ledger)

    # 用参考运行的平均每日成本外推五周；这里只验证结果表的数据结构。
    cost_scale = 35 / int(reference["run"]["configured_days"])
    metrics["breakdowns"]["module_usage"] = [
        item
        for item in metrics["breakdowns"]["module_usage"]
        if group == "analysis" or item["component"] != "analysis"
    ]
    for item in metrics["breakdowns"]["module_usage"]:
        for field in ("call_count", "input_tokens", "output_tokens", "cached_tokens"):
            if item.get(field) is not None:
                item[field] = round(item[field] * cost_scale)
        if item.get("reasoning_tokens") is not None:
            item["reasoning_tokens"] = round(item["reasoning_tokens"] * cost_scale)
        if item.get("elapsed_seconds") is not None:
            item["elapsed_seconds"] *= cost_scale
        item["cost_by_currency"] = {
            currency: amount * cost_scale
            for currency, amount in item["cost_by_currency"].items()
        }
    agent_cost = sum(
        item["cost_by_currency"].get("USD", 0.0)
        for item in metrics["breakdowns"]["module_usage"]
        if item["component"] in {"bash_agent", "analysis"}
    )
    metrics["breakdowns"]["agent_api_cost_by_currency"] = {"USD": agent_cost}
    return metrics


def _mock_subscription_series(group: str) -> list[dict[str, Any]]:
    points = []
    account_series = _interpolate_series(
        MOCK_SUBSCRIPTION_MILESTONES[group], "individual_accounts", integer=True
    )
    for point in account_series:
        accounts = point["individual_accounts"]
        points.append(
            {
                "day": point["day"],
                "mrr": float(accounts * 15),
                "individual_accounts": accounts,
                "enterprise_seats": 0,
            }
        )
    return points


def _interpolate_series(
    milestones: dict[int, float | int], field: str, *, integer: bool = False
) -> list[dict[str, Any]]:
    days = sorted(milestones)
    points = []
    for start, end in zip(days, days[1:]):
        start_value = float(milestones[start])
        end_value = float(milestones[end])
        for day in range(start, end):
            progress = (day - start) / (end - start)
            value = start_value + (end_value - start_value) * progress
            points.append({"day": day, field: round(value) if integer else value})
    final_day = days[-1]
    final_value = milestones[final_day]
    points.append(
        {"day": final_day, field: int(final_value) if integer else float(final_value)}
    )
    return points
