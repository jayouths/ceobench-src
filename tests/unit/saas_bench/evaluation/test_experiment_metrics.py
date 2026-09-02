"""多次运行聚合、组间比较和导出格式测试。"""

from __future__ import annotations

import csv

import pytest

from saas_bench.evaluation.experiment_metrics import aggregate_experiments
from saas_bench.evaluation.exports import export_experiment_metrics


def _run(
    run_id: str,
    *,
    outcome: str = "completed",
    final_cash: float = 100.0,
    survival_days: int = 35,
    ledger_revenue: float = 20.0,
) -> dict:
    return {
        "run": {
            "run_id": run_id,
            "status": "completed",
            "configured_days": 35,
        },
        "summary": {
            "outcome": outcome,
            "bankrupt": outcome == "bankrupt",
            "survival_days": survival_days,
            "final_cash": final_cash,
            "final_mrr": final_cash / 2,
        },
        "series": {
            "cash_daily": [
                {"day": 0, "value": 100.0},
                {"day": survival_days - 1, "value": final_cash},
            ]
        },
        "breakdowns": {
            "ledger_by_category": {
                "initial_funding": 100.0,
                "subscription_payment": ledger_revenue,
            },
            "agent_api_cost_by_currency": {"USD": 1.0},
            "environment_api_cost_by_currency": {"USD": 0.25},
        },
    }


def _metric(group: dict, name: str) -> dict:
    return next(item for item in group["scalar_metrics"] if item["metric"] == name)


def test_aggregate_describes_groups_and_compares_with_baseline():
    metrics = aggregate_experiments(
        {
            "baseline": [_run("b1", final_cash=100), _run("b2", final_cash=120)],
            "analysis": [_run("a1", final_cash=140), _run("a2", final_cash=160)],
        },
        baseline_group="baseline",
    )

    groups = {group["group"]: group for group in metrics["groups"]}
    assert _metric(groups["baseline"], "final_cash") == {
        "metric": "final_cash",
        "n": 2,
        "mean": 110.0,
        "std": pytest.approx(14.1421356237),
        "median": 110.0,
        "min": 100.0,
        "max": 120.0,
    }
    difference = next(
        item
        for item in metrics["comparisons"][0]["scalar_differences"]
        if item["metric"] == "final_cash"
    )
    assert difference["absolute_difference"] == pytest.approx(40.0)
    assert difference["relative_difference"] == pytest.approx(40 / 110)
    assert metrics["inference_status"] == "not_configured"


def test_bankrupt_runs_do_not_pollute_same_horizon_metrics_or_ledger():
    metrics = aggregate_experiments(
        {
            "baseline": [
                _run("complete", final_cash=100, ledger_revenue=20),
                _run(
                    "bankrupt",
                    outcome="bankrupt",
                    final_cash=-10,
                    survival_days=14,
                    ledger_revenue=999,
                ),
            ]
        },
        baseline_group="baseline",
    )

    group = metrics["groups"][0]
    assert group["bankruptcy_rate"] == pytest.approx(0.5)
    assert _metric(group, "final_cash")["n"] == 1
    assert _metric(group, "final_cash")["mean"] == pytest.approx(100)
    assert _metric(group, "survival_days")["n"] == 2
    assert all(item["metric"] != "bankrupt" for item in group["scalar_metrics"])
    revenue = next(
        item
        for item in group["ledger_by_category"]
        if item["category"] == "subscription_payment"
    )
    assert revenue["n"] == 1
    assert revenue["mean"] == pytest.approx(20)


def test_aggregate_rejects_incomplete_or_mismatched_runs():
    incomplete = _run("running")
    incomplete["run"]["status"] = "in_progress"
    with pytest.raises(ValueError, match="not finalized"):
        aggregate_experiments({"baseline": [incomplete]}, baseline_group="baseline")

    different_duration = _run("long")
    different_duration["run"]["configured_days"] = 497
    with pytest.raises(ValueError, match="same configured duration"):
        aggregate_experiments(
            {"baseline": [_run("short")], "analysis": [different_duration]},
            baseline_group="baseline",
        )


def test_export_writes_group_summary_comparison_and_series_tables(tmp_path):
    metrics = aggregate_experiments(
        {"baseline": [_run("b")], "analysis": [_run("a", final_cash=140)]},
        baseline_group="baseline",
    )

    paths = export_experiment_metrics(metrics, tmp_path)
    assert all(path.is_file() for path in paths)

    summary_rows = list(csv.DictReader((tmp_path / "group_summary.csv").open()))
    assert summary_rows[0]["run_count"] == "1"
    comparison_rows = list(
        csv.DictReader((tmp_path / "group_comparisons.csv").open())
    )
    assert {row["section"] for row in comparison_rows} == {
        "outcome",
        "summary",
        "ledger_by_category",
    }
    assert (tmp_path / "group_series.csv").read_text().startswith("group,")
