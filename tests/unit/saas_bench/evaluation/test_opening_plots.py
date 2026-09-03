from __future__ import annotations

from saas_bench.evaluation.opening_plots import plot_opening_results
from saas_bench.evaluation.opening_preview import build_mock_pair


def test_mock_pair_is_explicit_and_ledger_reconciles() -> None:
    analysis = {
        "run": {
            "run_id": "real-analysis",
            "configured_days": 14,
            "status": "completed",
        },
        "summary": {
            "final_cash": 996_026.26,
            "final_mrr": 90.0,
            "active_individual_subscriptions": 6,
            "enterprise_subscription_seats": 0,
        },
        "series": {
            "cash_daily": [
                {"day": day, "value": 1_000_000 - day * 100}
                for day in range(15)
            ],
            "subscription_daily": [],
        },
        "breakdowns": {
            "ledger_by_category": {},
            "module_usage": [
                {
                    "component": "analysis",
                    "cost_by_currency": {"USD": 0.03},
                },
                {
                    "component": "bash_agent",
                    "cost_by_currency": {"USD": 0.40},
                },
            ],
            "agent_api_cost_by_currency": {"USD": 0.43},
        },
    }

    baseline, analysis_mock = build_mock_pair(analysis)

    for metrics in (baseline, analysis_mock):
        assert metrics["preview"]["mock"] is True
        assert metrics["run"]["configured_days"] == 35
        assert metrics["summary"]["final_cash"] == sum(
            metrics["breakdowns"]["ledger_by_category"].values()
        )
        assert metrics["series"]["cash_daily"][0]["value"] == 1_000_000
        assert len(metrics["series"]["cash_daily"]) == 36

    assert baseline["run"]["experiment_name"] == "baseline_mock"
    assert analysis_mock["run"]["experiment_name"] == "analysis_mock"
    assert baseline["series"]["cash_daily"][-1]["value"] == 988_240
    assert analysis_mock["series"]["cash_daily"][-1]["value"] == 990_790
    assert baseline["summary"]["terminal_28d_average_weekly_net_cash_flow"] == -2040
    assert analysis_mock["summary"]["terminal_28d_average_weekly_net_cash_flow"] == -1461.96


def test_plot_opening_results_writes_all_figures(tmp_path) -> None:
    reference = {
        "run": {
            "run_id": "real-analysis",
            "configured_days": 14,
            "status": "completed",
        },
        "summary": {"final_cash": 996_026.26},
        "series": {"cash_daily": [], "subscription_daily": []},
        "breakdowns": {
            "ledger_by_category": {},
            "module_usage": [],
            "agent_api_cost_by_currency": {},
        },
    }
    baseline, analysis = build_mock_pair(reference)

    paths = plot_opening_results(
        baseline,
        analysis,
        tmp_path,
        watermark="PREVIEW - ALL DATA MOCK",
    )

    assert {path.name for path in paths} == {
        "cash_trajectory.png",
        "operating_outcomes.png",
        "cash_gap_waterfall.png",
    }
    assert all(path.stat().st_size > 10_000 for path in paths)
