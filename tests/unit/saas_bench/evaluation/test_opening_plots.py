from __future__ import annotations

from saas_bench.evaluation.opening_plots import plot_opening_results


def _metrics(
    group: str,
    *,
    final_cash: float,
    final_mrr: float,
    subscriptions: int,
    enterprise_seats: int,
    weekly_cash_flow: float,
    ledger: dict[str, float],
) -> dict:
    return {
        "run": {
            "run_id": f"{group}-run",
            "experiment_name": group,
            "configured_days": 35,
            "status": "completed",
        },
        "summary": {
            "final_cash": final_cash,
            "final_mrr": final_mrr,
            "active_individual_subscriptions": subscriptions,
            "enterprise_subscription_seats": enterprise_seats,
            "terminal_28d_average_weekly_net_cash_flow": weekly_cash_flow,
        },
        "series": {
            "cash_daily": [
                {
                    "day": day,
                    "value": 1_000_000
                    + (final_cash - 1_000_000) * day / 35,
                }
                for day in range(36)
            ],
        },
        "breakdowns": {
            "ledger_by_category": ledger,
        },
    }


def test_plot_opening_results_writes_all_figures(tmp_path) -> None:
    baseline = _metrics(
        "baseline",
        final_cash=988_000,
        final_mrr=150,
        subscriptions=10,
        enterprise_seats=0,
        weekly_cash_flow=-2_000,
        ledger={"initial_funding": 1_000_000, "operations": -12_000},
    )
    analysis = _metrics(
        "analysis",
        final_cash=990_000,
        final_mrr=210,
        subscriptions=14,
        enterprise_seats=0,
        weekly_cash_flow=-1_500,
        ledger={"initial_funding": 1_000_000, "operations": -10_000},
    )

    paths = plot_opening_results(
        baseline,
        analysis,
        tmp_path,
    )

    assert {path.name for path in paths} == {
        "cash_trajectory.png",
        "operating_outcomes.png",
        "cash_gap_waterfall.png",
    }
    assert all(path.stat().st_size > 10_000 for path in paths)
