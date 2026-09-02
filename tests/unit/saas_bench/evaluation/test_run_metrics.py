"""单次运行指标和导出格式测试。"""

from __future__ import annotations

import csv
import sqlite3

import pytest

from saas_bench.evaluation.exports import export_run_metrics
from saas_bench.evaluation.run_metrics import compute_run_metrics


def _database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE ledger (
            id INTEGER PRIMARY KEY, day INTEGER, category TEXT,
            amount REAL, note TEXT
        );
        CREATE TABLE service_day (
            day INTEGER PRIMARY KEY, total_usage_units INTEGER, p95_ms REAL,
            error_rate REAL, downtime_minutes INTEGER,
            capacity_tier INTEGER, capacity_units INTEGER
        );
        CREATE TABLE _eval_subscription_day (
            day INTEGER, customer_type TEXT, group_id TEXT, plan TEXT,
            active_accounts INTEGER, active_seats INTEGER, mrr REAL
        );
        CREATE TABLE _eval_subscription_event (
            day INTEGER, customer_type TEXT, event_type TEXT, seats INTEGER
        );
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY, customer_type TEXT,
            group_id TEXT, seat_count INTEGER
        );
        CREATE TABLE subscriptions (
            customer_id INTEGER, status TEXT, end_day INTEGER
        );
        CREATE TABLE issues (status TEXT, days_open INTEGER);
        CREATE TABLE ad_channel_leads (leads_generated INTEGER, spend REAL);
        CREATE TABLE predictions (
            submit_day INTEGER, horizon_days INTEGER, metric TEXT,
            predicted_value REAL, predicted_lower REAL, predicted_upper REAL
        );
        CREATE TABLE api_costs (
            id INTEGER PRIMARY KEY, model TEXT, purpose TEXT,
            input_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
            cost_amount REAL, currency TEXT
        );
        CREATE TABLE group_info_levels (group_id TEXT, info_level INTEGER);
        """
    )
    return conn


def test_compute_run_metrics_uses_database_facts_and_fixed_windows():
    conn = _database()
    conn.executemany(
        "INSERT INTO ledger(day, category, amount, note) VALUES (?, ?, ?, ?)",
        [
            (0, "initial_funding", 1_000, "Initial funding"),
            (0, "development", -10, "Daily development spend"),
            (0, "development", -5, "Targeted dev spend"),
            (0, "research_project", -20, "Research"),
            (7, "subscription_payment", 100, "Billing"),
        ],
    )
    for day in range(35):
        usage = 110 if day == 10 else 90
        latency = 200 if day == 10 else 100
        conn.execute(
            "INSERT INTO service_day VALUES (?, ?, ?, ?, ?, ?, ?)",
            (day, usage, latency, 0.01, 1, 1, 100),
        )
        small_accounts = 10 if day <= 6 else 15
        enterprise_seats = 20 if day <= 6 else 25
        conn.executemany(
            "INSERT INTO _eval_subscription_day VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (day, "small", "S1", "A", small_accounts, small_accounts, 100),
                (day, "large", "E1", "B", 1, enterprise_seats, 300),
            ],
        )
    conn.executemany(
        "INSERT INTO _eval_subscription_event VALUES (?, ?, ?, ?)",
        [
            (7, "small", "started", 1),
            (8, "small", "started", 1),
            (9, "small", "started", 1),
            (10, "small", "started", 1),
            (11, "small", "started", 1),
            (12, "small", "ended", 1),
            (13, "small", "ended", 1),
            (7, "large", "started", 10),
            (14, "large", "ended", 5),
        ],
    )
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)",
        [(1, "small", "S1", 1), (2, "large", "E1", 25)],
    )
    conn.executemany(
        "INSERT INTO subscriptions VALUES (?, 'subscribed', NULL)", [(1,), (2,)]
    )
    conn.executemany(
        "INSERT INTO issues VALUES (?, ?)", [("open", 4), ("open", 6), ("resolved", 8)]
    )
    conn.execute("INSERT INTO ad_channel_leads VALUES (10, 100)")
    conn.execute(
        "INSERT INTO predictions VALUES (0, 28, 'cash', 900, 800, 1_000)"
    )
    conn.execute(
        "INSERT INTO api_costs VALUES (1, 'social', 'customer_post', 10, 2, 3, 0.1, 'CNY')"
    )
    conn.executemany(
        "INSERT INTO group_info_levels VALUES (?, ?)", [("S1", 2), ("E1", 0)]
    )

    metrics = compute_run_metrics(
        conn,
        config={
            "run_id": "run-1",
            "experiment_name": "analysis-test",
            "seed": 42,
            "scenario": "default",
            "total_days": 35,
            "initial_cash": 1_000,
        },
        result={"outcome": "completed", "days_run": 35},
        checkpoint=None,
        trajectory_events=[
            {
                "event_type": "llm_call",
                "component": "analysis",
                "served_model": "qwen",
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_tokens": 5,
                "reasoning_tokens": None,
                "elapsed_seconds": 2.5,
                "cost_amount": 0,
                "currency": "USD",
            }
        ],
    )

    summary = metrics["summary"]
    assert summary["final_cash"] == pytest.approx(1_065)
    assert summary["max_cash_drawdown_absolute"] == pytest.approx(35)
    assert summary["max_cash_drawdown_rate"] == pytest.approx(0.035)
    assert summary["final_mrr"] == pytest.approx(400)
    assert summary["active_individual_subscriptions"] == 15
    assert summary["enterprise_subscription_seats"] == 25
    assert summary["terminal_28d_individual_subscription_net_growth"] == 5
    assert summary["terminal_28d_enterprise_seat_net_growth"] == 5
    assert summary["terminal_28d_individual_churn_rate"] == pytest.approx(2 / 15)
    assert summary["terminal_28d_enterprise_seat_churn_rate"] == pytest.approx(5 / 30)
    assert summary["mrr_segment_hhi"] == pytest.approx(0.625)
    assert summary["mrr_largest_segment_share"] == pytest.approx(0.75)
    assert summary["total_downtime_minutes"] == 35
    assert summary["overload_days"] == 1
    assert summary["unresolved_issues"] == 2
    assert summary["mean_unresolved_issue_age_days"] == pytest.approx(5)
    assert summary["advertising_leads_per_dollar"] == pytest.approx(0.1)
    assert summary["targeted_development_share"] == pytest.approx(1 / 3)
    assert summary["research_project_spend"] == pytest.approx(20)
    assert summary["discovered_customer_segments"] == 1
    assert summary["mean_discovered_segment_research_level"] == pytest.approx(2)
    assert summary["mature_cash_predictions"] == 1
    assert metrics["breakdowns"]["cash_prediction_by_horizon"][0][
        "horizon_days"
    ] == 28

    usage = metrics["breakdowns"]["module_usage"]
    assert [item["component"] for item in usage] == ["analysis", "social_llm"]
    assert usage[0]["reasoning_tokens"] is None
    assert usage[1]["elapsed_seconds"] is None
    assert metrics["breakdowns"]["agent_api_cost_by_currency"] == {"USD": 0.0}
    assert metrics["breakdowns"]["environment_api_cost_by_currency"] == {
        "CNY": pytest.approx(0.1)
    }


def test_short_run_marks_terminal_window_metrics_as_unavailable():
    conn = _database()
    conn.execute(
        "INSERT INTO ledger(day, category, amount, note) VALUES (0, 'initial_funding', 100, '')"
    )
    conn.execute("INSERT INTO service_day VALUES (0, 0, 10, 0, 0, 1, 100)")

    metrics = compute_run_metrics(
        conn,
        config={"initial_cash": 100},
        result=None,
        checkpoint={"day": 1},
        trajectory_events=[],
    )

    assert metrics["summary"]["terminal_28d_average_weekly_net_cash_flow"] is None
    assert metrics["summary"]["terminal_28d_individual_churn_rate"] is None


def test_export_preserves_null_as_blank_in_long_table(tmp_path):
    metrics = {
        "run": {"run_id": "r", "experiment_name": "e", "seed": 42},
        "summary": {"final_cash": 100.0, "cash_prediction_mape": None},
        "series": {"cash_daily": [{"day": 0, "value": 100.0}]},
        "breakdowns": {
            "ledger_by_category": {"initial_funding": 100.0},
            "agent_api_cost_by_currency": {"USD": 0.0},
            "environment_api_cost_by_currency": {},
            "segment_research_levels": [{"group_id": "S1", "info_level": 1}],
            "cash_prediction_by_horizon": [],
            "module_usage": [
                {
                    "component": "analysis",
                    "model": "qwen",
                    "call_count": 1,
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cached_tokens": 0,
                    "reasoning_tokens": None,
                    "elapsed_seconds": 1.0,
                    "cost_by_currency": {"USD": 0.0},
                }
            ],
        },
    }

    json_path, csv_path = export_run_metrics(metrics, tmp_path)
    assert json_path.is_file()
    rows = list(csv.DictReader(csv_path.open()))
    reasoning = next(row for row in rows if row["metric"] == "reasoning_tokens")
    assert reasoning["value"] == ""
