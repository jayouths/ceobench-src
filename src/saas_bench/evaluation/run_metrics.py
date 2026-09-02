"""从单次运行产物计算经营、风险和模型成本指标。"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from saas_bench.runtime.db_protection import load_session_db

from .metric_definitions import (
    TERMINAL_WINDOW_DAYS,
    calculate_drawdowns,
    safe_ratio,
    terminal_window_start,
)


METRICS_FORMAT_VERSION = 1


def evaluate_run(run_dir: Path | str) -> dict[str, Any]:
    """读取一个运行目录并返回可复算的单次实验指标。"""
    run_dir = Path(run_dir).expanduser().resolve()
    config = _read_json(run_dir / "config.json", required=True)
    result = _read_json(run_dir / "result.json", required=False)
    checkpoint = _read_json(run_dir / "checkpoint.json", required=False)
    trajectory = _read_jsonl(_find_log(run_dir, "trajectory_*.jsonl"))

    database_path = run_dir / "world.nmdb"
    if not database_path.is_file():
        raise FileNotFoundError(f"Missing run database: {database_path}")
    conn = load_session_db(database_path, in_memory=False)
    try:
        # 离线评价只读经营世界和私有评价事实，绝不回写实验状态。
        conn.execute("PRAGMA query_only = ON")
        return compute_run_metrics(
            conn,
            config=config,
            result=result,
            checkpoint=checkpoint,
            trajectory_events=trajectory,
        )
    finally:
        conn.close()


def compute_run_metrics(
    conn: sqlite3.Connection,
    *,
    config: dict[str, Any],
    result: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
    trajectory_events: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """基于已打开的数据库计算指标，便于独立测试所有公式。"""
    end_day = _latest_fact_day(conn)
    cash_daily = _cash_daily(conn, end_day)
    final_cash = _scalar(conn, "SELECT COALESCE(SUM(amount), 0) FROM ledger")
    initial_cash = float(config.get("initial_cash", 0.0))
    max_drawdown, max_drawdown_rate = calculate_drawdowns(
        [point["value"] for point in cash_daily], initial_cash=initial_cash
    )

    stock_daily = _subscription_daily(conn, end_day)
    final_stock = stock_daily[-1] if stock_daily else {
        "day": end_day,
        "mrr": 0.0,
        "individual_accounts": 0,
        "enterprise_seats": 0,
    }
    window_start = terminal_window_start(end_day)
    opening_day = window_start - 1
    opening_stock = next(
        (point for point in stock_daily if point["day"] == opening_day), None
    )
    if opening_day < 0:
        opening_stock = {
            "day": opening_day,
            "mrr": 0.0,
            "individual_accounts": 0,
            "enterprise_seats": 0,
        }
    full_terminal_window = end_day + 1 >= TERMINAL_WINDOW_DAYS

    subscription_window = _subscription_window_metrics(
        conn,
        start_day=window_start,
        end_day=end_day,
        opening_stock=opening_stock,
        full_window=full_terminal_window,
        final_stock=final_stock,
    )
    service_daily, service_summary = _service_metrics(conn)
    ledger_by_category = {
        str(row["category"]): float(row["amount"])
        for row in conn.execute(
            """
            SELECT category, SUM(amount) AS amount
            FROM ledger
            GROUP BY category
            ORDER BY category
            """
        ).fetchall()
    }
    module_usage = _module_usage(conn, trajectory_events)
    agent_costs, environment_costs = _cost_totals(module_usage)
    prediction_metrics, prediction_by_horizon = _prediction_metrics(
        conn, cash_daily, end_day
    )
    segment_research = _segment_research(conn)
    discovered_research_levels = [
        level["info_level"]
        for level in segment_research
        if level["info_level"] >= 1
    ]

    days_run = _days_run(result, checkpoint, end_day)
    outcome = (result or {}).get("outcome")
    if outcome is None:
        outcome = "bankrupt" if final_cash < 0 else "in_progress"

    summary: dict[str, Any] = {
        "outcome": outcome,
        "bankrupt": outcome == "bankrupt" or final_cash < 0,
        "survival_days": days_run,
        "final_cash": final_cash,
        "max_cash_drawdown_absolute": max_drawdown,
        "max_cash_drawdown_rate": max_drawdown_rate,
        "final_mrr": final_stock["mrr"],
        "active_individual_subscriptions": final_stock["individual_accounts"],
        "enterprise_subscription_seats": final_stock["enterprise_seats"],
        "terminal_28d_average_weekly_net_cash_flow": _terminal_cash_flow(
            conn, window_start, end_day, full_terminal_window
        ),
        "terminal_28d_subscription_revenue": _terminal_subscription_revenue(
            conn, window_start, end_day, full_terminal_window
        ),
        **subscription_window,
        "active_customer_segments": _active_customer_segments(conn),
        "discovered_customer_segments": len(discovered_research_levels),
        "mean_discovered_segment_research_level": _mean(
            discovered_research_levels
        ),
        "max_discovered_segment_research_level": max(
            discovered_research_levels, default=0
        ),
        **_mrr_concentration(conn, end_day),
        **service_summary,
        **_issue_metrics(conn),
        "advertising_leads_per_dollar": _advertising_efficiency(conn),
        **_development_spending(conn),
        **prediction_metrics,
    }

    run_status = "completed" if outcome in {"completed", "bankrupt"} else "in_progress"
    return {
        "format_version": METRICS_FORMAT_VERSION,
        "run": {
            "run_id": config.get("run_id"),
            "experiment_name": config.get("experiment_name"),
            "seed": config.get("seed"),
            "scenario": config.get("scenario"),
            "configured_days": config.get("total_days"),
            "latest_fact_day": end_day,
            "status": run_status,
        },
        "summary": summary,
        "series": {
            "cash_daily": cash_daily,
            "subscription_daily": stock_daily,
            "service_daily": service_daily,
        },
        "breakdowns": {
            "ledger_by_category": ledger_by_category,
            "module_usage": module_usage,
            "agent_api_cost_by_currency": agent_costs,
            "environment_api_cost_by_currency": environment_costs,
            "segment_research_levels": segment_research,
            "cash_prediction_by_horizon": prediction_by_horizon,
        },
    }


def _latest_fact_day(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT MAX(day) FROM (
            SELECT MAX(day) AS day FROM ledger
            UNION ALL
            SELECT MAX(day) AS day FROM service_day
            UNION ALL
            SELECT MAX(day) AS day FROM _eval_subscription_day
        )
        """
    ).fetchone()
    return int(row[0] or 0)


def _cash_daily(conn: sqlite3.Connection, end_day: int) -> list[dict[str, Any]]:
    daily_changes = {
        int(row["day"]): float(row["amount"])
        for row in conn.execute(
            "SELECT day, SUM(amount) AS amount FROM ledger GROUP BY day"
        ).fetchall()
    }
    balance = 0.0
    points = []
    for day in range(0, end_day + 1):
        balance += daily_changes.get(day, 0.0)
        points.append({"day": day, "value": balance})
    return points


def _subscription_daily(
    conn: sqlite3.Connection, end_day: int
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            day,
            COALESCE(SUM(mrr), 0) AS mrr,
            COALESCE(SUM(CASE WHEN customer_type = 'small'
                              THEN active_accounts ELSE 0 END), 0) AS individual_accounts,
            COALESCE(SUM(CASE WHEN customer_type = 'large'
                              THEN active_seats ELSE 0 END), 0) AS enterprise_seats
        FROM _eval_subscription_day
        GROUP BY day
        ORDER BY day
        """
    ).fetchall()
    by_day = {int(row["day"]): dict(row) for row in rows}
    return [
        {
            "day": day,
            "mrr": float(by_day.get(day, {}).get("mrr", 0.0)),
            "individual_accounts": int(
                by_day.get(day, {}).get("individual_accounts", 0)
            ),
            "enterprise_seats": int(
                by_day.get(day, {}).get("enterprise_seats", 0)
            ),
        }
        for day in range(0, end_day + 1)
    ]


def _subscription_window_metrics(
    conn: sqlite3.Connection,
    *,
    start_day: int,
    end_day: int,
    opening_stock: dict[str, Any] | None,
    full_window: bool,
    final_stock: dict[str, Any],
) -> dict[str, Any]:
    if not full_window or opening_stock is None:
        return {
            "terminal_28d_individual_subscription_net_growth": None,
            "terminal_28d_enterprise_seat_net_growth": None,
            "terminal_28d_individual_churn_rate": None,
            "terminal_28d_enterprise_seat_churn_rate": None,
        }

    event_rows = conn.execute(
        """
        SELECT customer_type, event_type,
               COUNT(*) AS accounts, COALESCE(SUM(seats), 0) AS seats
        FROM _eval_subscription_event
        WHERE day BETWEEN ? AND ?
        GROUP BY customer_type, event_type
        """,
        (start_day, end_day),
    ).fetchall()
    events = {
        (str(row["customer_type"]), str(row["event_type"])): dict(row)
        for row in event_rows
    }
    small_started = int(events.get(("small", "started"), {}).get("accounts", 0))
    small_ended = int(events.get(("small", "ended"), {}).get("accounts", 0))
    enterprise_started = int(
        events.get(("large", "started"), {}).get("seats", 0)
    )
    enterprise_ended = int(events.get(("large", "ended"), {}).get("seats", 0))
    return {
        "terminal_28d_individual_subscription_net_growth": (
            final_stock["individual_accounts"] - opening_stock["individual_accounts"]
        ),
        "terminal_28d_enterprise_seat_net_growth": (
            final_stock["enterprise_seats"] - opening_stock["enterprise_seats"]
        ),
        "terminal_28d_individual_churn_rate": safe_ratio(
            small_ended, opening_stock["individual_accounts"] + small_started
        ),
        "terminal_28d_enterprise_seat_churn_rate": safe_ratio(
            enterprise_ended,
            opening_stock["enterprise_seats"] + enterprise_started,
        ),
    }


def _terminal_cash_flow(
    conn: sqlite3.Connection, start_day: int, end_day: int, full_window: bool
) -> float | None:
    if not full_window:
        return None
    total = _scalar(
        conn,
        "SELECT COALESCE(SUM(amount), 0) FROM ledger WHERE day BETWEEN ? AND ?",
        (start_day, end_day),
    )
    return total / (TERMINAL_WINDOW_DAYS / 7)


def _terminal_subscription_revenue(
    conn: sqlite3.Connection, start_day: int, end_day: int, full_window: bool
) -> float | None:
    if not full_window:
        return None
    return _scalar(
        conn,
        """
        SELECT COALESCE(SUM(amount), 0)
        FROM ledger
        WHERE category = 'subscription_payment' AND day BETWEEN ? AND ?
        """,
        (start_day, end_day),
    )


def _active_customer_segments(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT c.group_id)
        FROM subscriptions AS s
        JOIN customers AS c ON c.customer_id = s.customer_id
        WHERE s.status = 'subscribed' AND s.end_day IS NULL
        """
    ).fetchone()
    return int(row[0])


def _mrr_concentration(conn: sqlite3.Connection, end_day: int) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT group_id, SUM(mrr) AS mrr
        FROM _eval_subscription_day
        WHERE day = ?
        GROUP BY group_id
        """,
        (end_day,),
    ).fetchall()
    amounts = [float(row["mrr"]) for row in rows]
    total = sum(amounts)
    if total <= 0:
        return {"mrr_segment_hhi": None, "mrr_largest_segment_share": None}
    shares = [amount / total for amount in amounts]
    return {
        "mrr_segment_hhi": sum(share * share for share in shares),
        "mrr_largest_segment_share": max(shares),
    }


def _service_metrics(
    conn: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT day, total_usage_units, p95_ms, error_rate, downtime_minutes,
               capacity_tier, capacity_units
        FROM service_day
        ORDER BY day
        """
    ).fetchall()
    series = []
    for row in rows:
        capacity = int(row["capacity_units"])
        usage = int(row["total_usage_units"])
        utilization = safe_ratio(usage, capacity)
        series.append(
            {
                "day": int(row["day"]),
                "total_usage_units": usage,
                "p95_latency_ms": float(row["p95_ms"]),
                "error_rate": float(row["error_rate"]),
                "downtime_minutes": int(row["downtime_minutes"]),
                "capacity_tier": int(row["capacity_tier"]),
                "capacity_units": capacity,
                "capacity_utilization": utilization,
                "overloaded": usage > capacity,
            }
        )
    if not series:
        return series, {
            "total_downtime_minutes": 0,
            "mean_error_rate": None,
            "mean_p95_latency_ms": None,
            "max_p95_latency_ms": None,
            "mean_capacity_utilization": None,
            "overload_days": 0,
        }
    return series, {
        "total_downtime_minutes": sum(p["downtime_minutes"] for p in series),
        "mean_error_rate": _mean(p["error_rate"] for p in series),
        "mean_p95_latency_ms": _mean(p["p95_latency_ms"] for p in series),
        "max_p95_latency_ms": max(p["p95_latency_ms"] for p in series),
        "mean_capacity_utilization": _mean(
            p["capacity_utilization"] for p in series
        ),
        "overload_days": sum(bool(p["overloaded"]) for p in series),
    }


def _issue_metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT COUNT(*) AS count, AVG(days_open) AS avg_days
        FROM issues
        WHERE status = 'open'
        """
    ).fetchone()
    return {
        "unresolved_issues": int(row["count"]),
        "mean_unresolved_issue_age_days": (
            float(row["avg_days"]) if row["avg_days"] is not None else None
        ),
    }


def _advertising_efficiency(conn: sqlite3.Connection) -> float | None:
    row = conn.execute(
        "SELECT COALESCE(SUM(leads_generated), 0), COALESCE(SUM(spend), 0) "
        "FROM ad_channel_leads"
    ).fetchone()
    return safe_ratio(float(row[0]), float(row[1]))


def _development_spending(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN category = 'development'
                              AND note = 'Daily development spend'
                         THEN -amount ELSE 0 END), 0) AS global_development,
            COALESCE(SUM(CASE WHEN category = 'development'
                              AND note = 'Targeted dev spend'
                         THEN -amount ELSE 0 END), 0) AS targeted_development,
            COALESCE(SUM(CASE WHEN category = 'research_project'
                         THEN -amount ELSE 0 END), 0) AS research_projects
        FROM ledger
        """
    ).fetchone()
    global_development = float(row["global_development"])
    targeted_development = float(row["targeted_development"])
    return {
        "targeted_development_share": safe_ratio(
            targeted_development, global_development + targeted_development
        ),
        "global_development_spend": global_development,
        "targeted_development_spend": targeted_development,
        "research_project_spend": float(row["research_projects"]),
    }


def _prediction_metrics(
    conn: sqlite3.Connection,
    cash_daily: list[dict[str, Any]],
    end_day: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cash_by_day = {point["day"]: point["value"] for point in cash_daily}
    rows = conn.execute(
        """
        SELECT submit_day, horizon_days, predicted_value,
               predicted_lower, predicted_upper
        FROM predictions
        WHERE metric = 'cash' AND submit_day + horizon_days <= ?
        ORDER BY submit_day, horizon_days
        """,
        (end_day,),
    ).fetchall()
    if not rows:
        return {
            "cash_prediction_mape": None,
            "cash_prediction_mean_signed_error": None,
            "cash_prediction_interval_coverage": None,
            "cash_prediction_mean_relative_interval_width": None,
            "mature_cash_predictions": 0,
        }, []
    observations_by_horizon: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    observations = []
    for row in rows:
        target_day = int(row["submit_day"]) + int(row["horizon_days"])
        actual = cash_by_day[target_day]
        predicted = float(row["predicted_value"])
        lower, upper = row["predicted_lower"], row["predicted_upper"]
        observation = {
            "actual": actual,
            "predicted": predicted,
            "lower": float(lower) if lower is not None else None,
            "upper": float(upper) if upper is not None else None,
        }
        observations.append(observation)
        observations_by_horizon[int(row["horizon_days"])].append(observation)

    overall = _summarize_cash_predictions(observations)
    by_horizon = [
        {
            "horizon_days": horizon,
            **_summarize_cash_predictions(horizon_observations),
        }
        for horizon, horizon_observations in sorted(observations_by_horizon.items())
    ]
    return overall, by_horizon


def _summarize_cash_predictions(
    observations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    absolute_percentage_errors = []
    signed_errors = []
    interval_coverage = []
    relative_widths = []
    count = 0
    for observation in observations:
        count += 1
        actual = observation["actual"]
        predicted = observation["predicted"]
        if actual != 0:
            absolute_percentage_errors.append(abs(predicted - actual) / abs(actual))
        signed_errors.append(predicted - actual)
        lower, upper = observation["lower"], observation["upper"]
        if lower is not None and upper is not None:
            interval_coverage.append(lower <= actual <= upper)
            if actual != 0:
                relative_widths.append((upper - lower) / abs(actual))
    return {
        "cash_prediction_mape": _mean(absolute_percentage_errors),
        "cash_prediction_mean_signed_error": _mean(signed_errors),
        "cash_prediction_interval_coverage": _mean(interval_coverage),
        "cash_prediction_mean_relative_interval_width": _mean(relative_widths),
        "mature_cash_predictions": count,
    }


def _module_usage(
    conn: sqlite3.Connection,
    trajectory_events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for event in trajectory_events:
        if event.get("event_type") != "llm_call":
            continue
        component = str(event.get("component") or "unknown")
        model = str(event.get("served_model") or event.get("requested_model") or "unknown")
        key = (component, model)
        group = groups.setdefault(key, _empty_usage(component, model))
        _add_usage_event(group, event)

    # 社交媒体属于模拟器环境调用，独立于 Agent 和创新模块统计。
    for row in conn.execute(
        """
        SELECT model, purpose, input_tokens, cached_tokens, output_tokens,
               cost_amount, currency
        FROM api_costs
        ORDER BY id
        """
    ).fetchall():
        model = str(row["model"])
        key = ("social_llm", model)
        group = groups.setdefault(key, _empty_usage("social_llm", model))
        _add_usage_event(group, dict(row), has_elapsed=False)

    return [groups[key] for key in sorted(groups)]


def _cost_totals(
    module_usage: Iterable[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    agent_costs: defaultdict[str, float] = defaultdict(float)
    environment_costs: defaultdict[str, float] = defaultdict(float)
    for usage in module_usage:
        target = (
            environment_costs
            if usage["component"] == "social_llm"
            else agent_costs
        )
        for currency, amount in usage["cost_by_currency"].items():
            target[currency] += float(amount)
    return dict(sorted(agent_costs.items())), dict(sorted(environment_costs.items()))


def _segment_research(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {
            "group_id": str(row["group_id"]),
            "info_level": int(row["info_level"]),
        }
        for row in conn.execute(
            "SELECT group_id, info_level FROM group_info_levels ORDER BY group_id"
        ).fetchall()
    ]


def _empty_usage(component: str, model: str) -> dict[str, Any]:
    return {
        "component": component,
        "model": model,
        "call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "elapsed_seconds": 0.0,
        "cost_by_currency": {},
    }


def _add_usage_event(
    group: dict[str, Any], event: dict[str, Any], *, has_elapsed: bool = True
) -> None:
    group["call_count"] += 1
    for field in ("input_tokens", "output_tokens", "cached_tokens"):
        group[field] += int(event.get(field) or 0)
    reasoning = event.get("reasoning_tokens")
    if group["reasoning_tokens"] is not None:
        group["reasoning_tokens"] = (
            group["reasoning_tokens"] + int(reasoning)
            if reasoning is not None
            else None
        )
    if has_elapsed:
        group["elapsed_seconds"] += float(event.get("elapsed_seconds") or 0.0)
    else:
        group["elapsed_seconds"] = None
    currency = event.get("currency")
    if currency:
        costs = group["cost_by_currency"]
        costs[str(currency)] = costs.get(str(currency), 0.0) + float(
            event.get("cost_amount") or 0.0
        )


def _days_run(
    result: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
    end_day: int,
) -> int:
    if result and result.get("days_run") is not None:
        return int(result["days_run"])
    if checkpoint and checkpoint.get("day") is not None:
        return int(checkpoint["day"])
    return end_day + 1


def _scalar(
    conn: sqlite3.Connection, sql: str, parameters: tuple[Any, ...] = ()
) -> float:
    return float(conn.execute(sql, parameters).fetchone()[0])


def _mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    return sum(materialized) / len(materialized) if materialized else None


def _read_json(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing run artifact: {path}")
        return None
    return json.loads(path.read_text())


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _find_log(run_dir: Path, pattern: str) -> Path | None:
    matches = sorted((run_dir / "logs").glob(pattern))
    if len(matches) > 1:
        raise RuntimeError(f"Expected one {pattern} log in {run_dir}, found {len(matches)}")
    return matches[0] if matches else None
