"""将单次实验指标导出为 JSON 和统一长表 CSV。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


LONG_TABLE_COLUMNS = (
    "run_id",
    "experiment_name",
    "seed",
    "section",
    "metric",
    "dimension",
    "dimension_value",
    "day",
    "value",
    "unit",
)


UNITS = {
    "final_cash": "simulated_currency",
    "max_cash_drawdown_absolute": "simulated_currency",
    "max_cash_drawdown_rate": "ratio",
    "final_mrr": "simulated_currency_per_month",
    "terminal_28d_average_weekly_net_cash_flow": "simulated_currency_per_week",
    "terminal_28d_subscription_revenue": "simulated_currency",
    "global_development_spend": "simulated_currency",
    "targeted_development_spend": "simulated_currency",
    "research_project_spend": "simulated_currency",
    "terminal_28d_individual_churn_rate": "ratio",
    "terminal_28d_enterprise_seat_churn_rate": "ratio",
    "mrr_segment_hhi": "index",
    "mrr_largest_segment_share": "ratio",
    "total_downtime_minutes": "minutes",
    "mean_error_rate": "ratio",
    "mean_p95_latency_ms": "milliseconds",
    "max_p95_latency_ms": "milliseconds",
    "mean_capacity_utilization": "ratio",
    "advertising_leads_per_dollar": "leads_per_simulated_currency",
    "targeted_development_share": "ratio",
    "cash_prediction_mape": "ratio",
    "cash_prediction_interval_coverage": "ratio",
    "cash_prediction_mean_relative_interval_width": "ratio",
    "cash_prediction_mean_signed_error": "simulated_currency",
}

SERIES_UNITS = {
    ("cash_daily", "value"): "simulated_currency",
    ("subscription_daily", "mrr"): "simulated_currency_per_month",
    ("subscription_daily", "individual_accounts"): "count",
    ("subscription_daily", "enterprise_seats"): "count",
    ("service_daily", "total_usage_units"): "usage_units",
    ("service_daily", "p95_latency_ms"): "milliseconds",
    ("service_daily", "error_rate"): "ratio",
    ("service_daily", "downtime_minutes"): "minutes",
    ("service_daily", "capacity_tier"): "tier",
    ("service_daily", "capacity_units"): "usage_units",
    ("service_daily", "capacity_utilization"): "ratio",
    ("service_daily", "overloaded"): "boolean",
}


def export_run_metrics(
    metrics: dict[str, Any], output_dir: Path | str
) -> tuple[Path, Path]:
    """保存完整 JSON，并将标量、时序和分组数据展开为统一长表。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics_long.csv"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    rows = metrics_to_long_rows(metrics)
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LONG_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def metrics_to_long_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """展开成适合 pandas、统计检验和绘图直接拼接的行结构。"""
    identity = {
        "run_id": metrics["run"].get("run_id"),
        "experiment_name": metrics["run"].get("experiment_name"),
        "seed": metrics["run"].get("seed"),
    }
    rows = []
    for metric, value in metrics["summary"].items():
        if isinstance(value, (dict, list)):
            continue
        rows.append(_row(identity, "summary", metric, value))

    for series_name, points in metrics["series"].items():
        for point in points:
            day = point.get("day")
            for metric, value in point.items():
                if metric == "day" or isinstance(value, (dict, list)):
                    continue
                rows.append(
                    _row(
                        identity,
                        series_name,
                        metric,
                        value,
                        day=day,
                        unit=SERIES_UNITS.get((series_name, metric)),
                    )
                )

    for category, value in metrics["breakdowns"]["ledger_by_category"].items():
        rows.append(
            _row(
                identity,
                "ledger_by_category",
                "cash_amount",
                value,
                dimension="category",
                dimension_value=category,
                unit="simulated_currency",
            )
        )
    for usage in metrics["breakdowns"]["module_usage"]:
        dimension_value = f'{usage["component"]}:{usage["model"]}'
        for metric in (
            "call_count",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "elapsed_seconds",
        ):
            rows.append(
                _row(
                    identity,
                    "module_usage",
                    metric,
                    usage[metric],
                    dimension="component_model",
                    dimension_value=dimension_value,
                    unit=(
                        "tokens"
                        if metric.endswith("_tokens")
                        else "seconds"
                        if metric == "elapsed_seconds"
                        else "count"
                    ),
                )
            )
        for currency, amount in usage["cost_by_currency"].items():
            rows.append(
                _row(
                    identity,
                    "module_usage",
                    "api_cost",
                    amount,
                    dimension="component_model_currency",
                    dimension_value=f"{dimension_value}:{currency}",
                    unit=currency,
                )
            )
    for scope in ("agent", "environment"):
        costs = metrics["breakdowns"].get(f"{scope}_api_cost_by_currency", {})
        for currency, amount in costs.items():
            rows.append(
                _row(
                    identity,
                    "api_cost_total",
                    "api_cost",
                    amount,
                    dimension="scope_currency",
                    dimension_value=f"{scope}:{currency}",
                    unit=currency,
                )
            )
    for item in metrics["breakdowns"].get("segment_research_levels", []):
        rows.append(
            _row(
                identity,
                "segment_research_levels",
                "info_level",
                item["info_level"],
                dimension="group_id",
                dimension_value=item["group_id"],
                unit="level",
            )
        )
    for item in metrics["breakdowns"].get("cash_prediction_by_horizon", []):
        for metric, value in item.items():
            if metric == "horizon_days":
                continue
            rows.append(
                _row(
                    identity,
                    "cash_prediction_by_horizon",
                    metric,
                    value,
                    dimension="horizon_days",
                    dimension_value=str(item["horizon_days"]),
                )
            )
    return rows


def _row(
    identity: dict[str, Any],
    section: str,
    metric: str,
    value: Any,
    *,
    dimension: str = "",
    dimension_value: str = "",
    day: int | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        **identity,
        "section": section,
        "metric": metric,
        "dimension": dimension,
        "dimension_value": dimension_value,
        "day": "" if day is None else day,
        "value": "" if value is None else value,
        "unit": unit or UNITS.get(metric, "count"),
    }
