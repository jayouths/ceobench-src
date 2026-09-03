"""绘制开题阶段 Baseline 与 Analysis 的核心结果图。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter


COLORS = {"baseline": "#6B7280", "analysis": "#167D6B"}
GROUP_LABELS = {"baseline": "Baseline", "analysis": "Baseline + Analysis"}
LEDGER_LABELS = {
    "subscription_payment": "Subscription revenue",
    "advertising": "Advertising",
    "lead_acquisition_cost": "Lead acquisition",
    "operations": "Operations",
    "compute": "Compute",
    "capacity": "Capacity",
    "development": "Development",
}


def load_metrics(path: Path | str) -> dict[str, Any]:
    """读取单次运行的统一指标文件。"""
    return json.loads(Path(path).read_text())


def plot_opening_results(
    baseline: dict[str, Any],
    analysis: dict[str, Any],
    output_dir: Path | str,
    *,
    watermark: str | None = None,
) -> list[Path]:
    """生成开题阶段三张核心结果图。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_pair(baseline, analysis)

    figures = (
        ("cash_trajectory.png", _plot_cash_trajectory(baseline, analysis)),
        ("operating_outcomes.png", _plot_operating_outcomes(baseline, analysis)),
        ("cash_gap_waterfall.png", _plot_cash_gap_waterfall(baseline, analysis)),
    )
    paths = []
    for filename, figure in figures:
        if watermark:
            _add_watermark(figure, watermark)
        path = output_dir / filename
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def _plot_cash_trajectory(
    baseline: dict[str, Any], analysis: dict[str, Any]
) -> Figure:
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    for group, metrics in (("baseline", baseline), ("analysis", analysis)):
        points = metrics["series"]["cash_daily"]
        axis.plot(
            [point["day"] for point in points],
            [point["value"] for point in points],
            color=COLORS[group],
            label=GROUP_LABELS[group],
            linewidth=2.4,
        )
    final_day = int(analysis["run"]["configured_days"])
    week_ticks = list(range(0, final_day + 1, 7))
    axis.set_xticks(week_ticks, [str(day // 7) for day in week_ticks])
    axis.set(title="Cash trajectory", xlabel="Simulation week", ylabel="Cash balance")
    axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    _clean_axis(axis)
    figure.tight_layout()
    return figure


def _plot_operating_outcomes(
    baseline: dict[str, Any], analysis: dict[str, Any]
) -> Figure:
    metrics = (
        ("final_mrr", "Final MRR", "${:,.0f}"),
        ("active_individual_subscriptions", "Active individual\nsubscriptions", "{:,.0f}"),
        ("enterprise_subscription_seats", "Enterprise seats", "{:,.0f}"),
        (
            "terminal_28d_average_weekly_net_cash_flow",
            "Terminal 28-day average\nweekly net cash flow",
            "${:,.0f}",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    for axis, (key, title, value_format) in zip(axes.flat, metrics, strict=True):
        values = [baseline["summary"].get(key), analysis["summary"].get(key)]
        if any(value is None for value in values):
            axis.text(
                0.5,
                0.5,
                "N/A\n(requires at least 28 days)",
                ha="center",
                va="center",
                color="#6B7280",
                fontsize=11,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            bars = axis.bar(
                [GROUP_LABELS["baseline"], GROUP_LABELS["analysis"]],
                values,
                color=[COLORS["baseline"], COLORS["analysis"]],
                width=0.58,
            )
            numeric_values = [float(value) for value in values]
            minimum = min(numeric_values)
            maximum = max(numeric_values)
            span = max(maximum - minimum, abs(maximum), abs(minimum), 1.0)
            if minimum == maximum == 0:
                axis.set_ylim(0, 1)
            elif maximum <= 0:
                axis.set_ylim(minimum - span * 0.18, 0)
            else:
                axis.set_ylim(min(0, minimum - span * 0.08), maximum + span * 0.18)
            for bar, value in zip(bars, values, strict=True):
                numeric_value = float(value)
                label_y = (
                    0.04
                    if minimum == maximum == 0
                    else numeric_value
                    + (span * 0.04 if numeric_value >= 0 else -span * 0.04)
                )
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    value_format.format(value),
                    ha="center",
                    va="bottom" if numeric_value >= 0 else "top",
                    fontsize=10,
                )
            axis.tick_params(axis="x", labelrotation=8)
            axis.grid(axis="y", alpha=0.18)
        axis.set_title(title, fontsize=11)
        _clean_axis(axis)
    figure.suptitle("Multidimensional operating outcomes", fontsize=15, y=1.01)
    figure.tight_layout()
    return figure


def _plot_cash_gap_waterfall(
    baseline: dict[str, Any], analysis: dict[str, Any]
) -> Figure:
    baseline_ledger = baseline["breakdowns"]["ledger_by_category"]
    analysis_ledger = analysis["breakdowns"]["ledger_by_category"]
    categories = sorted(
        (set(baseline_ledger) | set(analysis_ledger)) - {"initial_funding"},
        key=lambda item: abs(
            float(analysis_ledger.get(item, 0)) - float(baseline_ledger.get(item, 0))
        ),
        reverse=True,
    )
    differences = [
        float(analysis_ledger.get(category, 0))
        - float(baseline_ledger.get(category, 0))
        for category in categories
    ]
    starts = []
    running = 0.0
    for difference in differences:
        starts.append(running if difference >= 0 else running + difference)
        running += difference

    labels = [LEDGER_LABELS.get(category, category.replace("_", " ").title()) for category in categories]
    labels.append("Total cash gap")
    figure, axis = plt.subplots(figsize=(10.4, 5.6))
    bars = axis.bar(
        labels,
        differences + [running],
        bottom=starts + [0],
        color=["#238B75" if value >= 0 else "#C5524A" for value in differences]
        + ["#315B7D"],
        width=0.68,
    )
    for bar, value in zip(bars, differences + [running], strict=True):
        y = bar.get_y() + bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y + (8 if value >= 0 else -8),
            f"{value:+,.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    axis.axhline(0, color="#374151", linewidth=0.8)
    chart_top = max(
        running,
        *(
            start + max(value, 0)
            for start, value in zip(starts, differences, strict=True)
        ),
    )
    chart_bottom = min(
        0,
        *(
            start + min(value, 0)
            for start, value in zip(starts, differences, strict=True)
        ),
    )
    span = max(chart_top - chart_bottom, 1.0)
    axis.set_ylim(chart_bottom - span * 0.08, chart_top + span * 0.16)
    axis.set_ylabel("Contribution to final cash gap")
    axis.set_title("Cash gap decomposition: Analysis minus Baseline", pad=16)
    axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    axis.tick_params(axis="x", labelrotation=24)
    axis.grid(axis="y", alpha=0.18)
    _clean_axis(axis)
    figure.tight_layout()
    return figure


def _validate_pair(baseline: dict[str, Any], analysis: dict[str, Any]) -> None:
    baseline_days = baseline["run"].get("configured_days")
    analysis_days = analysis["run"].get("configured_days")
    if baseline_days != analysis_days:
        raise ValueError("Baseline and Analysis must use the same configured duration")
    for name, metrics in (("Baseline", baseline), ("Analysis", analysis)):
        if metrics["run"].get("status") != "completed":
            raise ValueError(f"{name} metrics are not finalized")


def _add_watermark(figure: Figure, text: str) -> None:
    # 预览图必须在图面上直接标明模拟数据，避免脱离目录说明后被误用。
    figure.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        rotation=24,
        fontsize=28,
        color="#B91C1C",
        alpha=0.16,
        weight="bold",
    )


def _clean_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
