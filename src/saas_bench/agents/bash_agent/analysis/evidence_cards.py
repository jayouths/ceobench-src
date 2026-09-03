"""把嵌套经营信号转换为角色模型容易使用的确定性证据卡片。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .models import Direction, EvidenceCard, Role
from .signal_catalog import SIGNAL_CATALOG, SignalDefinition
from .signal_models import AnalysisSignals, DataStatus


_ROLE_PREFIXES = {
    Role.MARKET: "MAR",
    Role.FINANCE: "FIN",
    Role.PRODUCT: "PRO",
    Role.CUSTOMER: "CUS",
}

# 这些字段用于说明一条列表记录属于哪个渠道、套餐或客户，但本身不构成经营证据。
_CONTEXT_KEYS = {
    "channel_id",
    "customer_id",
    "customer_type",
    "day",
    "group",
    "group_id",
    "plan",
    "post_id",
    "project_id",
    "source",
    "source_type",
    "thread_id",
}
_TECHNICAL_KEYS = {"coverage_days", "details_truncated", "ledger_max_id"}

# 五维状态判断每周都必须看到的最小事实集合。角色 LLM 可以选择额外重点，
# 但不能因一次随机筛选遗漏现金、宕机或客户存量等基础证据。
_STATE_CORE_METRICS = {
    Role.MARKET: (
        "market.effective_leads.individual",
        "market.effective_leads.enterprise_accounts",
        "market.effective_leads.total_accounts",
        "market.paid_acquisition.overall.effective_cpl",
        "market.social_feedback.post_count",
        "market.macro_condition",
    ),
    Role.FINANCE: (
        "finance.current_cash",
        "finance.operating_revenue.total",
        "finance.net_cash_flow",
        "finance.costs.recurring_total",
        "finance.service_delivery_margin",
        "finance.runway.cash_runway_days",
    ),
    Role.PRODUCT: (
        "product.usage.total_units",
        "product.capacity.average_utilization",
        "product.capacity.peak_utilization",
        "product.capacity.overload_days",
        "product.reliability.peak_p95_ms",
        "product.reliability.peak_error_rate",
        "product.reliability.downtime_minutes",
        "product.reliability.outage_days",
        "product.configuration.daily_operations_spend",
        "product.configuration.daily_development_spend",
    ),
    Role.CUSTOMER: (
        "customer.customer_base.active_individual_accounts",
        "customer.customer_base.active_enterprise_accounts",
        "customer.customer_base.active_enterprise_seats",
        "customer.customer_base.individual_net_change",
        "customer.customer_base.enterprise_seat_net_change",
        "customer.churn.weekly_account_churn_rate",
        "customer.churn.trailing_28d_account_churn_rate",
        "customer.issues.open_issues",
        "customer.issues.open_over_7_days",
        "customer.issues.average_resolution_days",
        "customer.enterprise_negotiations.open_threads",
        "customer.enterprise_negotiations.average_waiting_days",
    ),
}


def build_evidence_cards(signals: AnalysisSignals, role: Role) -> list[EvidenceCard]:
    """为一个角色生成稳定、有序且不能由 LLM 改写的事实集合。"""

    payload = getattr(signals, role.value).model_dump(mode="json")
    prefix = _ROLE_PREFIXES[role]
    cards = []
    for index, (metric, value, context) in enumerate(
        _walk_signal(payload, role.value, {}),
        start=1,
    ):
        definition = _definition_for(metric)
        cards.append(EvidenceCard(
            id=f"{prefix}-{index:03d}",
            metric=metric,
            meaning=_metric_meaning(metric, definition),
            fact=_format_fact(metric, value, context),
            window=definition.window,
            direction=_comparison_direction(value),
        ))
    if not cards:
        raise ValueError(f"no evidence cards generated for role {role.value}")
    return cards


def state_core_evidence_ids(cards: list[EvidenceCard], role: Role) -> list[str]:
    """返回状态重构不可缺少的角色证据，顺序由显式业务目录决定。"""

    cards_by_metric = {card.metric: card.id for card in cards}
    return [
        cards_by_metric[metric]
        for metric in _STATE_CORE_METRICS[role]
        if metric in cards_by_metric
    ]


def _walk_signal(
    value: Any,
    path: str,
    context: dict[str, Any],
) -> Iterator[tuple[str, Any, dict[str, Any]]]:
    """遇到完整观察或比较对象时停止下钻，避免一项指标被拆成多个矛盾事实。"""

    if _is_metric_comparison(value) or _is_numeric_observation(value):
        yield path, value, context
        return

    # 这几类记录必须整体解释。拆成 status、日期等叶子字段会让模型把
    # “观察覆盖天数”之类的技术元数据误认为经营结果。
    if isinstance(value, dict) and _is_atomic_record(path):
        yield path, value, context
        return

    if isinstance(value, dict):
        local_context = {
            **context,
            **{
                key: item
                for key, item in value.items()
                if key in _CONTEXT_KEYS and not isinstance(item, (dict, list))
            },
        }
        for key, item in value.items():
            if key in _CONTEXT_KEYS or key in _TECHNICAL_KEYS:
                continue
            yield from _walk_signal(item, f"{path}.{key}", local_context)
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_signal(item, f"{path}[{index}]", context)
        return

    # null 只表示某个普通字段缺失；比较对象内部的缺失状态已由上面的整体卡片表达。
    if value is not None:
        yield path, value, context


def _is_numeric_observation(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"value", "status"}


def _is_metric_comparison(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and {"current", "previous", "comparison_status"}.issubset(value)
    )


def _is_atomic_record(path: str) -> bool:
    return (
        path == "market.macro_condition"
        or path.startswith("product.research_pipeline.in_progress[")
        or path.startswith("product.research_pipeline.completed[")
        or path.startswith("customer.enterprise_negotiations.oldest_open_threads[")
    )


def _definition_for(metric: str) -> SignalDefinition:
    matches = [
        (path, definition)
        for path, definition in SIGNAL_CATALOG.items()
        if metric == path or metric.startswith(path + ".")
    ]
    if not matches:
        raise ValueError(f"missing signal definition for metric {metric!r}")
    return max(matches, key=lambda item: len(item[0]))[1]


def _metric_meaning(metric: str, definition: SignalDefinition) -> str:
    catalog_path = max(
        path
        for path in SIGNAL_CATALOG
        if metric == path or metric.startswith(path + ".")
    )
    detail = metric.removeprefix(catalog_path).lstrip(".")
    return definition.meaning if not detail else f"{definition.meaning}；字段 {detail}"


def _format_fact(
    metric: str,
    value: Any,
    context: dict[str, Any],
) -> str:
    context_text = _format_context(context)
    if _is_metric_comparison(value):
        current = _format_observation(metric, value["current"])
        previous = _format_observation(metric, value["previous"])
        if value["comparison_status"] == DataStatus.AVAILABLE.value:
            change = _format_metric_value(
                metric,
                value.get("absolute_change"),
                is_change=True,
            )
            relative = value.get("relative_change")
            relative_text = (
                "不可计算"
                if relative is None
                else f"{float(relative) * 100:.2f}%"
            )
            direction = {
                "up": "上升",
                "down": "下降",
                "flat": "持平",
            }[value["direction"]]
            detail = (
                f"当前值 {current}，前期值 {previous}，绝对变化 {change}，"
                f"相对变化 {relative_text}，方向为{direction}"
            )
        else:
            detail = (
                f"当前值 {current}，前期值 {previous}；比较数据不足，"
                "不能判断变化方向"
            )
    elif _is_numeric_observation(value):
        detail = f"当前值 {_format_observation(metric, value)}"
    else:
        detail = f"当前值 {_format_metric_value(metric, value)}"

    parts = []
    if context_text:
        parts.append(context_text)
    parts.append(detail)
    return "；".join(parts)


def _format_context(context: dict[str, Any]) -> str:
    return "，".join(
        f"{key}={_format_value(value)}" for key, value in context.items()
    )


def _format_observation(metric: str, observation: dict[str, Any]) -> str:
    status = observation["status"]
    if status == DataStatus.AVAILABLE.value:
        return _format_metric_value(metric, observation["value"])
    return {
        DataStatus.INSUFFICIENT_DATA.value: "数据不足",
        DataStatus.NOT_APPLICABLE.value: "不适用",
    }[status]


def _format_metric_value(metric: str, value: Any, *, is_change: bool = False) -> str:
    """根据指标本身确定展示单位，避免模型猜测比例和金额口径。"""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if _is_percentage_metric(metric):
            suffix = "个百分点" if is_change else "%"
            return f"{float(value) * 100:.2f}{suffix}"
        rendered = _format_value(value)
        if _is_currency_metric(metric):
            return f"{rendered} 模拟货币单位"
        if "p95_ms" in metric:
            return f"{rendered} 毫秒"
        if "downtime_minutes" in metric:
            return f"{rendered} 分钟"
        if any(part in metric for part in ("runway_days", "age_days", "resolution_days", "waiting_days")):
            return f"{rendered} 天"
        return rendered
    return _format_value(value)


def _is_percentage_metric(metric: str) -> bool:
    return any(
        marker in metric
        for marker in (
            ".share",
            "service_delivery_margin",
            "utilization",
            "overload_excess",
            "error_rate",
            "churn_rate",
        )
    )


def _is_currency_metric(metric: str) -> bool:
    return any(
        marker in metric
        for marker in (
            "current_cash",
            "operating_revenue",
            "net_cash_flow",
            ".costs.",
            ".spend",
            "_cpl",
            "daily_operations_spend",
            "daily_development_spend",
        )
    )


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _comparison_direction(value: Any) -> Direction | None:
    if not _is_metric_comparison(value):
        return None
    direction = value.get("direction")
    return Direction(direction) if direction is not None else None
