"""确定性证据卡片生成测试。"""

import pytest

from saas_bench.agents.bash_agent.analysis.evidence_cards import (
    _format_metric_value,
    _format_observation,
    _format_observation_fact,
    build_evidence_cards,
    state_core_evidence_ids,
)
from saas_bench.agents.bash_agent.analysis.models import Role
from saas_bench.agents.bash_agent.analysis.signals import SignalCollector
from saas_bench.agents.bash_agent.analysis.signal_models import UnavailableReason
from saas_bench.simulator.public_week_snapshot import build_public_week_snapshot


def _direct_query(conn):
    def query(sql):
        return [dict(row) for row in conn.execute(sql).fetchall()]

    return query


def test_cards_keep_complete_comparisons_and_do_not_expose_other_roles(
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim(seed=42)
    signals = SignalCollector(_direct_query(conn)).collect(
        build_public_week_snapshot(conn, 0)
    )

    cards = build_evidence_cards(signals, Role.FINANCE)
    metrics = {card.metric for card in cards}

    assert "finance.current_cash" in metrics
    assert "finance.net_cash_flow" in metrics
    assert "finance.runway.coverage_days" not in metrics
    assert not any(".current.value" in metric for metric in metrics)
    assert all(card.id.startswith("FIN-") for card in cards)
    assert all(card.metric.startswith("finance.") for card in cards)

    core_metrics = {
        card.metric for card in cards
        if card.id in state_core_evidence_ids(cards, Role.FINANCE)
    }
    assert core_metrics == {
        "finance.current_cash",
        "finance.operating_revenue.total",
        "finance.net_cash_flow",
        "finance.costs.recurring_total",
        "finance.service_delivery_margin",
        "finance.runway.cash_runway_days",
    }


def test_card_does_not_invent_direction_without_comparison_data(
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim(seed=42)
    signals = SignalCollector(_direct_query(conn)).collect(
        build_public_week_snapshot(conn, 0)
    )

    cards = build_evidence_cards(signals, Role.PRODUCT)
    capacity_tier = next(
        card for card in cards
        if card.metric == "product.configuration.capacity_tier"
    )

    assert capacity_tier.direction is None
    assert "不能判断变化方向" in capacity_tier.fact
    assert capacity_tier.window == "当前时点与上周时点"


def test_context_and_status_records_are_not_split_into_misleading_cards(
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim(seed=42)
    signals = SignalCollector(_direct_query(conn)).collect(
        build_public_week_snapshot(conn, 0)
    )

    market_metrics = {
        card.metric for card in build_evidence_cards(signals, Role.MARKET)
    }
    product_metrics = {
        card.metric for card in build_evidence_cards(signals, Role.PRODUCT)
    }

    assert "market.macro_condition" in market_metrics
    assert not any(
        metric.startswith("market.macro_condition.") for metric in market_metrics
    )
    assert not any(metric.endswith(".group") for metric in product_metrics)


def test_metric_values_include_unambiguous_units():
    assert _format_metric_value("finance.service_delivery_margin", -5.7129) == (
        "-571.29%"
    )
    assert _format_metric_value("finance.current_cash", 996637.838) == (
        "996637.838 模拟货币单位"
    )
    assert _format_metric_value("product.reliability.peak_p95_ms", 236.45) == (
        "236.45 毫秒"
    )


def test_churn_rate_without_customer_base_has_specific_status_text():
    assert _format_observation(
        "customer.churn.weekly_rates.enterprise_accounts",
        {
            "value": None,
            "status": "not_applicable",
            "reason": "no_customer_base",
        },
    ) == "无客户基数"
    assert _format_observation_fact(
        "customer.churn.weekly_rates.enterprise_accounts",
        {
            "value": None,
            "status": "not_applicable",
            "reason": "no_customer_base",
        },
        "本期",
    ) == "本期无客户基数，该指标不计算"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (UnavailableReason.INCOMPLETE_OBSERVATION_WINDOW, "观察窗口尚未完整"),
        (UnavailableReason.MISSING_PREVIOUS_SNAPSHOT, "缺少上一期状态快照"),
        (UnavailableReason.MISSING_PUBLIC_DATA, "缺少公开经营数据"),
        (UnavailableReason.NO_EFFECTIVE_LEADS, "无有效线索"),
        (UnavailableReason.NO_RAW_LEADS, "无原始线索"),
        (UnavailableReason.NO_OPERATING_REVENUE, "无经营收入"),
        (UnavailableReason.NO_RECURRING_NET_BURN, "未形成持续经常性净消耗"),
        (UnavailableReason.NO_CUSTOMER_BASE, "无客户基数"),
        (UnavailableReason.NO_OPEN_ISSUES, "无未解决工单"),
        (UnavailableReason.NO_RESOLVED_ISSUES, "无已解决工单"),
        (UnavailableReason.NO_OPEN_ENTERPRISE_NEGOTIATIONS, "无进行中的企业谈判"),
    ],
)
def test_each_unavailable_reason_has_specific_chinese_text(reason, expected):
    status = (
        "insufficient_data"
        if reason in {
            UnavailableReason.INCOMPLETE_OBSERVATION_WINDOW,
            UnavailableReason.MISSING_PREVIOUS_SNAPSHOT,
            UnavailableReason.MISSING_PUBLIC_DATA,
        }
        else "not_applicable"
    )
    assert _format_observation(
        "example.metric",
        {"value": None, "status": status, "reason": reason.value},
    ) == expected
