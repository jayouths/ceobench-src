"""确定性证据卡片生成测试。"""

from saas_bench.agents.bash_agent.analysis.evidence_cards import (
    _format_metric_value,
    build_evidence_cards,
    state_core_evidence_ids,
)
from saas_bench.agents.bash_agent.analysis.models import Role
from saas_bench.agents.bash_agent.analysis.signals import SignalCollector
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
