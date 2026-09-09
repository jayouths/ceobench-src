"""企业报价工具契约回归测试。"""

import json

import pytest

import saas_bench.simulator.enterprise as enterprise_module
from saas_bench.simulator.config import BenchmarkConfig
from saas_bench.simulator.enterprise import create_negotiation_thread


def _create_active_enterprise_thread(conn) -> tuple[int, int]:
    cursor = conn.execute(
        """
        INSERT INTO customers (
            customer_type, group_id, created_day,
            steepness_left, steepness_right, c_max, q_max, q_min,
            usage_demand, quality_sensitivity, price_sensitivity,
            willingness_to_pay, usage_scale, patience, seat_count, email
        )
        VALUES (
            'large', 'E1', 0,
            0.01, 0.01, 200.0, 0.9, 0.1,
            10.0, 0.5, 0.5,
            200.0, 10.0, 0.5, 20, 'deal-test@example.com'
        )
        """
    )
    customer_id = cursor.lastrowid
    conn.execute(
        """
        INSERT INTO customer_state (
            customer_id, satisfaction, relationship,
            current_steepness_left, current_steepness_right,
            current_c_max, current_q_max, current_q_min
        )
        VALUES (?, 0.5, 0.5, 0.01, 0.01, 200.0, 0.9, 0.1)
        """,
        (customer_id,),
    )
    conn.execute(
        """
        INSERT INTO subscriptions (
            customer_id, plan, listed_price, effective_price, seat_count,
            start_day, status, billing_day_mod30, first_billing_done,
            contract_months, contract_end_day
        )
        VALUES (?, 'B', 50.0, 50.0, 20, 0, 'subscribed', 0, 1, 1, 30)
        """,
        (customer_id,),
    )
    thread_id, _ = create_negotiation_thread(
        conn,
        customer_id,
        "renegotiation",
        0,
    )
    conn.commit()
    return customer_id, thread_id


def test_enterprise_deal_stores_the_fixed_monthly_contract(
    make_initialized_sim,
    make_agent_tools,
):
    """标准报价只包含套餐和价格，合同期限由模拟器固定为一个月。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, thread_id = _create_active_enterprise_thread(conn)

    result = make_agent_tools(conn, config).send_enterprise_deal(
        [[customer_id, [["B", 50.0]]]],
    )

    assert result.success
    saved = conn.execute(
        """
        SELECT offer_json FROM enterprise_turns
        WHERE thread_id = ? AND sender = 'agent'
        ORDER BY message_id DESC LIMIT 1
        """,
        (thread_id,),
    ).fetchone()
    assert json.loads(saved["offer_json"]) == [
        {"plan": "B", "price_per_seat": 50.0, "contract_months": 1},
    ]


@pytest.mark.parametrize(
    "offering",
    [
        ["B", 50.0, 12],
        {"plan": "B", "price_per_seat": 50.0, "contract_months": 12},
    ],
)
def test_enterprise_deal_rejects_agent_supplied_contract_duration(
    make_initialized_sim,
    make_agent_tools,
    offering,
):
    """旧格式不得被静默改成单月合同。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, thread_id = _create_active_enterprise_thread(conn)

    result = make_agent_tools(conn, config).send_enterprise_deal(
        [[customer_id, [offering]]],
    )

    assert not result.success
    assert "month-to-month" in result.message
    agent_turns = conn.execute(
        """
        SELECT COUNT(*) FROM enterprise_turns
        WHERE thread_id = ? AND sender = 'agent'
        """,
        (thread_id,),
    ).fetchone()[0]
    assert agent_turns == 0


def test_existing_enterprise_evaluates_offer_at_promoted_price(
    make_initialized_sim,
    make_agent_tools,
    monkeypatch,
):
    """存量企业按促销后净价接受报价，成交仍保留促销前报价。"""
    benchmark_config = BenchmarkConfig(seed=42, promotion_global=20.0)
    conn, simulator, config = make_initialized_sim(
        config=benchmark_config,
        seed=42,
    )
    customer_id, thread_id = _create_active_enterprise_thread(conn)
    monkeypatch.setattr(
        enterprise_module,
        "get_qualities_for_all_plans_batch",
        lambda _conn, customer_ids, _config: {
            customer_id: {"A": 0.5, "B": 0.5, "C": 0.5}
            for customer_id in customer_ids
        },
    )

    result = make_agent_tools(conn, config).send_enterprise_deal(
        [[customer_id, [["B", 100.0]]]],
    )
    assert result.success

    reply_day = conn.execute(
        """
        SELECT next_reply_day FROM enterprise_turns
        WHERE thread_id = ? ORDER BY message_id DESC LIMIT 1
        """,
        (thread_id,),
    ).fetchone()["next_reply_day"]
    simulator.current_day = reply_day
    simulator._process_scheduled_replies(simulator.get_current_config(), [])

    subscription = conn.execute(
        """
        SELECT listed_price, promotion, effective_price
        FROM subscriptions WHERE customer_id = ?
        """,
        (customer_id,),
    ).fetchone()
    assert tuple(subscription) == pytest.approx((100.0, 20.0, 80.0))
