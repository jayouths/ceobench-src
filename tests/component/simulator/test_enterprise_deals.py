"""企业报价工具契约回归测试。"""

import json

import pytest

import saas_bench.simulator.enterprise as enterprise_module
from saas_bench.simulator.config import BenchmarkConfig
from saas_bench.simulator.enterprise import (
    create_negotiation_thread,
    get_qualities_for_all_plans_batch,
)


def _insert_enterprise_customer(conn) -> int:
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
    return customer_id


def _create_active_enterprise_thread(conn) -> tuple[int, int]:
    customer_id = _insert_enterprise_customer(conn)
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


def _create_rejectable_enterprise_thread(
    conn,
    thread_type: str,
) -> tuple[int, int]:
    """创建可供 Agent 主动拒绝的线索或存量客户谈判。"""
    customer_id = _insert_enterprise_customer(conn)
    subscription_status = "lead" if thread_type == "new_lead" else "subscribed"
    conn.execute(
        """
        INSERT INTO subscriptions (
            customer_id, plan, listed_price, effective_price, seat_count,
            start_day, status, billing_day_mod30, first_billing_done,
            contract_months, contract_end_day
        )
        VALUES (?, 'B', 50.0, 50.0, 20, 0, ?, 0, 1, 1, 30)
        """,
        (customer_id, subscription_status),
    )
    thread_id, _ = create_negotiation_thread(
        conn,
        customer_id,
        thread_type,
        0,
    )
    conn.commit()
    return customer_id, thread_id


def test_new_enterprise_lead_uses_expected_demand_for_plan_qualities(
    make_initialized_sim,
):
    """新线索用预计用量评估零、低和足额配额，不得把未采样用量当作零需求。"""
    conn, simulator, config = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    simulator._create_enterprise_lead(customer_id, {"seat_count": 20})
    conn.execute(
        """
        UPDATE config_history
        SET tier_A = 4, tier_B = 4, tier_C = 4,
            quota_A = 0, quota_B = 100, quota_C = 200
        WHERE day = 0
        """
    )

    qualities = get_qualities_for_all_plans_batch(conn, [customer_id], config)

    # 预计需求为 10 单位/席位 × 20 席位 = 200，三档配额因子为 0、0.5、1。
    assert qualities[customer_id] == pytest.approx(
        {"A": 0.0, "B": 0.1, "C": 0.2}
    )


def test_existing_enterprise_customer_uses_sampled_usage_for_plan_qualities(
    make_initialized_sim,
):
    """存量企业客户仍以订阅中的真实用量评估配额。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, _ = _create_active_enterprise_thread(conn)
    conn.execute(
        "UPDATE subscriptions SET daily_usage_rate = 50 WHERE customer_id = ?",
        (customer_id,),
    )
    conn.execute(
        """
        UPDATE config_history
        SET tier_A = 4, tier_B = 4, tier_C = 4,
            quota_A = 0, quota_B = 25, quota_C = 50
        WHERE day = 0
        """
    )

    qualities = get_qualities_for_all_plans_batch(conn, [customer_id], config)

    # Day 0 的每周倍率为 1；真实需求 50 应得到 0、0.5、1 三档配额因子。
    assert qualities[customer_id] == pytest.approx(
        {"A": 0.0, "B": 0.1, "C": 0.2}
    )


@pytest.mark.parametrize(
    "offering",
    [
        ["B", 50.0],
        {"plan": "B", "price": 50.0},
    ],
)
def test_enterprise_deal_normalizes_the_fixed_monthly_contract(
    make_initialized_sim,
    make_agent_tools,
    offering,
):
    """支持的输入格式统一转为标准报价，合同期限固定为一个月。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, thread_id = _create_active_enterprise_thread(conn)

    result = make_agent_tools(conn, config).send_enterprise_deal(
        [[customer_id, [offering]]],
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


@pytest.mark.parametrize(
    ("offering", "expected_message"),
    [
        (["D", 50.0], "plan must be one of A, B, C"),
        (["A", -1.0], "finite non-negative number"),
        (["A", float("nan")], "finite non-negative number"),
        (["A", True], "finite non-negative number"),
        ({"price_per_seat": 50.0}, "must include plan"),
    ],
)
def test_enterprise_deal_rejects_invalid_offering_without_side_effects(
    make_initialized_sim,
    make_agent_tools,
    offering,
    expected_message,
):
    """非法报价不能消耗谈判轮次、安排回复或惩罚客户关系。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, thread_id = _create_active_enterprise_thread(conn)

    result = make_agent_tools(conn, config, day=5).send_enterprise_deal(
        [[customer_id, [offering]]],
    )

    assert not result.success
    assert expected_message in result.message
    thread = conn.execute(
        """
        SELECT COUNT(*) AS agent_turns, MAX(next_reply_day) AS next_reply_day
        FROM enterprise_turns
        WHERE thread_id = ? AND sender = 'agent'
        """,
        (thread_id,),
    ).fetchone()
    relationship = conn.execute(
        "SELECT relationship FROM customer_state WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()["relationship"]
    assert thread["agent_turns"] == 0
    assert thread["next_reply_day"] is None
    assert relationship == pytest.approx(0.5)


def test_enterprise_deal_rejects_more_than_configured_offerings(
    make_initialized_sim,
    make_agent_tools,
):
    """超出公开上限的报价必须整体失败，不能静默截断后发送。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, thread_id = _create_active_enterprise_thread(conn)
    offerings = [["A", 10.0], ["B", 20.0], ["C", 30.0], ["A", 40.0]]

    result = make_agent_tools(conn, config).send_enterprise_deal(
        [[customer_id, offerings]],
    )

    assert not result.success
    assert "offerings must contain 1-3 items" in result.message
    assert conn.execute(
        """
        SELECT COUNT(*) FROM enterprise_turns
        WHERE thread_id = ? AND sender = 'agent'
        """,
        (thread_id,),
    ).fetchone()[0] == 0


def test_enterprise_deal_rejects_non_integer_customer_id(
    make_initialized_sim,
    make_agent_tools,
):
    """客户 ID 格式错误应返回工具失败，而不是抛出转换异常。"""
    conn, _, config = make_initialized_sim(seed=42)

    result = make_agent_tools(conn, config).send_enterprise_deal(
        [["not-an-id", [["A", 10.0]]]],
    )

    assert not result.success
    assert "customer_id must be a positive integer" in result.message


@pytest.mark.parametrize(
    ("thread_type", "expected_status", "damages_reputation"),
    [
        ("new_lead", "lost", False),
        ("renegotiation", "cancelled", False),
        ("renewal", "cancelled", False),
        ("churn_prevention", "cancelled", True),
        ("plan_change", "cancelled", True),
    ],
)
def test_reject_enterprise_deal_applies_documented_terminal_effects(
    make_initialized_sim,
    make_agent_tools,
    thread_type,
    expected_status,
    damages_reputation,
):
    """主动拒绝必须结束谈判，并按谈判类型落实线索或客户流失。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, thread_id = _create_rejectable_enterprise_thread(
        conn,
        thread_type,
    )
    initial_reputation = conn.execute(
        "SELECT reputation FROM group_reputation WHERE group_id = 'E1'",
    ).fetchone()["reputation"]

    result = make_agent_tools(conn, config, day=7).reject_enterprise_deal(
        [{"customer_id": customer_id}],
    )

    assert result.success
    assert result.data["results"] == [{
        "customer_id": customer_id,
        "thread_type": thread_type,
        "success": True,
    }]
    subscription = conn.execute(
        "SELECT status, end_day FROM subscriptions WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    latest_turn = conn.execute(
        """
        SELECT closed, close_reason
        FROM enterprise_turns
        WHERE thread_id = ?
        ORDER BY message_id DESC LIMIT 1
        """,
        (thread_id,),
    ).fetchone()
    current_reputation = conn.execute(
        "SELECT reputation FROM group_reputation WHERE group_id = 'E1'",
    ).fetchone()["reputation"]

    assert tuple(subscription) == (expected_status, 7)
    assert tuple(latest_turn) == (1, "agent_rejected")
    if damages_reputation:
        assert current_reputation < initial_reputation
        assert conn.execute(
            """
            SELECT COUNT(*) FROM reputation_history
            WHERE group_id = 'E1' AND change_reason = ?
            """,
            (f"{thread_type}_agent_rejected_churn",),
        ).fetchone()[0] == 1
    else:
        assert current_reputation == pytest.approx(initial_reputation)


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


def test_enterprise_lead_uses_snapshotted_promotion_for_first_period(
    make_initialized_sim,
    make_agent_tools,
    monkeypatch,
):
    """企业新客谈判和首期收费使用获客时快照的优惠，之后自动取消。"""
    benchmark_config = BenchmarkConfig(
        seed=42,
        lead_promotion_by_channel_group={"linkedin": {"E1": 20.0}},
    )
    conn, simulator, config = make_initialized_sim(
        config=benchmark_config,
        seed=42,
    )
    customer_id = _insert_enterprise_customer(conn)
    simulator._create_enterprise_lead(
        customer_id,
        {
            "group_id": "E1",
            "seat_count": 20,
            "_lead_channel": "linkedin",
            "_lead_promo_used": 20.0,
        },
    )
    lead = conn.execute(
        "SELECT promotion FROM subscriptions WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    assert lead["promotion"] == pytest.approx(20.0)

    # 快照完成后即使优惠配置变化，本条线索仍沿用获客时看到的价格。
    simulator.config.lead_promotion_by_channel_group = {}
    monkeypatch.setattr(
        enterprise_module,
        "get_qualities_for_all_plans_batch",
        lambda _conn, customer_ids, _config: {
            customer_id: {"A": 0.5, "B": 0.5, "C": 0.5}
            for customer_id in customer_ids
        },
    )
    assert make_agent_tools(conn, config).send_enterprise_deal(
        [[customer_id, [["B", 100.0]]]],
    ).success

    thread = conn.execute(
        """
        SELECT thread_id, next_reply_day FROM enterprise_turns
        WHERE customer_id = ? ORDER BY message_id DESC LIMIT 1
        """,
        (customer_id,),
    ).fetchone()
    simulator.current_day = thread["next_reply_day"]
    simulator._process_scheduled_replies(simulator.get_current_config(), [])

    first_period = conn.execute(
        """
        SELECT listed_price, promotion, effective_price, first_billing_done
        FROM subscriptions WHERE customer_id = ?
        """,
        (customer_id,),
    ).fetchone()
    assert tuple(first_period) == pytest.approx((100.0, 20.0, 80.0, 0))

    payment = simulator._process_billing(simulator.get_current_config())
    assert payment == pytest.approx(1_600.0)
    next_period = conn.execute(
        """
        SELECT promotion, effective_price, first_billing_done
        FROM subscriptions WHERE customer_id = ?
        """,
        (customer_id,),
    ).fetchone()
    assert tuple(next_period) == pytest.approx((0.0, 100.0, 1))
