"""企业报价工具契约回归测试。"""

import copy
import json
import math

import pytest

import saas_bench.simulator.enterprise as enterprise_module
from saas_bench.simulator.config import BenchmarkConfig
from saas_bench.simulator.enterprise import (
    batch_schedule_customer_replies,
    create_negotiation_thread,
    get_negotiation_state,
    get_negotiation_states_batch,
    get_quality_for_plan,
    get_qualities_for_all_plans_batch,
)
from saas_bench.simulator.environment import get_thread_inbox_items


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
    customer_id = _create_enterprise_subscriber(conn)
    thread_id, _ = create_negotiation_thread(
        conn,
        customer_id,
        "renegotiation",
        0,
    )
    conn.commit()
    return customer_id, thread_id


def _create_enterprise_subscriber(conn) -> int:
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
    conn.commit()
    return customer_id


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


def test_enterprise_inbox_uses_new_lead_seat_snapshot(make_initialized_sim):
    """新线索应只计一次，并保留创建谈判时的席位规模。"""

    conn, _, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    create_negotiation_thread(conn, customer_id, "new_lead", current_day=3)
    conn.execute(
        "UPDATE customers SET seat_count = 75 WHERE customer_id = ?",
        (customer_id,),
    )

    assert get_thread_inbox_items(conn, day=7, week_start_day=1) == [
        "📨 1 new enterprise leads this week (20 total seats)"
    ]


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

    qualities = get_qualities_for_all_plans_batch(
        conn, [customer_id], config, current_day=0
    )

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

    qualities = get_qualities_for_all_plans_batch(
        conn, [customer_id], config, current_day=0
    )

    # Day 0 的每周倍率为 1；真实需求 50 应得到 0、0.5、1 三档配额因子。
    assert qualities[customer_id] == pytest.approx(
        {"A": 0.0, "B": 0.1, "C": 0.2}
    )


def test_enterprise_plan_qualities_include_current_tenure(make_initialized_sim):
    """企业报价评估应按当前日期计入客户黏性加成。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, _ = _create_active_enterprise_thread(conn)
    conn.execute(
        """
        UPDATE config_history
        SET tier_A = 4, quota_A = 100000
        WHERE day = 0
        """
    )

    qualities = get_qualities_for_all_plans_batch(
        conn, [customer_id], config, current_day=90
    )

    expected_stickiness = config.stickiness_log_scale * math.log(4)
    assert qualities[customer_id]["A"] == pytest.approx(
        config.base_product_quality + expected_stickiness
    )


@pytest.mark.parametrize("raw_quality, expected", [(2.5, 2.5), (-3.0, -1.0)])
def test_enterprise_quality_has_no_upper_cap_but_keeps_lower_floor(
    make_initialized_sim, raw_quality, expected,
):
    conn, _, config = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    config.base_product_quality = raw_quality
    conn.execute(
        "UPDATE config_history SET tier_A = 4, quota_A = 100000 WHERE day = 0"
    )

    assert get_quality_for_plan(conn, "A", customer_id, config) == pytest.approx(
        expected
    )
    qualities = get_qualities_for_all_plans_batch(
        conn, [customer_id], config, current_day=0
    )
    assert qualities[customer_id]["A"] == pytest.approx(expected)


def test_zero_relationship_remains_worst_in_enterprise_paths(
    make_initialized_sim,
):
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, thread_id = _create_active_enterprise_thread(conn)
    conn.execute(
        "UPDATE customer_state SET relationship = 0.0 WHERE customer_id = ?",
        (customer_id,),
    )
    conn.execute(
        """
        UPDATE config_history
        SET tier_A = 4, tier_B = 4, tier_C = 4,
            quota_A = 100000, quota_B = 100000, quota_C = 100000
        WHERE day = 0
        """
    )

    single_state = get_negotiation_state(conn, thread_id)
    batch_state = get_negotiation_states_batch(conn, [thread_id])[thread_id]
    scalar_quality = get_quality_for_plan(conn, "A", customer_id, config)
    batch_qualities = get_qualities_for_all_plans_batch(
        conn,
        [customer_id],
        config,
        current_day=0,
    )

    expected_quality = (
        config.base_product_quality - config.relationship_quality_bonus_max
    )
    assert single_state.relationship == pytest.approx(0.0)
    assert batch_state.relationship == pytest.approx(0.0)
    assert scalar_quality == pytest.approx(expected_quality)
    assert batch_qualities[customer_id] == pytest.approx(
        {"A": expected_quality, "B": expected_quality, "C": expected_quality}
    )


def test_enterprise_plan_qualities_use_current_weekday_multiplier(
    make_initialized_sim,
):
    """企业报价评估应使用当天的每周用量倍率。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id, _ = _create_active_enterprise_thread(conn)
    conn.execute(
        """
        UPDATE subscriptions
        SET start_day = 2, daily_usage_rate = 100
        WHERE customer_id = ?
        """,
        (customer_id,),
    )
    conn.execute(
        """
        UPDATE config_history
        SET tier_A = 4, quota_A = 100
        WHERE day = 0
        """
    )

    qualities = get_qualities_for_all_plans_batch(
        conn, [customer_id], config, current_day=2
    )

    # Day 2 的用量倍率为 1.15，100 配额只能覆盖 100/115 的需求。
    assert qualities[customer_id]["A"] == pytest.approx(
        config.base_product_quality * (100 / 115)
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


def test_enterprise_reply_schedule_is_independent_of_batch_order(
    make_initialized_sim,
):
    """每个谈判事件的回复日只取决于事件身份，不取决于批量顺序。"""
    conn, _, _ = make_initialized_sim(seed=42)
    _, first_thread_id = _create_active_enterprise_thread(conn)
    _, second_thread_id = _create_active_enterprise_thread(conn)

    batch_schedule_customer_replies(
        conn,
        [first_thread_id, second_thread_id],
        current_day=7,
        seed=42,
    )
    first_order = {
        row["thread_id"]: row["next_reply_day"]
        for row in conn.execute(
            """
            SELECT thread_id, next_reply_day
            FROM enterprise_turns
            WHERE thread_id IN (?, ?)
            """,
            (first_thread_id, second_thread_id),
        )
    }

    conn.execute(
        "UPDATE enterprise_turns SET next_reply_day = NULL WHERE thread_id IN (?, ?)",
        (first_thread_id, second_thread_id),
    )
    batch_schedule_customer_replies(
        conn,
        [second_thread_id, first_thread_id],
        current_day=7,
        seed=42,
    )
    reversed_order = {
        row["thread_id"]: row["next_reply_day"]
        for row in conn.execute(
            """
            SELECT thread_id, next_reply_day
            FROM enterprise_turns
            WHERE thread_id IN (?, ?)
            """,
            (first_thread_id, second_thread_id),
        )
    }

    assert reversed_order == first_order


def test_sending_enterprise_deal_does_not_advance_simulator_rng(
    make_initialized_sim,
    make_agent_tools,
):
    """Agent 发送报价不得改变模拟器后续外生事件的随机序列。"""
    conn, simulator, config = make_initialized_sim(seed=42)
    customer_id, _ = _create_active_enterprise_thread(conn)
    rng_state_before = copy.deepcopy(simulator.rng.bit_generator.state)

    result = make_agent_tools(conn, config, day=7, seed=42).send_enterprise_deal(
        [[customer_id, [["B", 50.0]]]],
    )

    assert result.success
    assert simulator.rng.bit_generator.state == rng_state_before


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


def test_enterprise_deal_rejects_duplicate_customer_before_creating_threads(
    make_initialized_sim,
    make_agent_tools,
):
    """同批重复客户必须整批失败，不能创建两条互相冲突的谈判。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id = _create_enterprise_subscriber(conn)

    result = make_agent_tools(conn, config).send_enterprise_deal([
        [customer_id, [["B", 50.0]]],
        [customer_id, [["B", 45.0]]],
    ])

    assert not result.success
    assert "duplicate customer_id" in result.message
    assert result.data["results"] == [{
        "customer_id": customer_id,
        "error": "duplicate customer_id",
    }]
    assert conn.execute(
        "SELECT COUNT(*) FROM enterprise_turns WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()[0] == 0

    malformed_duplicate = make_agent_tools(conn, config).send_enterprise_deal([
        [customer_id, [["B", 50.0]]],
        [customer_id, []],
    ])
    assert not malformed_duplicate.success
    assert "duplicate customer_id" in malformed_duplicate.message
    assert conn.execute(
        "SELECT COUNT(*) FROM enterprise_turns WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()[0] == 0


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


def test_rejecting_enterprise_deal_does_not_advance_simulator_rng(
    make_initialized_sim,
    make_agent_tools,
):
    """Agent 主动拒绝造成的随机声誉损伤不得消耗主随机流。"""
    conn, simulator, config = make_initialized_sim(seed=42)
    customer_id, _ = _create_rejectable_enterprise_thread(
        conn,
        "churn_prevention",
    )
    rng_state_before = copy.deepcopy(simulator.rng.bit_generator.state)

    result = make_agent_tools(conn, config, day=7, seed=42).reject_enterprise_deal(
        [{"customer_id": customer_id}],
    )

    assert result.success
    assert simulator.rng.bit_generator.state == rng_state_before


def test_existing_enterprise_evaluates_offer_at_promoted_price(
    make_initialized_sim,
    make_agent_tools,
    monkeypatch,
):
    """存量企业按促销后净价接受报价，成交仍保留促销前报价。"""
    benchmark_config = BenchmarkConfig(seed=42, promotion_global=60.0)
    conn, simulator, config = make_initialized_sim(
        config=benchmark_config,
        seed=42,
    )
    customer_id, thread_id = _create_active_enterprise_thread(conn)
    monkeypatch.setattr(
        enterprise_module,
        "get_qualities_for_all_plans_batch",
        lambda _conn, customer_ids, _config, *, current_day: {
            customer_id: {"A": 1.0, "B": 1.0, "C": 1.0}
            for customer_id in customer_ids
        },
    )

    result = make_agent_tools(conn, config).send_enterprise_deal(
        [[customer_id, [["B", 250.0]]]],
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
    # 报价高于 200 的预算，但促销后实际支付 190，仍应允许客户接受。
    assert tuple(subscription) == pytest.approx((250.0, 60.0, 190.0))


def test_enterprise_rejects_over_budget_offer_even_with_extreme_quality(
    make_initialized_sim,
    make_agent_tools,
    monkeypatch,
):
    """质量任意高都不能突破企业客户的每席预算硬上限。"""
    conn, simulator, config = make_initialized_sim(seed=42)
    customer_id, thread_id = _create_active_enterprise_thread(conn)
    monkeypatch.setattr(
        enterprise_module,
        "get_qualities_for_all_plans_batch",
        lambda _conn, customer_ids, _config, *, current_day: {
            customer_id: {"A": 100.0, "B": 100.0, "C": 100.0}
            for customer_id in customer_ids
        },
    )

    assert make_agent_tools(conn, config).send_enterprise_deal(
        [[customer_id, [["B", 500.0]]]],
    ).success
    reply_day = conn.execute(
        """
        SELECT next_reply_day FROM enterprise_turns
        WHERE thread_id = ? ORDER BY message_id DESC LIMIT 1
        """,
        (thread_id,),
    ).fetchone()["next_reply_day"]
    simulator.current_day = reply_day
    simulator._process_scheduled_replies(simulator.get_current_config(), [])

    reply = conn.execute(
        """
        SELECT message_text, current_offer_price FROM enterprise_turns
        WHERE thread_id = ? AND sender = 'customer'
        ORDER BY message_id DESC LIMIT 1
        """,
        (thread_id,),
    ).fetchone()
    subscription = conn.execute(
        "SELECT listed_price FROM subscriptions WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    assert reply["message_text"].startswith("Counter-offer:")
    assert reply["current_offer_price"] <= 200.0
    assert subscription["listed_price"] == pytest.approx(50.0)


def test_enterprise_prefers_affordable_offer_over_higher_quality_over_budget_offer(
    make_initialized_sim,
    make_agent_tools,
    monkeypatch,
):
    """多方案择优时必须排除超预算方案，而不是让质量优势覆盖预算。"""
    conn, simulator, config = make_initialized_sim(seed=42)
    customer_id, thread_id = _create_active_enterprise_thread(conn)
    monkeypatch.setattr(
        enterprise_module,
        "get_qualities_for_all_plans_batch",
        lambda _conn, customer_ids, _config, *, current_day: {
            customer_id: {"A": 100.0, "B": 1.0, "C": 0.0}
            for customer_id in customer_ids
        },
    )

    assert make_agent_tools(conn, config).send_enterprise_deal(
        [[customer_id, [["A", 500.0], ["B", 100.0]]]],
    ).success
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
        "SELECT plan, listed_price FROM subscriptions WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    assert subscription["plan"] == "B"
    assert subscription["listed_price"] == pytest.approx(100.0)


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
        lambda _conn, customer_ids, _config, *, current_day: {
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
