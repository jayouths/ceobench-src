"""企业客户成交与收费周期回归测试。"""

import pytest

from saas_bench.simulator.config import (
    BenchmarkConfig,
    ENTERPRISE_RESPONSE_TIMEOUT_DAYS,
)
from saas_bench.simulator.enterprise import (
    close_thread,
    create_negotiation_thread,
    get_negotiation_state,
    get_threads_awaiting_agent_response,
    get_threads_needing_reply,
)
from saas_bench.simulator.database import (
    add_enterprise_turn,
    mark_enterprise_thread_dead,
)


def _insert_enterprise_customer(conn, *, seats: float = 10) -> int:
    cursor = conn.execute(
        """
        INSERT INTO customers (
            customer_type, group_id, created_day,
            steepness_left, steepness_right, c_max, q_max, q_min,
            usage_demand, quality_sensitivity, price_sensitivity,
            willingness_to_pay, usage_scale, patience, seat_count, email
        )
        VALUES (
            'large', 'E1', 2,
            0.01, 0.01, 200.0, 0.9, 0.1,
            10.0, 0.5, 0.5,
            200.0, 10.0, 0.5, ?, 'billing-test@example.com'
        )
        """,
        (seats,),
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


def _insert_enterprise_subscription(
    conn,
    customer_id: int,
    *,
    billing_day: int,
    seats: int = 10,
    price: float = 80.0,
    promotion: float = 0.0,
) -> None:
    conn.execute(
        """
        INSERT INTO subscriptions (
            customer_id, plan, listed_price, promotion, effective_price, seat_count,
            start_day, status, billing_day_mod30, first_billing_done,
            contract_months, contract_end_day
        )
        VALUES (?, 'B', ?, ?, ?, ?, 1, 'subscribed', ?, 1, 1, ?)
        """,
        (
            customer_id,
            price,
            promotion,
            max(0.0, price - promotion),
            seats,
            billing_day % 30,
            billing_day + 30,
        ),
    )


def _disable_unrelated_negotiation_work(monkeypatch, simulator) -> None:
    monkeypatch.setattr(
        simulator,
        "_process_agent_response_timeouts",
        lambda config, events: None,
    )
    monkeypatch.setattr(
        simulator,
        "_process_contract_renewals",
        lambda config, active_customers: None,
    )


def test_new_enterprise_customer_is_billed_on_deal_day(make_initialized_sim):
    """线索出现日不得让成交后的首期收费延迟到旧账单日。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn, seats=10)

    simulator.current_day = 2
    simulator._create_enterprise_lead(customer_id, {"seat_count": 10})
    thread_id = conn.execute(
        "SELECT thread_id FROM enterprise_turns WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()["thread_id"]

    simulator.current_day = 5
    state = get_negotiation_state(conn, thread_id)
    simulator._finalize_deal(
        thread_id,
        state,
        final_price=80.0,
        agent_offer={
            "plan": "B",
            "price_per_seat": 80.0,
            "contract_months": 1,
        },
    )
    simulator._process_billing(simulator.get_current_config())

    subscription = conn.execute(
        """
        SELECT start_day, billing_day_mod30, first_billing_done
        FROM subscriptions WHERE customer_id = ?
        """,
        (customer_id,),
    ).fetchone()
    payment = conn.execute(
        """
        SELECT amount FROM ledger
        WHERE day = 5 AND category = 'subscription_payment'
        """
    ).fetchone()

    assert tuple(subscription) == (5, 5, 1)
    assert payment["amount"] == pytest.approx(800.0)


def test_active_subscription_seats_are_the_billing_authority(make_initialized_sim):
    """隐藏浮点席位尚未同步时，账单仍只按当前生效的订阅席位收费。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn, seats=37.8)
    _insert_enterprise_subscription(
        conn,
        customer_id,
        billing_day=5,
        seats=10,
        price=80.0,
    )

    simulator.current_day = 5
    payments = simulator._process_billing(simulator.get_current_config())

    assert payments == pytest.approx(800.0)


def test_seat_drift_syncs_effective_subscription_seats(make_initialized_sim):
    """浮点漂移结果应立即截断并同步到当前有效订阅。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn, seats=37.8)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5, seats=10)

    simulator._sync_active_subscription_seat_counts()

    subscription_seats = conn.execute(
        "SELECT seat_count FROM subscriptions WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()["seat_count"]
    hidden_seats = conn.execute(
        "SELECT seat_count FROM customers WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()["seat_count"]
    assert subscription_seats == 37
    assert hidden_seats == pytest.approx(37.8)


def test_enterprise_reply_timeout_matches_weekly_decision_cycle(
    make_initialized_sim,
):
    """企业消息等待六天仍可回复，满七天才进入超时处理。"""
    conn, _, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    thread_id, _ = create_negotiation_thread(conn, customer_id, "new_lead", 1)

    assert get_threads_awaiting_agent_response(
        conn,
        1 + ENTERPRISE_RESPONSE_TIMEOUT_DAYS - 1,
    ) == []
    timed_out = get_threads_awaiting_agent_response(
        conn,
        1 + ENTERPRISE_RESPONSE_TIMEOUT_DAYS,
    )
    assert [thread["thread_id"] for thread in timed_out] == [thread_id]


def _create_dead_enterprise_thread(conn) -> int:
    """构造最新 Agent 报价已超时、历史客户消息仍正常的线程。"""
    customer_id = _insert_enterprise_customer(conn)
    thread_id, _ = create_negotiation_thread(
        conn,
        customer_id,
        "new_lead",
        1,
    )
    add_enterprise_turn(
        conn,
        thread_id,
        2,
        "agent",
        next_reply_day=8,
    )
    mark_enterprise_thread_dead(conn, thread_id, "timeout")
    conn.commit()
    return thread_id


def test_dead_enterprise_thread_is_not_revived_by_direct_queries(
    make_initialized_sim,
):
    """后备查询不得在排除终态行后把上一条历史消息当成最新状态。"""
    conn, _, _ = make_initialized_sim(seed=42)
    _create_dead_enterprise_thread(conn)

    assert get_threads_needing_reply(conn, current_day=8) == []
    assert get_threads_awaiting_agent_response(conn, current_day=8) == []


def test_dead_enterprise_thread_is_not_revived_in_daily_cache(
    make_initialized_sim,
    monkeypatch,
):
    """模拟日主链路的最新谈判缓存必须保留线程终态。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    _create_dead_enterprise_thread(conn)
    simulator.current_day = 8
    observed = {}

    monkeypatch.setattr(
        simulator,
        "_process_agent_response_timeouts",
        lambda config, events: observed.update(
            awaiting=get_threads_awaiting_agent_response(conn, current_day=8),
        ),
    )
    monkeypatch.setattr(
        simulator,
        "_process_scheduled_replies",
        lambda config, events: observed.update(
            replies=get_threads_needing_reply(conn, current_day=8),
        ),
    )
    monkeypatch.setattr(
        simulator,
        "_check_negotiation_triggers",
        lambda config, overload, outage, active_customers: set(),
    )
    monkeypatch.setattr(
        simulator,
        "_process_contract_renewals",
        lambda config, active_customers: None,
    )

    simulator._process_enterprise_negotiations(
        simulator.get_current_config(),
        overload=0.0,
        outage=False,
    )

    assert observed == {"awaiting": [], "replies": []}


def test_billing_resumes_when_churn_prevention_closes_today(
    make_initialized_sim,
    monkeypatch,
):
    """收费时已经关闭的流失挽回会话不得继续暂停付款。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5)
    thread_id, _ = create_negotiation_thread(
        conn, customer_id, "churn_prevention", 4
    )

    simulator.current_day = 5
    _disable_unrelated_negotiation_work(monkeypatch, simulator)
    monkeypatch.setattr(
        simulator,
        "_process_scheduled_replies",
        lambda config, events: close_thread(conn, thread_id, "accepted"),
    )
    monkeypatch.setattr(
        simulator,
        "_check_negotiation_triggers",
        lambda config, overload, outage, active_customers: set(),
    )

    config = simulator.get_current_config()
    simulator._process_enterprise_negotiations(
        config,
        overload=0.0,
        outage=False,
    )
    payments = simulator._process_billing(config)

    assert payments == pytest.approx(800.0)


def test_billing_pauses_when_churn_prevention_starts_today(
    make_initialized_sim,
    monkeypatch,
):
    """收费前新建的流失挽回会话必须立即暂停付款。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5)

    simulator.current_day = 5
    _disable_unrelated_negotiation_work(monkeypatch, simulator)
    monkeypatch.setattr(
        simulator,
        "_process_scheduled_replies",
        lambda config, events: None,
    )

    def create_churn_prevention(config, overload, outage, active_customers):
        create_negotiation_thread(
            conn, customer_id, "churn_prevention", simulator.current_day
        )
        return {customer_id}

    monkeypatch.setattr(
        simulator,
        "_check_negotiation_triggers",
        create_churn_prevention,
    )

    config = simulator.get_current_config()
    simulator._process_enterprise_negotiations(
        config,
        overload=0.0,
        outage=False,
    )
    payments = simulator._process_billing(config)

    assert payments == 0.0


def test_outage_can_trigger_enterprise_churn_prevention(
    make_initialized_sim,
    monkeypatch,
):
    """企业客户的去留判断必须感知当日故障。"""
    benchmark_config = BenchmarkConfig(seed=42, base_product_quality=0.6)
    conn, simulator, _ = make_initialized_sim(
        config=benchmark_config,
        seed=42,
    )
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5)
    simulator.current_day = 5
    config = simulator.get_current_config()
    config.update({"tier_A": 4, "tier_B": 4, "tier_C": 4})
    simulator._cache_step_day_globals(config)

    # 把参与约束固定在 0.5，单独验证故障惩罚是否进入企业链路。
    monkeypatch.setattr(
        simulator,
        "_compute_satisfaction",
        lambda _left, _right, _c_max, quality, _price, _q_max, _q_min: (
            quality - 0.5
        ),
    )
    monkeypatch.setattr(
        simulator,
        "_check_plan_change_opportunity",
        lambda *args, **kwargs: None,
    )

    without_outage = simulator._check_negotiation_triggers(
        config,
        overload=0.0,
        outage=False,
        active_thread_customers=set(),
    )
    with_outage = simulator._check_negotiation_triggers(
        config,
        overload=0.0,
        outage=True,
        active_thread_customers=set(),
    )

    assert without_outage == set()
    assert with_outage == {customer_id}
    thread = conn.execute(
        """
        SELECT thread_type FROM enterprise_turns
        WHERE customer_id = ?
        ORDER BY message_id DESC LIMIT 1
        """,
        (customer_id,),
    ).fetchone()
    assert thread["thread_type"] == "churn_prevention"


def test_customer_targeted_ads_reduce_enterprise_churn_quality(
    make_initialized_sim,
    monkeypatch,
):
    """企业流失判断必须计入只对该客户生效的广告副作用。"""
    benchmark_config = BenchmarkConfig(seed=42, base_product_quality=0.6)
    conn, simulator, _ = make_initialized_sim(
        config=benchmark_config,
        seed=42,
    )
    customer_id = _insert_enterprise_customer(conn)
    conn.execute(
        "UPDATE customers SET ads_quality_sensitivity = 0.2 WHERE customer_id = ?",
        (customer_id,),
    )
    _insert_enterprise_subscription(conn, customer_id, billing_day=5)
    simulator.current_day = 5
    config = simulator.get_current_config()
    config.update({"tier_A": 4, "tier_B": 4, "tier_C": 4})
    simulator._cache_step_day_globals(config)
    evaluated_qualities = []

    def record_quality(_left, _right, _c_max, quality, *_args):
        evaluated_qualities.append(quality)
        return 1.0

    monkeypatch.setattr(simulator, "_compute_satisfaction", record_quality)
    monkeypatch.setattr(
        simulator,
        "_check_plan_change_opportunity",
        lambda *args, **kwargs: None,
    )

    simulator._check_negotiation_triggers(
        config,
        overload=0.0,
        outage=False,
        active_thread_customers=set(),
    )
    simulator.config.ads_strength_by_customer[customer_id] = 1.0
    simulator._check_negotiation_triggers(
        config,
        overload=0.0,
        outage=False,
        active_thread_customers=set(),
    )

    assert evaluated_qualities[1] == pytest.approx(evaluated_qualities[0] - 0.2)


def test_enterprise_churn_uses_effective_subscription_price(
    make_initialized_sim,
    monkeypatch,
):
    """企业流失判断应使用当前实际支付价格，而不是促销前原价。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(
        conn,
        customer_id,
        billing_day=5,
        price=100.0,
        promotion=20.0,
    )
    simulator.current_day = 5
    config = simulator.get_current_config()
    simulator._cache_step_day_globals(config)
    evaluated_prices = []

    def record_price(_left, _right, _c_max, _quality, price, _q_max, _q_min):
        evaluated_prices.append(price)
        return 1.0

    monkeypatch.setattr(simulator, "_compute_satisfaction", record_price)
    monkeypatch.setattr(
        simulator,
        "_check_plan_change_opportunity",
        lambda *args, **kwargs: None,
    )

    simulator._check_negotiation_triggers(
        config,
        overload=0.0,
        outage=False,
        active_thread_customers=set(),
    )

    assert evaluated_prices == [80.0]
