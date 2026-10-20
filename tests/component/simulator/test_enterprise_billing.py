"""企业客户成交与收费周期回归测试。"""

from copy import deepcopy
from unittest.mock import Mock

from numpy.random import default_rng
import pytest

from saas_bench.simulator.config import (
    BenchmarkConfig,
    ENTERPRISE_RESPONSE_TIMEOUT_DAYS,
    INDIVIDUAL_PREFERENCE_DRIFT,
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
    create_issue,
    init_database,
    mark_enterprise_thread_dead,
)
from saas_bench.simulator.simulation import Simulator


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


@pytest.mark.parametrize(
    ("day", "first_billing_done", "contract_months", "should_churn"),
    [
        (5, 0, 1, False),
        (34, 1, 1, False),
        (35, 1, 1, True),
        (36, 1, 1, False),
        (65, 1, 1, True),
        (35, 0, 1, False),
        (35, 1, 6, False),
    ],
)
def test_monthly_enterprise_churn_only_runs_at_paid_renewals(
    make_initialized_sim, monkeypatch,
    day, first_billing_done, contract_months, should_churn,
):
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5)
    conn.execute("""
        UPDATE subscriptions SET start_day = 5, contract_months = ?, first_billing_done = ?
        WHERE customer_id = ?
    """, (contract_months, first_billing_done, customer_id))
    simulator.current_day = day
    probability = Mock(return_value=1.0)
    monkeypatch.setattr(simulator, "_get_involuntary_churn_mu", probability)

    simulator._process_monthly_enterprise_renewals()

    subscription = conn.execute(
        "SELECT status, end_day, churn_reason FROM subscriptions WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    expected = ("cancelled", day, "involuntary") if should_churn else ("subscribed", None, None)
    assert tuple(subscription) == expected
    assert probability.call_count == int(should_churn)
    if should_churn:
        probability.assert_called_once_with("E1")


def test_monthly_enterprise_survives_renewal_and_pays_automatically(
    make_initialized_sim, monkeypatch,
):
    """没抽中自然流失就继续按原合同收费，不新建续约谈判。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5)
    conn.execute("UPDATE subscriptions SET start_day = 5 WHERE customer_id = ?", (customer_id,))
    probability = Mock(return_value=0.5)
    monkeypatch.setattr(simulator, "_get_involuntary_churn_mu", probability)
    rng = Mock(wraps=simulator.rng)
    rng.random.return_value = 0.75
    monkeypatch.setattr(simulator, "rng", rng)

    for day in (35, 65):
        simulator.current_day = day
        simulator._process_monthly_enterprise_renewals()
        assert simulator._process_billing(simulator.get_current_config()) == pytest.approx(800.0)

    assert probability.call_count == rng.random.call_count == 2
    assert conn.execute("SELECT COUNT(*) FROM enterprise_turns").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM subscriptions WHERE status = 'subscribed'").fetchone()[0] == 1


def test_disabled_natural_churn_does_not_draw_or_cancel_monthly_enterprise(
    make_initialized_sim, monkeypatch,
):
    config = BenchmarkConfig(seed=42, enable_involuntary_churn=False)
    conn, simulator, _ = make_initialized_sim(seed=42, config=config)
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5)
    conn.execute("UPDATE subscriptions SET start_day = 5 WHERE customer_id = ?", (customer_id,))
    simulator.current_day = 35
    rng = Mock(wraps=simulator.rng)
    monkeypatch.setattr(simulator, "rng", rng)

    simulator._process_monthly_enterprise_renewals()

    rng.random.assert_not_called()
    assert conn.execute("SELECT status FROM subscriptions WHERE customer_id = ?", (customer_id,)).fetchone()[0] == "subscribed"


def test_monthly_natural_churn_stops_pending_replies_and_billing(
    make_initialized_sim,
    monkeypatch,
):
    """续费流失先于当日谈判，取消及席位只采集一次，不附加声誉惩罚。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5)
    conn.execute("UPDATE subscriptions SET start_day = 5 WHERE customer_id = ?", (customer_id,))
    thread_id, original_message_id = create_negotiation_thread(
        conn, customer_id, "renegotiation", 34,
    )
    reply_id = add_enterprise_turn(
        conn, thread_id, 34, "agent", next_reply_day=35,
        offer_json='[{"plan":"B","price":80}]',
    )
    issue_id = create_issue(conn, customer_id, "E1", open_day=33)
    reputation = [tuple(row) for row in conn.execute("SELECT * FROM group_reputation")]
    simulator.current_day = 35
    monkeypatch.setattr(simulator, "_get_involuntary_churn_mu", lambda group: 1.0)

    events = simulator._process_enterprise_negotiations(
        simulator.get_current_config(), overload=0.0, outage=False,
    )
    # 再次处理已取消客户不能再写一次取消，也不能再尝试回复。
    simulator._process_monthly_enterprise_renewals()
    payments = simulator._process_billing(simulator.get_current_config())

    assert events == []
    assert payments == 0.0
    assert conn.execute("SELECT COUNT(*) FROM ledger WHERE category = 'subscription_payment'").fetchone()[0] == 0
    assert get_threads_needing_reply(conn, current_day=35) == []
    assert get_threads_awaiting_agent_response(conn, current_day=35) == []
    assert tuple(conn.execute(
        "SELECT _internal_status, next_reply_day FROM enterprise_turns WHERE message_id = ?",
        (reply_id,),
    ).fetchone()) == ("customer_churned", None)
    assert conn.execute("SELECT _internal_status FROM enterprise_turns WHERE message_id = ?", (original_message_id,)).fetchone()[0] is None
    assert [tuple(row) for row in conn.execute("SELECT * FROM group_reputation")] == reputation
    assert conn.execute("SELECT COUNT(*) FROM reputation_history").fetchone()[0] == 0
    assert tuple(conn.execute(
        "SELECT status, resolution_type FROM issues WHERE issue_id = ?", (issue_id,),
    ).fetchone()) == ("resolved", "customer_churned")
    assert tuple(conn.execute("""
        SELECT day, seats FROM _subscription_activity_event
        WHERE event_type = 'cancelled'
    """).fetchone()) == (35, 10)
    assert conn.execute("SELECT COUNT(*) FROM _subscription_activity_event WHERE event_type = 'cancelled'").fetchone()[0] == 1


def test_monthly_enterprise_renewal_matches_after_checkpoint_restore(
    make_initialized_sim, tmp_path, monkeypatch,
):
    """从续费前一天恢复，真实日推进的取消、采集和随机流必须与连续运行一致。"""
    conn, simulator, config = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5)
    conn.execute("UPDATE subscriptions SET start_day = 5 WHERE customer_id = ?", (customer_id,))
    simulator.current_day = 34
    simulator.save_rng_states()
    conn.commit()

    restored_conn = init_database(tmp_path / "restored-renewal.db")
    try:
        conn.backup(restored_conn)
        restored = Simulator(restored_conn, deepcopy(config), default_rng(42))
        restored.initialize(resume=True)
        restored.current_day = 34
        assert restored.restore_rng_states()
        for sim in (simulator, restored):
            monkeypatch.setattr(sim, "_get_involuntary_churn_mu", lambda group: 1.0)

        continuous_result = simulator.step_day()
        restored_result = restored.step_day()

        assert continuous_result == restored_result
        assert continuous_result.enterprise_cancellations == 1
        assert continuous_result.enterprise_cancelled_seats == 10
        assert conn.execute("SELECT COUNT(*) FROM ledger WHERE day = 35 AND category = 'subscription_payment'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM _eval_subscription_day WHERE day = 35 AND customer_type = 'large'").fetchone()[0] == 0
        assert list(conn.iterdump()) == list(restored_conn.iterdump())
    finally:
        restored_conn.close()


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
    assert conn.execute(
        "SELECT effective_c_max FROM subscriptions WHERE customer_id = ?", (customer_id,),
    ).fetchone()[0] == pytest.approx(200.0)
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


def test_billing_uses_internal_seats_and_refreshes_public_snapshot(make_initialized_sim):
    """沿用作者：按内部整数席位收费，并在账单事件更新公开快照。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn, seats=37.8)
    _insert_enterprise_subscription(
        conn,
        customer_id,
        billing_day=5,
        seats=10,
        price=80.0,
    )

    simulator.current_day = 4
    settings = simulator.get_current_config()
    simulator._cache_step_day_globals(settings)
    events = simulator._update_customer_satisfaction(settings, overload=0.0, outage=False)
    assert events[customer_id]["seat_count"] == 37
    thread_id, _ = create_negotiation_thread(conn, customer_id, "renegotiation", 4)
    assert get_negotiation_state(conn, thread_id).seat_count == 37
    assert conn.execute("SELECT seat_count FROM subscriptions WHERE customer_id = ?", (customer_id,)).fetchone()[0] == 10

    simulator.current_day = 5
    payments = simulator._process_billing(simulator.get_current_config())

    assert payments == pytest.approx(80.0 * 37)
    assert conn.execute("SELECT seat_count FROM subscriptions WHERE customer_id = ?", (customer_id,)).fetchone()[0] == 37


@pytest.mark.parametrize("group_id", ["E1", "E2", "D_E04", "D_E09"])
@pytest.mark.parametrize("pmi", [40.0, 50.0, 60.0])
def test_seat_drift_macro_direction_preserves_public_snapshot(
    make_initialized_sim, group_id, pmi,
):
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn, seats=1000)
    conn.execute(
        "UPDATE customers SET group_id = ? WHERE customer_id = ?",
        (group_id, customer_id),
    )
    _insert_enterprise_subscription(conn, customer_id, billing_day=5, seats=1000)
    simulator.current_day = 90
    simulator._macro_multipliers = simulator._compute_macro_multipliers(pmi)
    rate = (1 + INDIVIDUAL_PREFERENCE_DRIFT[group_id]["seat_count_drift"]) ** 30 - 1
    multiplier = simulator.get_macro_multiplier(group_id, "seat_count")
    expected_rate = rate / multiplier if rate < 0 else rate * multiplier

    simulator._apply_preference_drift(days=30)

    seats = conn.execute(
        "SELECT seat_count FROM customers WHERE customer_id = ?", (customer_id,),
    ).fetchone()[0]
    subscription_seats = conn.execute(
        "SELECT seat_count FROM subscriptions WHERE customer_id = ?", (customer_id,),
    ).fetchone()[0]
    assert seats == pytest.approx(1000 * (1 + expected_rate))
    assert subscription_seats == 1000
    neutral_seats = 1000 * (1 + rate)
    if pmi < 50:
        assert seats < neutral_seats
    elif pmi > 50:
        assert seats > neutral_seats
    else:
        assert seats == pytest.approx(neutral_seats)


def test_monthly_seat_drift_updates_snapshot_only_at_billing(make_initialized_sim):
    """月度漂移立即影响内部席位，公开订阅席位等到自己的账单日刷新。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn, seats=1000)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5, seats=1000)
    simulator.current_day = 90
    simulator._apply_preference_drift(days=30)

    subscription_seats = conn.execute(
        "SELECT seat_count FROM subscriptions WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()["seat_count"]
    hidden_seats = conn.execute(
        "SELECT seat_count FROM customers WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()["seat_count"]
    assert subscription_seats == 1000
    assert int(hidden_seats) != subscription_seats
    simulator.current_day = 95
    simulator._process_billing(simulator.get_current_config())
    assert conn.execute("SELECT seat_count FROM subscriptions WHERE customer_id = ?", (customer_id,)).fetchone()[0] == int(hidden_seats)


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


@pytest.mark.parametrize("pmi,contract_months,expected", [
    (40.0, 1, True), (50.0, 1, False), (60.0, 1, False), (40.0, 6, False),
])
def test_macro_budget_preserves_contract_window_and_stored_prices(
    make_initialized_sim, monkeypatch, pmi, contract_months, expected,
):
    conn, simulator, _ = make_initialized_sim(seed=42)
    customer_id = _insert_enterprise_customer(conn)
    _insert_enterprise_subscription(conn, customer_id, billing_day=5, price=170.0)
    conn.execute("UPDATE subscriptions SET contract_months = ?, contract_end_day = 180 WHERE customer_id = ?", (contract_months, customer_id))
    simulator.current_day = 5
    simulator._macro_multipliers = simulator._compute_macro_multipliers(pmi)
    config = simulator.get_current_config()
    simulator._cache_step_day_globals(config)
    # 即使质量足够，超出预算仍须协商；合同保护期内则不提前触发。
    monkeypatch.setattr(simulator, "_compute_comprehensive_quality_inline", lambda *args: 1.2)
    plan_change = Mock()
    monkeypatch.setattr(simulator, "_check_plan_change_opportunity", plan_change)

    suspended = simulator._check_negotiation_triggers(config, 0.0, False, set())

    assert suspended == ({customer_id} if expected else set())
    if contract_months == 1 and not expected:
        assert plan_change.call_args.args[5] == pytest.approx(200 * (1 + 1.2 * (pmi - 50) / 50))
    row = conn.execute("SELECT listed_price, effective_price, status, end_day FROM subscriptions WHERE customer_id = ?", (customer_id,)).fetchone()
    assert tuple(row) == (170.0, 170.0, "subscribed", None)


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
