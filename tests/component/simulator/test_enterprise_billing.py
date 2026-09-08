"""企业客户成交与收费周期回归测试。"""

import pytest

from saas_bench.simulator.enterprise import (
    close_thread,
    create_negotiation_thread,
    get_negotiation_state,
)


def _insert_enterprise_customer(conn, *, seats: int = 10) -> int:
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
) -> None:
    conn.execute(
        """
        INSERT INTO subscriptions (
            customer_id, plan, listed_price, effective_price, seat_count,
            start_day, status, billing_day_mod30, first_billing_done,
            contract_months, contract_end_day
        )
        VALUES (?, 'B', ?, ?, ?, 1, 'subscribed', ?, 1, 1, ?)
        """,
        (customer_id, price, price, seats, billing_day % 30, billing_day + 30),
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
        lambda config, active_customers: set(),
    )

    config = simulator.get_current_config()
    simulator._process_enterprise_negotiations(config)
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

    def create_churn_prevention(config, active_customers):
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
    simulator._process_enterprise_negotiations(config)
    payments = simulator._process_billing(config)

    assert payments == 0.0
