"""企业客户成交与收费周期回归测试。"""

import pytest

from saas_bench.simulator.enterprise import get_negotiation_state


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
