"""实验私有经营事实的采集口径回归测试。"""

import pytest

from saas_bench.evaluation.fact_recorder import record_subscription_day


def _insert_customer(conn, customer_type: str, group_id: str, seats: int) -> int:
    cursor = conn.execute(
        """
        INSERT INTO customers (
            customer_type, group_id, created_day,
            steepness_left, steepness_right, c_max, usage_demand,
            quality_sensitivity, price_sensitivity, willingness_to_pay,
            usage_scale, patience, seat_count
        )
        VALUES (?, ?, 1, 0.01, 0.01, 100.0, 10.0,
                0.5, 0.5, 100.0, 1.0, 0.5, ?)
        """,
        (customer_type, group_id, seats),
    )
    return cursor.lastrowid


def _insert_subscription(
    conn,
    customer_id: int,
    *,
    plan: str,
    price: float,
    start_day: int,
    status: str = "subscribed",
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO subscriptions (
            customer_id, plan, listed_price, effective_price,
            start_day, status, billing_day_mod30
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (customer_id, plan, price, price, start_day, status, start_day % 30),
    )
    return cursor.lastrowid


def _mark_simulation_day(conn, day: int) -> None:
    conn.execute(
        """
        INSERT INTO service_day (
            day, total_usage_units, p95_ms, error_rate,
            downtime_minutes, capacity_tier, capacity_units
        )
        VALUES (?, 0, 0.0, 0.0, 0, 0, 0)
        """,
        (day,),
    )


def test_subscription_events_capture_real_status_transition_day(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    small = _insert_customer(conn, "small", "S1", 1)
    enterprise = _insert_customer(conn, "large", "E1", 10)
    enterprise_lead = _insert_customer(conn, "large", "E1", 6)

    _mark_simulation_day(conn, 7)
    small_sub = _insert_subscription(
        conn, small, plan="A", price=20.0, start_day=7
    )
    enterprise_sub = _insert_subscription(
        conn, enterprise, plan="C", price=100.0, start_day=7
    )
    lead_sub = _insert_subscription(
        conn,
        enterprise_lead,
        plan="pending",
        price=0.0,
        start_day=7,
        status="lead",
    )

    _mark_simulation_day(conn, 8)
    conn.execute(
        """
        UPDATE subscriptions
        SET status = 'subscribed', plan = 'B', listed_price = 80.0,
            effective_price = 80.0, start_day = 8
        WHERE subscription_id = ?
        """,
        (lead_sub,),
    )
    # end_day 可以指向未来合同日期；事件日必须仍是实际状态切换的第 8 天。
    conn.execute(
        """
        UPDATE subscriptions
        SET status = 'cancelled', end_day = 90, churn_reason = 'involuntary'
        WHERE subscription_id = ?
        """,
        (enterprise_sub,),
    )

    events = [
        dict(row)
        for row in conn.execute(
            """
            SELECT day, subscription_id, customer_type, group_id,
                   event_type, plan, seats, mrr, reason
            FROM _eval_subscription_event
            ORDER BY event_id
            """
        ).fetchall()
    ]

    assert events == [
        {
            "day": 7,
            "subscription_id": small_sub,
            "customer_type": "small",
            "group_id": "S1",
            "event_type": "started",
            "plan": "A",
            "seats": 1,
            "mrr": pytest.approx(20.0),
            "reason": None,
        },
        {
            "day": 7,
            "subscription_id": enterprise_sub,
            "customer_type": "large",
            "group_id": "E1",
            "event_type": "started",
            "plan": "C",
            "seats": 10,
            "mrr": pytest.approx(1000.0),
            "reason": None,
        },
        {
            "day": 8,
            "subscription_id": lead_sub,
            "customer_type": "large",
            "group_id": "E1",
            "event_type": "started",
            "plan": "B",
            "seats": 6,
            "mrr": pytest.approx(480.0),
            "reason": None,
        },
        {
            "day": 8,
            "subscription_id": enterprise_sub,
            "customer_type": "large",
            "group_id": "E1",
            "event_type": "ended",
            "plan": "C",
            "seats": 10,
            "mrr": pytest.approx(1000.0),
            "reason": "involuntary",
        },
    ]


def test_subscription_day_records_immutable_daily_stock(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    small = _insert_customer(conn, "small", "S1", 1)
    enterprise = _insert_customer(conn, "large", "E1", 10)
    _insert_subscription(conn, small, plan="A", price=20.0, start_day=7)
    enterprise_sub = _insert_subscription(
        conn, enterprise, plan="C", price=100.0, start_day=7
    )

    record_subscription_day(conn, 7)
    conn.execute(
        "UPDATE customers SET seat_count = 12 WHERE customer_id = ?",
        (enterprise,),
    )
    conn.execute(
        """
        UPDATE subscriptions SET plan = 'B', listed_price = 90.0,
                                 effective_price = 90.0
        WHERE subscription_id = ?
        """,
        (enterprise_sub,),
    )
    record_subscription_day(conn, 8)

    day_7 = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM _eval_subscription_day WHERE day = 7 ORDER BY customer_type"
        ).fetchall()
    ]
    day_8 = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM _eval_subscription_day WHERE day = 8 ORDER BY customer_type"
        ).fetchall()
    ]

    assert day_7 == [
        {
            "day": 7,
            "customer_type": "large",
            "group_id": "E1",
            "plan": "C",
            "active_accounts": 1,
            "active_seats": 10,
            "mrr": pytest.approx(1000.0),
        },
        {
            "day": 7,
            "customer_type": "small",
            "group_id": "S1",
            "plan": "A",
            "active_accounts": 1,
            "active_seats": 1,
            "mrr": pytest.approx(20.0),
        },
    ]
    assert day_8[0] == {
        "day": 8,
        "customer_type": "large",
        "group_id": "E1",
        "plan": "B",
        "active_accounts": 1,
        "active_seats": 12,
        "mrr": pytest.approx(1080.0),
    }
    assert day_8[1]["mrr"] == pytest.approx(20.0)


def test_step_day_records_subscription_day(make_initialized_sim):
    conn, simulator, _ = make_initialized_sim()
    customer_id = _insert_customer(conn, "small", "TEST", 1)
    _insert_subscription(conn, customer_id, plan="A", price=20.0, start_day=0)

    simulator.step_day()

    row = conn.execute(
        """
        SELECT active_accounts, active_seats, mrr
        FROM _eval_subscription_day
        WHERE day = 1 AND customer_type = 'small'
          AND group_id = 'TEST' AND plan = 'A'
        """
    ).fetchone()
    assert tuple(row) == pytest.approx((1, 1, 20.0))
