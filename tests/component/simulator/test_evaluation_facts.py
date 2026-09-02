"""实验私有经营事实的采集口径回归测试。"""

import pytest

from saas_bench.evaluation.fact_recorder import (
    SegmentAcquisitionFact,
    record_segment_day,
    record_subscription_day,
)
from saas_bench.simulator.config import AD_CHANNELS


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

    segment_row = conn.execute(
        """
        SELECT reputation, market_capacity_multiplier,
               calendar_cycle_multiplier, macroeconomic_multiplier,
               social_media_multiplier, demand_surge_multiplier,
               channel_leads_expected, network_leads_expected,
               total_leads_expected, actual_leads
        FROM _eval_segment_day
        WHERE day = 1 AND group_id = 'S1'
        """
    ).fetchone()
    assert segment_row is not None
    segment = dict(segment_row)
    assert segment["market_capacity_multiplier"] is not None
    assert segment["actual_leads"] is not None
    assert segment["total_leads_expected"] == pytest.approx(
        segment["reputation"]
        * segment["market_capacity_multiplier"]
        * segment["calendar_cycle_multiplier"]
        * segment["macroeconomic_multiplier"]
        * segment["social_media_multiplier"]
        * segment["demand_surge_multiplier"]
        * (segment["channel_leads_expected"] + segment["network_leads_expected"])
    )

    group_count = conn.execute("SELECT COUNT(*) FROM group_parameters").fetchone()[0]
    quality_count = conn.execute(
        "SELECT COUNT(*) FROM _eval_quality_day WHERE day = 1"
    ).fetchone()[0]
    assert quality_count == group_count * 3
    quality = dict(
        conn.execute(
            """
            SELECT * FROM _eval_quality_day
            WHERE day = 1 AND group_id = 'S1' AND plan = 'A'
            """
        ).fetchone()
    )
    assert quality["delivered_quality"] == pytest.approx(
        (
            quality["base_product_quality"]
            + quality["shared_quality_bonus"]
            + quality["group_quality_bonus"]
        )
        * quality["tier_multiplier"]
    )


def test_initial_channel_effectiveness_is_recorded(make_initialized_sim):
    original_effectiveness = {
        channel_id: dict(channel.leads_per_1000_dollars)
        for channel_id, channel in AD_CHANNELS.items()
    }
    try:
        conn, simulator, _ = make_initialized_sim()
        initial_rows = conn.execute(
            """
            SELECT channel_id, group_id, leads_per_1000_dollars
            FROM _eval_channel_effectiveness_event
            WHERE day = 0
            ORDER BY channel_id, group_id
            """
        ).fetchall()

        assert initial_rows
        assert all(row["leads_per_1000_dollars"] >= 0 for row in initial_rows)

        simulator.current_day = 30
        simulator._apply_monthly_leads_noise()
        changed_rows = conn.execute(
            """
            SELECT channel_id, group_id, leads_per_1000_dollars
            FROM _eval_channel_effectiveness_event
            WHERE day = 30
            ORDER BY channel_id, group_id
            """
        ).fetchall()
        assert len(changed_rows) == len(initial_rows)
        assert all(row["leads_per_1000_dollars"] >= 0 for row in changed_rows)
    finally:
        # AD_CHANNELS 是模拟器进程内共享状态，测试结束后恢复，避免污染其他用例。
        for channel_id, values in original_effectiveness.items():
            AD_CHANNELS[channel_id].leads_per_1000_dollars = values


def test_segment_day_records_all_groups_and_real_acquisition_inputs(
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim()
    customer_id = _insert_customer(conn, "small", "S1", 1)
    _insert_subscription(conn, customer_id, plan="A", price=20.0, start_day=7)
    conn.execute(
        """
        INSERT INTO customer_state (customer_id, satisfaction, relationship)
        VALUES (?, 0.25, 0.75)
        """,
        (customer_id,),
    )
    conn.execute(
        "UPDATE group_reputation SET reputation = 0.6 WHERE group_id = 'S1'"
    )
    conn.execute(
        "UPDATE group_awareness SET awareness = 0.2 WHERE group_id = 'S1'"
    )
    conn.execute(
        """
        UPDATE group_parameters
        SET drift_q_bias_total = 0.03, drift_c_max_total = -4.0
        WHERE group_id = 'S1'
        """
    )
    conn.execute(
        "UPDATE global_drift_state SET global_q_bias_total = 0.04 WHERE id = 1"
    )

    acquisition = SegmentAcquisitionFact(
        market_capacity_multiplier=0.9,
        calendar_cycle_multiplier=1.1,
        macroeconomic_multiplier=0.8,
        social_media_multiplier=1.2,
        demand_surge_multiplier=1.0,
        channel_leads_expected=12.0,
        network_leads_expected=3.0,
        total_leads_expected=8.5536,
        actual_leads=9,
    )
    record_segment_day(conn, 7, {"S1": acquisition})

    total_groups = conn.execute("SELECT COUNT(*) FROM group_parameters").fetchone()[0]
    recorded_groups = conn.execute(
        "SELECT COUNT(*) FROM _eval_segment_day WHERE day = 7"
    ).fetchone()[0]
    assert recorded_groups == total_groups

    s1 = dict(
        conn.execute(
            "SELECT * FROM _eval_segment_day WHERE day = 7 AND group_id = 'S1'"
        ).fetchone()
    )
    assert s1 == {
        "day": 7,
        "group_id": "S1",
        "info_level": 1,
        "reputation": pytest.approx(0.6),
        "awareness": pytest.approx(0.2),
        "group_quality_drift": pytest.approx(0.03),
        "group_budget_drift": pytest.approx(-4.0),
        "global_quality_drift": pytest.approx(0.04),
        "satisfaction_sample_accounts": 1,
        "avg_satisfaction": pytest.approx(0.25),
        "min_satisfaction": pytest.approx(0.25),
        "max_satisfaction": pytest.approx(0.25),
        "avg_relationship": pytest.approx(0.75),
        "market_capacity_multiplier": pytest.approx(0.9),
        "calendar_cycle_multiplier": pytest.approx(1.1),
        "macroeconomic_multiplier": pytest.approx(0.8),
        "social_media_multiplier": pytest.approx(1.2),
        "demand_surge_multiplier": pytest.approx(1.0),
        "channel_leads_expected": pytest.approx(12.0),
        "network_leads_expected": pytest.approx(3.0),
        "total_leads_expected": pytest.approx(8.5536),
        "actual_leads": 9,
    }

    undiscovered = dict(
        conn.execute(
            """
            SELECT esd.*
            FROM _eval_segment_day AS esd
            JOIN group_info_levels AS gil ON gil.group_id = esd.group_id
            WHERE esd.day = 7 AND gil.info_level = 0
            LIMIT 1
            """
        ).fetchone()
    )
    assert undiscovered["satisfaction_sample_accounts"] == 0
    assert undiscovered["avg_satisfaction"] is None
    assert undiscovered["total_leads_expected"] is None
    assert undiscovered["actual_leads"] is None


def test_no_hidden_snapshot_tables_are_created(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    hidden_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        if row[0].startswith("_hidden_")
    }
    assert hidden_tables == set()
