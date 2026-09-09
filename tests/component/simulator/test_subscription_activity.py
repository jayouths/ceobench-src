"""订阅生命周期活动统计测试。"""

from saas_bench.simulator.subscription_activity import (
    consume_weekly_subscription_activity,
)


def _insert_customer(conn, customer_type: str, seat_count: int) -> int:
    cursor = conn.execute(
        """
        INSERT INTO customers (
            customer_type, group_id, created_day,
            steepness_left, steepness_right, c_max, q_max, q_min,
            usage_demand, quality_sensitivity, price_sensitivity,
            willingness_to_pay, usage_scale, patience, seat_count
        )
        VALUES (?, ?, 0, 0.01, 0.01, 200, 0.9, 0.1,
                10, 0.5, 0.5, 200, 10, 0.5, ?)
        """,
        (customer_type, "S1" if customer_type == "small" else "E1", seat_count),
    )
    return int(cursor.lastrowid)


def _insert_subscription(
    conn,
    customer_id: int,
    *,
    plan: str,
    status: str = "subscribed",
    seats: int = 1,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO subscriptions (
            customer_id, plan, listed_price, effective_price, seat_count,
            start_day, status, billing_day_mod30
        )
        VALUES (?, ?, 50, 50, ?, 0, ?, 0)
        """,
        (customer_id, plan, seats, status),
    )
    return int(cursor.lastrowid)


def test_subscription_activity_captures_all_customer_types_and_plan_changes(
    make_initialized_sim,
):
    """统一触发器按账户类型记录取消、席位和套餐变更。"""
    conn, _, _ = make_initialized_sim(seed=42)
    small = _insert_customer(conn, "small", 1)
    enterprise = _insert_customer(conn, "large", 25)
    small_sub = _insert_subscription(conn, small, plan="B")
    enterprise_sub = _insert_subscription(
        conn, enterprise, plan="B", seats=25
    )

    conn.execute(
        "UPDATE subscriptions SET plan = 'C' WHERE subscription_id = ?",
        (small_sub,),
    )
    conn.execute(
        "UPDATE subscriptions SET plan = 'A' WHERE subscription_id = ?",
        (enterprise_sub,),
    )
    conn.execute(
        "UPDATE subscriptions SET status = 'cancelled', end_day = 7 WHERE subscription_id = ?",
        (small_sub,),
    )
    conn.execute(
        "UPDATE subscriptions SET status = 'cancelled', end_day = 7 WHERE subscription_id = ?",
        (enterprise_sub,),
    )

    activity = consume_weekly_subscription_activity(conn)

    assert activity.individual_cancellations == 1
    assert activity.enterprise_cancellations == 1
    assert activity.enterprise_cancelled_seats == 25
    assert activity.individual_upgrades == 1
    assert activity.enterprise_upgrades == 0
    assert activity.individual_downgrades == 0
    assert activity.enterprise_downgrades == 1

    # 周报游标已持久化，同一批事件不得在下周重复出现。
    assert consume_weekly_subscription_activity(conn).enterprise_cancellations == 0


def test_lead_conversion_does_not_count_as_enterprise_plan_change(
    make_initialized_sim,
):
    """新线索首次选择套餐属于成交，不是存量客户升降级。"""
    conn, _, _ = make_initialized_sim(seed=42)
    enterprise = _insert_customer(conn, "large", 10)
    subscription_id = _insert_subscription(
        conn,
        enterprise,
        plan="A",
        status="lead",
        seats=10,
    )

    conn.execute(
        """
        UPDATE subscriptions
        SET status = 'subscribed', plan = 'C'
        WHERE subscription_id = ?
        """,
        (subscription_id,),
    )

    activity = consume_weekly_subscription_activity(conn)
    assert activity.enterprise_upgrades == 0
    assert activity.enterprise_downgrades == 0


def test_week_result_includes_activity_after_previous_week_boundary(
    make_initialized_sim,
):
    """上周报表返回后发生的取消必须进入下一份周报。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    enterprise = _insert_customer(conn, "large", 25)
    subscription_id = _insert_subscription(
        conn,
        enterprise,
        plan="B",
        seats=25,
    )
    # 模拟 Agent 在上一份周报之后、推进下周之前取消订阅。
    conn.execute(
        """
        UPDATE subscriptions
        SET status = 'cancelled', end_day = 0
        WHERE subscription_id = ?
        """,
        (subscription_id,),
    )

    first_week = simulator.step_week()
    second_week = simulator.step_week()

    assert first_week.enterprise_cancellations == 1
    assert first_week.enterprise_cancelled_seats == 25
    assert second_week.enterprise_cancellations == 0
    assert second_week.enterprise_cancelled_seats == 0
