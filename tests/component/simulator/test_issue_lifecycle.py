"""客户工单生命周期回归测试。"""


def _insert_customer_with_subscription(conn) -> tuple[int, int]:
    customer_id = conn.execute(
        """
        INSERT INTO customers (
            customer_type, group_id, created_day,
            steepness_left, steepness_right, c_max, usage_demand,
            quality_sensitivity, price_sensitivity, willingness_to_pay,
            usage_scale, patience, seat_count
        )
        VALUES ('small', 'S1', 1, 0.01, 0.01, 100.0, 10.0,
                0.5, 0.5, 100.0, 1.0, 0.5, 1)
        """
    ).lastrowid
    conn.execute(
        """
        INSERT INTO customer_state (customer_id, satisfaction, open_issue_days)
        VALUES (?, 0.5, 3)
        """,
        (customer_id,),
    )
    subscription_id = conn.execute(
        """
        INSERT INTO subscriptions (
            customer_id, plan, listed_price, effective_price,
            start_day, status, billing_day_mod30
        )
        VALUES (?, 'A', 20.0, 20.0, 1, 'subscribed', 1)
        """,
        (customer_id,),
    ).lastrowid
    return customer_id, subscription_id


def test_last_subscription_end_closes_customer_issues(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    customer_id, subscription_id = _insert_customer_with_subscription(conn)
    conn.execute(
        """
        INSERT INTO service_day (
            day, total_usage_units, p95_ms, error_rate,
            downtime_minutes, capacity_tier, capacity_units
        )
        VALUES (7, 0, 0.0, 0.0, 0, 0, 50000)
        """
    )
    issue_id = conn.execute(
        """
        INSERT INTO issues (customer_id, group_id, open_day, days_open, status)
        VALUES (?, 'S1', 4, 3, 'open')
        """,
        (customer_id,),
    ).lastrowid

    conn.execute(
        """
        UPDATE subscriptions
        SET status = 'cancelled', end_day = 7, churn_reason = 'involuntary'
        WHERE subscription_id = ?
        """,
        (subscription_id,),
    )

    issue = conn.execute(
        """
        SELECT status, resolved_day, resolution_type
        FROM issues WHERE issue_id = ?
        """,
        (issue_id,),
    ).fetchone()
    customer_state = conn.execute(
        "SELECT open_issue_days FROM customer_state WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()

    assert dict(issue) == {
        "status": "resolved",
        "resolved_day": 7,
        "resolution_type": "customer_churned",
    }
    assert customer_state["open_issue_days"] == 0


def test_issue_stays_open_while_customer_has_an_active_subscription(
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim()
    customer_id, ending_subscription_id = _insert_customer_with_subscription(conn)
    conn.execute(
        """
        INSERT INTO subscriptions (
            customer_id, plan, listed_price, effective_price,
            start_day, status, billing_day_mod30
        )
        VALUES (?, 'B', 80.0, 80.0, 2, 'subscribed', 2)
        """,
        (customer_id,),
    )
    issue_id = conn.execute(
        """
        INSERT INTO issues (customer_id, group_id, open_day, days_open, status)
        VALUES (?, 'S1', 4, 3, 'open')
        """,
        (customer_id,),
    ).lastrowid

    conn.execute(
        """
        UPDATE subscriptions
        SET status = 'cancelled', end_day = 7
        WHERE subscription_id = ?
        """,
        (ending_subscription_id,),
    )

    issue = conn.execute(
        "SELECT status, resolved_day, resolution_type FROM issues WHERE issue_id = ?",
        (issue_id,),
    ).fetchone()
    assert dict(issue) == {
        "status": "open",
        "resolved_day": None,
        "resolution_type": None,
    }


def test_last_subscription_end_closes_all_open_issues(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    customer_id, subscription_id = _insert_customer_with_subscription(conn)
    conn.execute(
        """
        INSERT INTO service_day (
            day, total_usage_units, p95_ms, error_rate,
            downtime_minutes, capacity_tier, capacity_units
        )
        VALUES (9, 0, 0.0, 0.0, 0, 0, 50000)
        """
    )
    conn.executemany(
        """
        INSERT INTO issues (customer_id, group_id, open_day, days_open, status)
        VALUES (?, 'S1', ?, ?, 'open')
        """,
        [
            (customer_id, 2, 7),
            (customer_id, 6, 3),
        ],
    )

    conn.execute(
        """
        UPDATE subscriptions
        SET status = 'cancelled', end_day = 9, churn_reason = 'involuntary'
        WHERE subscription_id = ?
        """,
        (subscription_id,),
    )

    issues = conn.execute(
        """
        SELECT status, resolved_day, resolution_type
        FROM issues
        WHERE customer_id = ?
        ORDER BY issue_id
        """,
        (customer_id,),
    ).fetchall()
    assert [dict(issue) for issue in issues] == [
        {
            "status": "resolved",
            "resolved_day": 9,
            "resolution_type": "customer_churned",
        },
        {
            "status": "resolved",
            "resolved_day": 9,
            "resolution_type": "customer_churned",
        },
    ]
