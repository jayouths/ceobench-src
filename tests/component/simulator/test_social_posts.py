"""社交媒体查询口径回归测试。"""

import pytest

from saas_bench.simulator.database import add_social_media_post


def _first_customer_id(conn) -> int:
    return conn.execute(
        "SELECT customer_id FROM customers ORDER BY customer_id LIMIT 1"
    ).fetchone()[0]


def _add_post(conn, *, day: int, customer_id: int) -> None:
    add_social_media_post(
        conn,
        day=day,
        customer_id=customer_id,
        sentiment="neutral",
        content=f"post from day {day}",
    )


def test_social_posts_use_current_simulation_day_window(
    make_initialized_sim,
    make_agent_tools,
):
    """最近七天包含当日和前六天，不包含更早或未来的帖子。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id = _first_customer_id(conn)
    for day in (3, 4, 7, 10, 11):
        _add_post(conn, day=day, customer_id=customer_id)
    conn.commit()

    result = make_agent_tools(conn, config, day=10).get_social_posts(days=7)

    assert result.success
    assert [post["day"] for post in result.data["posts"]] == [10, 7, 4]
    assert result.data["total"] == 3
    assert "last 7 days" in result.message


def test_social_posts_apply_limit_after_time_filtering(
    make_initialized_sim,
    make_agent_tools,
):
    """limit 只截断已经落入时间窗口的帖子。"""
    conn, _, config = make_initialized_sim(seed=42)
    customer_id = _first_customer_id(conn)
    for day in (8, 9, 10):
        _add_post(conn, day=day, customer_id=customer_id)
    conn.commit()

    result = make_agent_tools(conn, config, day=10).get_social_posts(
        days=7,
        limit=2,
    )

    assert result.success
    assert [post["day"] for post in result.data["posts"]] == [10, 9]


@pytest.mark.parametrize(
    ("days", "limit", "expected_message"),
    [
        (0, 50, "Days must be a positive integer"),
        (-1, 50, "Days must be a positive integer"),
        (7, 0, "Limit must be a positive integer"),
    ],
)
def test_social_posts_reject_invalid_windows(
    make_initialized_sim,
    make_agent_tools,
    days,
    limit,
    expected_message,
):
    conn, _, config = make_initialized_sim(seed=42)

    result = make_agent_tools(conn, config).get_social_posts(
        days=days,
        limit=limit,
    )

    assert not result.success
    assert result.message == expected_message
