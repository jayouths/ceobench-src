"""社交媒体查询与 Agent 帖子生效时序回归测试。"""

import math

import pytest

from saas_bench.simulator.database import (
    add_agent_social_post,
    add_social_media_post,
    compute_social_media_multiplier,
)


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


def test_agent_post_uses_full_effect_on_first_acquisition_day(
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim(seed=42)
    add_agent_social_post(
        conn,
        day=7,
        content="A useful product update.",
        effect_by_group={"S1": 1.0},
    )

    first_day = compute_social_media_multiplier(conn, current_day=8, group_id="S1")
    second_day = compute_social_media_multiplier(conn, current_day=9, group_id="S1")

    assert first_day == pytest.approx(1.25)
    assert second_day == pytest.approx(
        1.0 + 0.25 * math.exp(-math.log(2) / 3.0)
    )


def test_step_day_judges_agent_posts_before_generating_customers(
    make_initialized_sim,
    monkeypatch,
):
    _, simulator, _ = make_initialized_sim(seed=42)
    calls = []

    monkeypatch.setattr(
        simulator,
        "_process_agent_social_posts",
        lambda config: calls.append("judge_agent_posts"),
    )

    def generate_customers(config):
        calls.append("generate_customers")
        return {
            "total_new": 0,
            "total_leads": 0,
            "new_individual_leads": 0,
            "new_enterprise_leads": 0,
            "new_individual_subscribers": 0,
            "segment_acquisition": {},
        }

    monkeypatch.setattr(simulator, "_generate_new_customers", generate_customers)

    simulator.step_day()

    assert calls == ["judge_agent_posts", "generate_customers"]
