"""社交媒体查询与 Agent 帖子生效时序回归测试。"""

import json
import math

import pytest

from saas_bench.simulator.database import (
    add_agent_social_post,
    add_social_media_post,
    compute_social_media_multiplier,
)
from saas_bench.simulator.public_week_snapshot import (
    build_public_week_snapshot,
    load_public_week_snapshot,
    render_weekly_dashboard,
    save_public_week_snapshot,
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


@pytest.mark.parametrize(
    ("dashboard_day", "expected_days"),
    [(0, []), (7, [0, 1, 6]), (10, [6, 7, 8]), (14, [7, 8, 13]), (21, [14, 15, 20])],
)
def test_dashboard_social_posts_include_week_start_but_not_current_day(
    make_initialized_sim, dashboard_day, expected_days,
):
    conn, _, _ = make_initialized_sim(seed=42)
    post_days = (0, 1, 6, 7, 8, 13, 14, 15, 20, 21, 22)
    try:
        post_ids = {
            day: add_agent_social_post(conn, day, f"Update from day {day}")
            for day in post_days
        }
        snapshot = build_public_week_snapshot(conn, dashboard_day)
        assert {post.post_id for post in snapshot.social_posts} == {
            post_ids[day] for day in expected_days
        }
    finally:
        conn.close()


@pytest.mark.parametrize("post_day", [0, 7])
def test_dashboard_social_feedback_survives_snapshot_restore(
    make_initialized_sim, make_agent_tools, post_day,
):
    conn, _, config = make_initialized_sim(seed=42)
    try:
        result = make_agent_tools(conn, config, day=post_day).post_social_media("Product update")
        assert result.success
        post_id = result.data['agent_post_id']
        comment_ids = [
            add_social_media_post(
                conn, day=post_day + offset, customer_id=_first_customer_id(conn),
                sentiment="positive", content="Useful update", reply_to_agent_post_id=post_id,
            )
            for offset in (1, 5)
        ]
        conn.execute(
            "UPDATE agent_social_media_posts SET views = 1234, comment_post_ids = ?, "
            "effect_by_group = ?, reasoning_by_group = ? WHERE agent_post_id = ?",
            (json.dumps(comment_ids), '{"S1":0.85}', '{"S1":"PRIVATE_JUDGE_REASONING"}', post_id),
        )
        snapshot = build_public_week_snapshot(conn, post_day + 7)
        assert snapshot.to_dict()['social_posts'] == [{
            'post_id': post_id, 'views': 1234, 'comment_count': 2,
            'comment_post_ids': comment_ids, 'content_preview': "Product update",
            'content_truncated': False,
        }]
        dashboard = render_weekly_dashboard(snapshot)
        assert f"Post #{post_id}: 1234 views, 2 comments" in dashboard
        assert "PRIVATE_JUDGE_REASONING" not in dashboard

        save_public_week_snapshot(conn, snapshot)
        conn.commit()
        # 恢复读取当时保存的反馈，不能按后来变化的帖子数据重新计算。
        conn.execute("UPDATE agent_social_media_posts SET views = 9999 WHERE agent_post_id = ?", (post_id,))
        restored = load_public_week_snapshot(conn)
        assert restored == snapshot
        assert render_weekly_dashboard(restored) == dashboard
        assert build_public_week_snapshot(conn, post_day + 14).social_posts == []
    finally:
        conn.close()
