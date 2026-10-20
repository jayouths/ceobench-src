"""竞争帖子只围绕最新活动事件生成，不增加公开信息总量。"""

import pytest

from saas_bench.simulator.config import BenchmarkConfig


@pytest.mark.parametrize("day,events,expected", [
    (102, [(100, 103, "older"), (101, 104, "latest")], "latest"),
    (102, [(101, 104, "latest"), (100, 103, "older")], "latest"),
    (102, [(101, 104, "first"), (101, 104, "last")], "last"),
    (102, [(101, 104, "current"), (103, 106, "future")], "current"),
    (102, [(103, 106, "future")], None),
    (103, [(100, 103, "boundary")], "boundary"),
    (104, [(100, 103, "expired")], None),
    (102, [], None),
])
def test_competitor_posts_select_latest_active_event(
    make_initialized_sim, monkeypatch, day, events, expected,
):
    config = BenchmarkConfig(
        seed=42, competitor_post_boost_noise_min=0.0,
        competitor_post_boost_noise_max=0.0,
    )
    conn, simulator, _ = make_initialized_sim(config=config, seed=42)
    simulator.current_day = day
    conn.executemany("""
        INSERT INTO competitor_events
            (start_day, post_end_day, boost_amount, description, applied)
        VALUES (?, ?, 0.12, ?, 1)
    """, events)
    # 只捕获生成请求，不调用环境模型；检验真实选择及落表链路。
    contexts = []

    def capture(_competitor, boost, severity, description, _product, _perspective):
        contexts.append((boost, severity, description))
        return f"context: {description}"

    monkeypatch.setattr(simulator, "customer_simulator", object())
    monkeypatch.setattr(simulator, "_generate_competitor_post_llm", capture)
    notification_count = conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0]
    drift = tuple(conn.execute("SELECT * FROM global_drift_state").fetchone())

    simulator._generate_competitor_event_posts()

    count = config.competitor_event_posts_per_day if expected else 0
    assert contexts == [(0.12, "major", expected)] * count
    posts = conn.execute("""
        SELECT content, sentiment, reputation_impact, influence_score
        FROM social_media_posts WHERE day = ? AND customer_id = ?
    """, (day, simulator._market_observer_id)).fetchall()
    assert [tuple(post) for post in posts] == [
        (f"context: {expected}", "neutral", 0.0, 0.0)
    ] * count
    assert conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0] == notification_count
    assert tuple(conn.execute("SELECT * FROM global_drift_state").fetchone()) == drift
