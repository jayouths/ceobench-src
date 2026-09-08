"""Regression tests for CEOBench startup state and initial visibility."""

import pytest
from saas_bench.simulator.config import (
    INITIAL_CUSTOMER_GROUPS,
    get_info_level_noise_label,
)
from saas_bench.simulator.database import (
    get_cash,
    get_global_state,
    get_group_info_level,
    get_undiscovered_groups,
)








def test_initial_state_matches_benchmark_defaults(make_initialized_sim):
    conn, _sim, config = make_initialized_sim()

    assert get_cash(conn) == pytest.approx(config.initial_cash)
    assert get_global_state(conn, "q_shared_bonus") == pytest.approx(0.0)

    ledger = conn.execute("SELECT day, category, amount FROM ledger").fetchall()
    assert [(row["day"], row["category"], row["amount"]) for row in ledger] == [
        (0, "initial_funding", config.initial_cash)
    ]

    initial_config = conn.execute(
        "SELECT * FROM config_history WHERE day = 0"
    ).fetchone()
    assert initial_config["price_A"] == pytest.approx(config.default_price_A)
    assert initial_config["price_B"] == pytest.approx(config.default_price_B)
    assert initial_config["price_C"] == pytest.approx(config.default_price_C)
    assert initial_config["tier_A"] == config.default_tier_A
    assert initial_config["tier_B"] == config.default_tier_B
    assert initial_config["tier_C"] == config.default_tier_C
    assert initial_config["spend_advertising"] == pytest.approx(
        config.default_spend_advertising
    )
    assert initial_config["spend_operations"] == pytest.approx(
        config.default_spend_operations
    )
    assert initial_config["spend_development"] == pytest.approx(
        config.default_spend_development
    )
    assert initial_config["capacity_tier"] == config.default_capacity_tier
    assert initial_config["quota_A"] == config.default_quota_A
    assert initial_config["quota_B"] == config.default_quota_B
    assert initial_config["quota_C"] == config.default_quota_C

    for group_id, group_cfg in INITIAL_CUSTOMER_GROUPS.items():
        assert get_group_info_level(conn, group_id) == 1

        snapshot = conn.execute(
            """
            SELECT snapshot_day, snapshot_c_max, snapshot_q_min, snapshot_market_cap
            FROM group_insight_snapshots
            WHERE group_id = ?
            """,
            (group_id,),
        ).fetchone()
        assert snapshot["snapshot_day"] == 0
        assert snapshot["snapshot_c_max"] == pytest.approx(group_cfg.c_max_mean)
        assert snapshot["snapshot_q_min"] == pytest.approx(group_cfg.q_min_mean)
        assert snapshot["snapshot_market_cap"] == pytest.approx(
            group_cfg.base_market_cap
        )

    assert len(get_undiscovered_groups(conn)) == (
        config.discoverable_individual_count + config.discoverable_enterprise_count
    )


def test_initial_market_tools_report_level_one_information(
    make_initialized_sim,
    make_agent_tools,
):
    """Day 0 的数据库等级和 Agent 实际查询结果必须一致。"""
    conn, _sim, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)
    expected_noise = get_info_level_noise_label(config, 1)

    overview = tools.get_market_overview()
    assert overview.success
    assert overview.data["undiscovered_count"] == (
        config.discoverable_individual_count + config.discoverable_enterprise_count
    )
    initial_groups = {
        group["group_id"]: group for group in overview.data["known_groups"]
    }
    for group_id in INITIAL_CUSTOMER_GROUPS:
        assert initial_groups[group_id]["info_level"] == 1
        assert initial_groups[group_id]["noise"] == expected_noise

    insight = tools.get_group_insights("S1")
    assert insight.success
    assert insight.data["info_level"] == 1
    assert insight.data["noise"] == expected_noise
