"""Regression tests for CEOBench market research and segment discovery."""

import pytest

from saas_bench.config import BenchmarkConfig
from saas_bench.database import (
    get_cash,
    get_group_info_level,
    get_undiscovered_groups,
)


def test_market_research_discovers_one_group_and_charges_cash(
    make_initialized_sim,
    make_agent_tools,
):
    config = BenchmarkConfig(seed=123, market_research_discover_prob=1.0)
    conn, _sim, config = make_initialized_sim(config=config, seed=123)
    tools = make_agent_tools(conn, config, seed=123)

    undiscovered_before = set(get_undiscovered_groups(conn))
    cash_before = get_cash(conn)

    result = tools.research_market()

    assert result.success
    assert get_cash(conn) == pytest.approx(cash_before - config.discovery_cost_level_1)

    discovered_group_id = result.data["discovered_group_id"]
    assert discovered_group_id in undiscovered_before
    assert get_group_info_level(conn, discovered_group_id) == 1
    assert len(get_undiscovered_groups(conn)) == len(undiscovered_before) - 1

    discovery = conn.execute(
        """
        SELECT day, cost, success, discovered_group_id, remaining_undiscovered
        FROM segment_discovery
        """
    ).fetchone()
    assert discovery["day"] == 0
    assert discovery["cost"] == pytest.approx(config.discovery_cost_level_1)
    assert discovery["success"] == 1
    assert discovery["discovered_group_id"] == discovered_group_id
    assert discovery["remaining_undiscovered"] == len(undiscovered_before) - 1

    snapshot = conn.execute(
        "SELECT snapshot_day FROM group_insight_snapshots WHERE group_id = ?",
        (discovered_group_id,),
    ).fetchone()
    assert snapshot["snapshot_day"] == 0
