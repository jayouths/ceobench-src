"""Regression tests for CEOBench R&D project accounting and completion."""

import pytest

from saas_bench.simulator.config import (
    MIN_RESEARCH_DURATION_DAYS,
    MIN_RESEARCH_QUALITY_BOOST,
    RESEARCH_TIERS,
    RESEARCH_TIERS_BY_ID,
)
from saas_bench.simulator.database import get_cash, get_global_state


def test_research_project_completion_applies_sampled_quality_boost(
    make_initialized_sim,
    make_agent_tools,
):
    conn, sim, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)
    research_tier = RESEARCH_TIERS_BY_ID[1]

    cash_before = get_cash(conn)
    result = tools.start_research_project(1)

    assert result.success
    assert get_cash(conn) == pytest.approx(cash_before - research_tier.cost)

    project = conn.execute(
        """
        SELECT project_id, tier, status, started_day, expected_completion_day,
               expected_quality_boost, actual_completion_day, quality_boost_applied
        FROM research_projects
        WHERE tier = 1
        """
    ).fetchone()
    assert project["project_id"] == "t1_1"
    assert project["status"] == "in_progress"
    assert project["started_day"] == 0
    assert project["expected_completion_day"] >= MIN_RESEARCH_DURATION_DAYS
    assert project["expected_quality_boost"] >= MIN_RESEARCH_QUALITY_BOOST
    assert project["actual_completion_day"] is None
    assert project["quality_boost_applied"] == pytest.approx(0.0)

    sim.current_day = project["expected_completion_day"] - 1
    sim._process_research_projects({})
    assert get_global_state(conn, "q_shared_bonus") == pytest.approx(0.0)

    sim.current_day = project["expected_completion_day"]
    sim._process_research_projects({})

    completed = conn.execute(
        """
        SELECT status, actual_completion_day, quality_boost_applied
        FROM research_projects
        WHERE project_id = 't1_1'
        """
    ).fetchone()
    expected_boost = project["expected_quality_boost"]
    assert completed["status"] == "completed"
    assert completed["actual_completion_day"] == project["expected_completion_day"]
    assert completed["quality_boost_applied"] == pytest.approx(expected_boost)
    assert get_global_state(conn, "q_shared_bonus") == pytest.approx(expected_boost)
    assert get_global_state(
        conn, "unreleased_base_quality_improvement"
    ) == pytest.approx(expected_boost)


def test_research_project_list_exposes_authoritative_distributions(
    make_initialized_sim,
    make_agent_tools,
):
    """研发列表必须逐档返回真实成本、分布参数和抽样下限。"""
    conn, _, config = make_initialized_sim()
    result = make_agent_tools(conn, config).list_research_projects()

    assert result.success
    assert len(result.data["tiers"]) == len(RESEARCH_TIERS)
    for returned, configured in zip(result.data["tiers"], RESEARCH_TIERS):
        assert returned["tier"] == configured.tier
        assert returned["name"] == configured.name
        assert returned["cost"] == configured.cost
        assert returned["mean_days"] == configured.mean_days
        assert returned["std_days"] == configured.std_days
        assert returned["minimum_duration_days"] == MIN_RESEARCH_DURATION_DAYS
        assert returned["mean_quality_boost"] == configured.mean_quality_boost
        assert returned["std_quality_boost"] == configured.std_quality_boost
        assert returned["minimum_quality_boost"] == MIN_RESEARCH_QUALITY_BOOST
