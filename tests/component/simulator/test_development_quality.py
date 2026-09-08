"""Regression tests for global and targeted development quality gains."""

import math

import pytest

from saas_bench.simulator.config import (
    DEV_QUALITY_SPEND_SCALE,
    GLOBAL_DEV_QUALITY_COEFFICIENT,
    TARGETED_DEV_QUALITY_COEFFICIENT,
    BenchmarkConfig,
    compute_development_quality_improvement,
    get_development_quality_formula,
)
from saas_bench.simulator.database import get_global_state


@pytest.mark.parametrize("daily_spend", [0, 100, 1_000, 5_000, 25_000])
@pytest.mark.parametrize(
    ("targeted", "coefficient"),
    [
        (False, GLOBAL_DEV_QUALITY_COEFFICIENT),
        (True, TARGETED_DEV_QUALITY_COEFFICIENT),
    ],
)
def test_development_quality_formula_matches_authoritative_parameters(
    daily_spend,
    targeted,
    coefficient,
):
    expected = (
        coefficient * math.log(1 + daily_spend / DEV_QUALITY_SPEND_SCALE)
        if daily_spend > 0
        else 0.0
    )
    assert compute_development_quality_improvement(
        daily_spend,
        targeted=targeted,
    ) == pytest.approx(expected)


def test_simulator_and_targeted_tool_use_authoritative_formula(
    make_initialized_sim,
    make_agent_tools,
):
    """真实质量累积和 Agent 回执必须使用同一套研发参数。"""
    daily_spend = 5_000.0
    config = BenchmarkConfig(
        quality_shared_noise_scale=0.0,
        targeted_dev_spend={"S1": daily_spend},
    )
    conn, sim, config = make_initialized_sim(config=config)

    sim._update_global_state(
        {"spend_development": daily_spend, "spend_operations": 0.0}
    )

    assert get_global_state(conn, "q_shared_bonus") == pytest.approx(
        compute_development_quality_improvement(daily_spend)
    )
    assert get_global_state(conn, "q_group_bonus_S1") == pytest.approx(
        compute_development_quality_improvement(daily_spend, targeted=True)
    )

    tools = make_agent_tools(conn, config)
    result = tools.set_targeted_dev_spend({"S1": daily_spend})
    assert result.success
    assert get_development_quality_formula(targeted=True) in result.message
