"""Regression tests for CEOBench startup state and initial visibility."""

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.config import INITIAL_CUSTOMER_GROUPS
from saas_bench.customer_llm import _optional_reasoning
from saas_bench.database import (
    get_cash,
    get_global_state,
    get_group_info_level,
    get_undiscovered_groups,
)


def test_runner_preserves_reasoning_configuration():
    omitted = BashAgentRunner(
        model="test-model",
        provider="openai",
        base_url="http://localhost:11434/v1",
        api_key="test",
    )
    disabled = BashAgentRunner(
        model="test-model",
        provider="openai",
        base_url="http://localhost:11434/v1",
        api_key="test",
        reasoning_effort="none",
    )

    assert omitted.reasoning_effort is None
    assert disabled.reasoning_effort == "none"
    assert omitted.temperature is None
    assert omitted.top_p is None


def test_runner_preserves_sampling_configuration():
    runner = BashAgentRunner(
        model="test-model",
        provider="openai",
        base_url="http://localhost:11434/v1",
        api_key="test",
        temperature=0.6,
        top_p=0.95,
    )

    assert runner.temperature == pytest.approx(0.6)
    assert runner.top_p == pytest.approx(0.95)


def test_simulator_reasoning_omits_only_unconfigured_value():
    assert _optional_reasoning(None) == {}
    assert _optional_reasoning("none") == {"reasoning": {"effort": "none"}}
    assert _optional_reasoning("high") == {"reasoning": {"effort": "high"}}


def test_bash_agent_routes_supported_endpoint_to_responses_api():
    agent = BashAgent.__new__(BashAgent)
    agent.use_anthropic = False
    agent.supports_responses_api = True
    agent._call_openai = lambda: "chat"
    agent._call_openai_responses = lambda: "responses"

    agent.reasoning_effort = None
    assert agent._call_llm() == "responses"

    agent.reasoning_effort = "none"
    assert agent._call_llm() == "responses"

    agent.supports_responses_api = False
    assert agent._call_llm() == "chat"


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
