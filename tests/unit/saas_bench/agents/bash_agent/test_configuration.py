"""Bash Agent 对应逻辑的快速单元测试。"""

import re

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent

from saas_bench.agents.bash_agent.runner import BashAgentRunner

from saas_bench.experiment.experiment_config import load_experiment_config
from tests.support.harness import TEST_CONFIG


def test_runner_rejects_missing_decision_model_identity():
    with pytest.raises(ValueError, match="model must be explicitly configured"):
        BashAgentRunner(
            model=None,
            api_type="openai_responses",
            max_output_tokens=100,
            max_decision_turns_per_batch=100,
        )

def test_runner_rejects_missing_decision_request_limit():
    with pytest.raises(ValueError, match="max_output_tokens must be configured"):
        BashAgentRunner(
            model="decision",
            api_type="openai_responses",
            tool_choice="required",
            max_decision_turns_per_batch=100,
            max_invalid_responses_per_turn=3,
        )

def test_agent_rejects_missing_model_identity():
    with pytest.raises(ValueError, match="agent model must be explicitly configured"):
        BashAgent(tool_descriptions=[], client=object(), api_type="openai_responses")

def test_short_experiment_is_allowed_and_runner_rounds_to_zero(tmp_path):
    text = TEST_CONFIG.read_text()
    path = tmp_path / "short.toml"
    path.write_text(text.replace("days = 7", "days = 6", 1))

    config = load_experiment_config(path)
    runner = BashAgentRunner(
        experiment_name="test",
        model=config.decision_agent.model,
        api_type=config.decision_agent.api_type,
        tool_choice=config.decision_agent.tool_choice,
        base_url=config.decision_agent.base_url,
        api_key_required=False,
        max_output_tokens=config.decision_agent.max_output_tokens,
        pricing=config.decision_agent.pricing,
        total_days=config.experiment.days,
        max_decision_turns_per_batch=config.experiment.max_decision_turns_per_batch,
        max_invalid_responses_per_turn=config.experiment.max_invalid_responses_per_turn,
        workspace_base=tmp_path / "runs",
    )

    assert runner.total_days == 0
    assert runner.workspace_dir.parent.name == "test"
    assert re.fullmatch(
        r"\d{8}-\d{6}_seed-42_[0-9a-f]{8}", runner.workspace_dir.name
    )

def test_runner_preserves_reasoning_configuration():
    omitted = BashAgentRunner(
        experiment_name="test",
        model="test-model",
        api_type="openai_responses",
        tool_choice="required",
        base_url="http://localhost:11434/v1",
        api_key_required=False,
        max_output_tokens=100,
        max_decision_turns_per_batch=100,
        max_invalid_responses_per_turn=3,
        pricing={"test-model": {"currency": "USD", "uncached_input_cost_per_million": 0.0, "cached_input_cost_per_million": 0.0, "output_cost_per_million": 0.0}},
    )
    disabled = BashAgentRunner(
        experiment_name="test",
        model="test-model",
        api_type="openai_responses",
        tool_choice="required",
        base_url="http://localhost:11434/v1",
        api_key_required=False,
        max_output_tokens=100,
        max_decision_turns_per_batch=100,
        max_invalid_responses_per_turn=3,
        pricing={"test-model": {"currency": "USD", "uncached_input_cost_per_million": 0.0, "cached_input_cost_per_million": 0.0, "output_cost_per_million": 0.0}},
        reasoning_effort="none",
    )

    assert omitted.reasoning_effort is None
    assert disabled.reasoning_effort == "none"
    assert omitted.temperature is None
    assert omitted.top_p is None

def test_runner_preserves_sampling_configuration():
    runner = BashAgentRunner(
        experiment_name="test",
        model="test-model",
        api_type="openai_responses",
        tool_choice="required",
        base_url="http://localhost:11434/v1",
        api_key_required=False,
        max_output_tokens=100,
        max_decision_turns_per_batch=100,
        max_invalid_responses_per_turn=3,
        pricing={"test-model": {"currency": "USD", "uncached_input_cost_per_million": 0.0, "cached_input_cost_per_million": 0.0, "output_cost_per_million": 0.0}},
        temperature=0.6,
        top_p=0.95,
    )

    assert runner.temperature == pytest.approx(0.6)
    assert runner.top_p == pytest.approx(0.95)

def test_bash_agent_routes_by_explicit_api_type():
    agent = BashAgent.__new__(BashAgent)
    agent._call_openai = lambda: "chat"
    agent._call_openai_responses = lambda: "responses"

    agent.api_type = "openai_responses"
    assert agent._call_llm() == "responses"
    agent.api_type = "openai_chat_completions"
    assert agent._call_llm() == "chat"
    agent.api_type = "unknown"
    with pytest.raises(ValueError, match="Unsupported decision-agent api_type"):
        agent._call_llm()
