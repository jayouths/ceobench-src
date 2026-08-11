import json
import sqlite3
from pathlib import Path

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.config import BenchmarkConfig
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.experiment_config import load_experiment_config
from saas_bench.server_entry import (
    _apply_simulator_llm_config,
    _apply_simulator_llm_env_overrides,
    _restore_simulator_llm_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return object()


class RecordingOpenAI:
    def __init__(self):
        self.responses = RecordingResponses()


def test_experiment_config_loads_all_experiment_and_model_fields():
    config = load_experiment_config(PROJECT_ROOT / "experiments/experiment.toml")

    assert config.experiment.seed == 42
    assert config.experiment.days == 70
    assert config.experiment.initial_cash == pytest.approx(1_000_000)
    assert config.decision_agent.model == "qwen3-coder:30b"
    assert config.decision_agent.reasoning_effort is None
    assert config.decision_agent.temperature == pytest.approx(0.7)
    assert config.decision_agent.input_cost_per_million == 0.0
    assert config.social_llm.model == "qwen3-coder:30b"
    assert config.social_llm.base_url == "http://localhost:11434/v1"
    assert config.social_llm.max_output_tokens == 1000
    assert config.enterprise_llm.model == "qwen3-coder:30b"
    assert config.enterprise_llm.temperature == pytest.approx(0.7)
    assert config.enterprise_llm.output_cost_per_million == 0.0


def test_experiment_config_path_is_required():
    with pytest.raises(ValueError, match="explicit experiment config is required"):
        load_experiment_config(None)


def test_runner_rejects_missing_decision_model_identity():
    with pytest.raises(ValueError, match="model must be explicitly configured"):
        BashAgentRunner(model=None, provider="openai")
    with pytest.raises(ValueError, match="provider must be explicitly configured"):
        BashAgentRunner(model="decision", provider=None)


def test_agent_rejects_missing_model_identity():
    with pytest.raises(ValueError, match="agent model must be explicitly configured"):
        BashAgent(tool_descriptions=[], client=object())


def test_smoke_config_is_limited_to_one_week():
    config = load_experiment_config(PROJECT_ROOT / "experiments/smoke.toml")

    assert config.experiment.days == 7
    assert config.experiment.label == "smoke-qwen-coder"


def test_full_config_uses_benchmark_horizon():
    config = load_experiment_config(PROJECT_ROOT / "experiments/full.toml")

    assert config.experiment.days == 500
    assert config.experiment.label == "full-qwen-coder"


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        ("[experiment]\nseed = 42\n", "models must be explicitly configured"),
        (
            "[models.decision_agent]\nprovider = 'openai'\nmodel = 'decision'\n",
            "models.social_llm must be explicitly configured",
        ),
        (
            """
[models.decision_agent]
provider = "openai"
model = "decision"
[models.social_llm]
provider = "openai"
model = "social"
[models.enterprise_llm]
provider = "openai"
""",
            "models.enterprise_llm must explicitly configure: model",
        ),
    ],
)
def test_every_model_identity_must_be_explicit(tmp_path, config_text, message):
    path = tmp_path / "missing-model.toml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


def test_experiment_config_rejects_unknown_settings(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text(
        """
[experiment]
seed = 42
unknown = true
[models.decision_agent]
provider = "openai"
model = "decision"
[models.social_llm]
provider = "openai"
model = "social"
[models.enterprise_llm]
provider = "openai"
model = "enterprise"
"""
    )

    with pytest.raises(ValueError, match="unknown experiment setting"):
        load_experiment_config(path)


def test_model_costs_must_be_configured_as_a_pair(tmp_path):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    path = tmp_path / "partial-cost.toml"
    path.write_text(text.replace("output_cost_per_million = 0.0\n", "", 1))

    with pytest.raises(ValueError, match="input and output costs together"):
        load_experiment_config(path)


def test_short_experiment_is_allowed_and_runner_rounds_to_zero(tmp_path):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    path = tmp_path / "short.toml"
    path.write_text(text.replace("days = 7", "days = 6", 1))

    config = load_experiment_config(path)
    runner = BashAgentRunner(
        model=config.decision_agent.model,
        provider=config.decision_agent.provider,
        base_url=config.decision_agent.base_url,
        api_key="test",
        total_days=config.experiment.days,
        workspace_base=tmp_path / "runs",
    )

    assert runner.total_days == 0


def test_resume_rejects_changed_experiment_configuration(tmp_path):
    run_dir = tmp_path / "run_existing"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "existing",
        "model": "original-model",
        "provider": "openai",
        "base_url": "http://localhost:11434/v1",
        "seed": 42,
        "scenario": "default",
        "total_days": 70,
        "initial_cash": 1_000_000.0,
    }))

    with pytest.raises(ValueError, match="model: previous='original-model'"):
        BashAgentRunner(
            model="changed-model",
            provider="openai",
            base_url="http://localhost:11434/v1",
            api_key="test",
            seed=42,
            scenario="default",
            total_days=70,
            initial_cash=1_000_000.0,
            continue_from=run_dir,
        )


def test_simulator_settings_survive_environment_and_session_round_trip(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_KEY", "test-key")
    overrides = {
        "social_post_llm_provider": "openai",
        "social_post_llm_model": "social-test",
        "social_post_llm_base_url": "http://social.test/v1",
        "social_post_llm_api_key_env": "LOCAL_LLM_KEY",
        "social_post_llm_reasoning_effort": "none",
        "social_post_llm_temperature": 0.31,
        "social_post_llm_top_p": 0.72,
        "social_post_llm_max_tokens": 123,
        "social_post_llm_timeout_seconds": 45.0,
        "enterprise_llm_provider": "openai",
        "enterprise_llm_model": "enterprise-test",
        "enterprise_llm_base_url": "http://enterprise.test/v1",
        "enterprise_llm_api_key_env": "LOCAL_LLM_KEY",
        "enterprise_llm_reasoning_effort": None,
        "enterprise_llm_temperature": 0.41,
        "enterprise_llm_top_p": 0.82,
        "enterprise_llm_max_tokens": 234,
        "enterprise_llm_timeout_seconds": 55.0,
    }
    monkeypatch.setenv("CEOBENCH_SIMULATOR_LLM_CONFIG", json.dumps(overrides))

    created = BenchmarkConfig()
    _apply_simulator_llm_env_overrides(created)
    session_values = _apply_simulator_llm_config(created)
    restored = BenchmarkConfig()
    _restore_simulator_llm_config(restored, {"simulator_llm": session_values})

    for field, expected in overrides.items():
        assert getattr(restored, field) == expected


def test_social_and_enterprise_openai_calls_receive_configured_api_parameters():
    social = RecordingOpenAI()
    enterprise = RecordingOpenAI()
    config = BenchmarkConfig(
        social_post_llm_provider="openai",
        social_post_llm_model="social-test",
        social_post_llm_reasoning_effort="none",
        social_post_llm_temperature=0.31,
        social_post_llm_top_p=0.72,
        social_post_llm_max_tokens=123,
        enterprise_llm_provider="openai",
        enterprise_llm_model="enterprise-test",
        enterprise_llm_reasoning_effort="high",
        enterprise_llm_temperature=0.41,
        enterprise_llm_top_p=0.82,
        enterprise_llm_max_tokens=234,
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=config,
        social_openai_client=social,
        enterprise_openai_client=enterprise,
    )

    simulator.create_social_response("social system", "social user")
    simulator.create_enterprise_response("enterprise system", "enterprise user")

    assert social.responses.calls == [{
        "model": "social-test",
        "input": [
            {"role": "system", "content": "social system"},
            {"role": "user", "content": "social user"},
        ],
        "max_output_tokens": 123,
        "temperature": 0.31,
        "top_p": 0.72,
        "reasoning": {"effort": "none"},
    }]
    assert enterprise.responses.calls == [{
        "model": "enterprise-test",
        "input": [
            {"role": "system", "content": "enterprise system"},
            {"role": "user", "content": "enterprise user"},
        ],
        "max_output_tokens": 234,
        "temperature": 0.41,
        "top_p": 0.82,
        "reasoning": {"effort": "high"},
    }]


def test_local_model_cost_is_zero_when_explicitly_configured():
    config = BenchmarkConfig(
        agent_llm_model="qwen3-coder:30b",
        social_post_llm_input_cost_per_million=0.0,
        social_post_llm_output_cost_per_million=0.0,
        enterprise_llm_input_cost_per_million=0.0,
        enterprise_llm_output_cost_per_million=0.0,
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=config,
    )

    assert simulator._calculate_cost(
        1_000_000, 1_000_000, model="qwen3-coder:30b", purpose="customer_social_post"
    ) == 0.0
    assert simulator._calculate_cost(
        1_000_000, 1_000_000, model="qwen3-coder:30b", purpose="customer_negotiation"
    ) == 0.0


def test_unknown_model_cost_requires_explicit_pricing():
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=BenchmarkConfig(agent_llm_model="unknown-model"),
    )

    with pytest.raises(ValueError, match="No token pricing configured"):
        simulator._calculate_cost(
            1, 1, model="unknown-model", purpose="customer_social_post"
        )


def test_decision_agent_request_builder_uses_config_without_hidden_defaults():
    agent = BashAgent.__new__(BashAgent)
    agent.model = "decision-test"
    agent.max_output_tokens = 345
    agent.temperature = 0.51
    agent.top_p = 0.92
    agent.reasoning_effort = "none"
    agent._get_system_prompt_with_memory = lambda: "system"

    params = agent._build_openai_responses_kwargs([{"role": "user", "content": "x"}], [])

    assert params["max_output_tokens"] == 345
    assert params["temperature"] == pytest.approx(0.51)
    assert params["top_p"] == pytest.approx(0.92)
    assert params["reasoning"] == {"effort": "none", "summary": "auto"}

    agent.temperature = None
    agent.top_p = None
    agent.reasoning_effort = None
    omitted = agent._build_openai_responses_kwargs([], [])
    assert "temperature" not in omitted
    assert "top_p" not in omitted
    assert "reasoning" not in omitted
