"""实验 TOML 配置的快速单元测试。"""

import re
from pathlib import Path

import pytest

from saas_bench.experiment_config import load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_experiment_config_loads_all_experiment_and_model_fields():
    config = load_experiment_config(PROJECT_ROOT / "experiments/experiment.toml")

    assert config.experiment.seed == 42
    assert config.experiment.days == 70
    assert config.experiment.initial_cash == pytest.approx(1_000_000)
    assert config.experiment.max_decision_turns_per_batch == 100
    assert config.experiment.max_invalid_responses_per_turn == 3
    assert config.modules.analysis.enabled is False
    assert config.modules.analysis.max_schema_retries == 1
    assert config.modules.analysis.max_enterprise_threads == 50
    assert config.analysis is None
    assert config.decision_agent.model == "qwen3-coder:30b"
    assert config.decision_agent.reasoning_effort is None
    assert config.decision_agent.temperature == pytest.approx(0.7)
    assert config.decision_agent.pricing["qwen3-coder:30b"] == {
        "currency": "CNY",
        "uncached_input_cost_per_million": 0.0,
        "cached_input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
    }
    assert config.social_llm.model == "qwen3-coder:30b"
    assert config.social_llm.base_url == "http://localhost:11434/v1"
    assert config.social_llm.max_output_tokens == 1000

def test_analysis_model_is_required_only_when_module_is_enabled(tmp_path):
    baseline_text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    enabled_text = baseline_text.replace("enabled = false", "enabled = true", 1)
    missing_model_path = tmp_path / "analysis-missing-model.toml"
    missing_model_path.write_text(enabled_text)

    with pytest.raises(
        ValueError,
        match=r"models\.analysis must be explicitly configured",
    ):
        load_experiment_config(missing_model_path)

    configured_path = tmp_path / "analysis-configured.toml"
    configured_path.write_text(
        enabled_text
        + """

[models.analysis]
provider = "openai_compatible"
api_type = "openai_chat_completions"
model = "analysis-model"
base_url = "http://localhost:11434/v1"
api_key_required = false
reasoning_effort = "none"
temperature = 0.2
max_output_tokens = 2000
timeout_seconds = 600

[models.analysis.pricing."analysis-model"]
currency = "CNY"
uncached_input_cost_per_million = 0.0
cached_input_cost_per_million = 0.0
output_cost_per_million = 0.0

[models.analysis.tasks.role_report]
max_output_tokens = 1500

[models.analysis.tasks.state_reconstruction]
max_output_tokens = 2000
"""
    )

    config = load_experiment_config(configured_path)

    assert config.modules.analysis.enabled is True
    assert config.analysis is not None
    assert config.analysis.model == "analysis-model"
    assert config.analysis.tasks["role_report"]["max_output_tokens"] == 1500
    assert config.analysis.tasks["state_reconstruction"]["max_output_tokens"] == 2000

def test_analysis_settings_must_be_explicitly_configured(tmp_path):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()

    missing_modules_path = tmp_path / "missing-modules.toml"
    missing_modules_path.write_text(
        text.replace(
            "[modules.analysis]\nenabled = false\nmax_schema_retries = 1                  # JSON Schema 校验失败后的最大修复次数\nmax_enterprise_threads = 50             # signals.json 最多保留的开放企业谈判明细数\n\n",
            "",
            1,
        )
    )
    with pytest.raises(ValueError, match="modules must be explicitly configured"):
        load_experiment_config(missing_modules_path)

    missing_enabled_path = tmp_path / "missing-analysis-enabled.toml"
    missing_enabled_path.write_text(text.replace("enabled = false\n", "", 1))
    with pytest.raises(
        ValueError,
        match="modules.analysis must explicitly configure: enabled",
    ):
        load_experiment_config(missing_enabled_path)

    missing_retries_path = tmp_path / "missing-analysis-retries.toml"
    missing_retries_path.write_text(
        text.replace("max_schema_retries = 1", "", 1)
    )
    with pytest.raises(
        ValueError,
        match="modules.analysis must explicitly configure: max_schema_retries",
    ):
        load_experiment_config(missing_retries_path)

@pytest.mark.parametrize("value", ["1", '"true"'])
def test_analysis_enabled_must_be_boolean(tmp_path, value):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    path = tmp_path / "invalid-analysis-switch.toml"
    path.write_text(text.replace("enabled = false", f"enabled = {value}", 1))

    with pytest.raises(ValueError, match="modules.analysis.enabled must be a boolean"):
        load_experiment_config(path)

@pytest.mark.parametrize("value", ["-1", "true"])
def test_analysis_schema_retries_must_be_non_negative_integer(tmp_path, value):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    path = tmp_path / "invalid-analysis-schema-retries.toml"
    path.write_text(
        text.replace(
            "max_schema_retries = 1",
            f"max_schema_retries = {value}",
            1,
        )
    )

    with pytest.raises(ValueError, match="must be a non-negative integer"):
        load_experiment_config(path)


@pytest.mark.parametrize("value", ["0", "-1", "true"])
def test_analysis_enterprise_thread_limit_must_be_positive_integer(tmp_path, value):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    path = tmp_path / "invalid-analysis-thread-limit.toml"
    path.write_text(
        text.replace("max_enterprise_threads = 50", f"max_enterprise_threads = {value}", 1)
    )

    with pytest.raises(ValueError, match="max_enterprise_threads must be a positive integer"):
        load_experiment_config(path)

def test_experiment_config_path_is_required():
    with pytest.raises(ValueError, match="explicit experiment config path is required"):
        load_experiment_config(None)

def test_smoke_config_is_limited_to_one_week():
    config = load_experiment_config(PROJECT_ROOT / "experiments/smoke.toml")

    assert config.experiment.days == 7
    assert config.experiment.label == "smoke-qwen-coder"

def test_deepseek_smoke_config_uses_official_peak_prices():
    config = load_experiment_config(
        PROJECT_ROOT / "experiments/smoke-deepseek.toml"
    )

    assert config.experiment.days == 7
    assert config.decision_agent.model == "deepseek-v4-pro"
    assert config.decision_agent.reasoning_effort == "low"
    assert config.decision_agent.api_key_env == "DEEPSEEK_API_KEY"
    assert config.decision_agent.pricing["deepseek-v4-pro"] == {
        "currency": "USD",
        "uncached_input_cost_per_million": 1.32,
        "cached_input_cost_per_million": 0.044,
        "output_cost_per_million": 3.96,
    }
    assert config.social_llm.model == "deepseek-v4-flash"
    assert config.social_llm.reasoning_effort == "none"
    assert config.social_llm.pricing["deepseek-v4-flash"] == {
        "currency": "USD",
        "uncached_input_cost_per_million": 0.44,
        "cached_input_cost_per_million": 0.014,
        "output_cost_per_million": 1.32,
    }

def test_autodl_models_map_to_official_deepseek_pricing():
    config = load_experiment_config(
        PROJECT_ROOT / "experiments/smoke-autodl.toml"
    )

    assert config.decision_agent.model == "deepseek-v4-pro-202606"
    assert config.decision_agent.pricing_model_map == {
        "deepseek-v4-pro-202606": "deepseek-v4-pro",
        "DeepSeek-V4-Pro": "deepseek-v4-pro",
        "DeepSeek-V4-Pro-0813": "deepseek-v4-pro",
    }
    assert config.social_llm.model == "DeepSeek-V4-Flash"
    assert config.social_llm.pricing_model_map[
        "deepseek-v4-flash-202605"
    ] == "deepseek-v4-flash"

def test_full_config_uses_benchmark_horizon():
    config = load_experiment_config(PROJECT_ROOT / "experiments/full.toml")

    assert config.experiment.days == 500
    assert config.experiment.label == "full-qwen-coder"

@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        (
            "[experiment]\nseed = 42\nmax_decision_turns_per_batch = 100\nmax_invalid_responses_per_turn = 3\n",
            "models must be explicitly configured",
        ),
        (
            "[experiment]\nmax_decision_turns_per_batch = 100\nmax_invalid_responses_per_turn = 3\n"
            "[models.decision_agent]\nprovider = 'openai'\napi_type = 'openai_responses'\nmodel = 'decision'\nmax_output_tokens = 100\napi_key_required = false\n[models.decision_agent.pricing.decision]\ncurrency = 'USD'\nuncached_input_cost_per_million = 0\ncached_input_cost_per_million = 0\noutput_cost_per_million = 0\n",
            "models.social_llm must be explicitly configured",
        ),
    ],
)
def test_every_model_identity_must_be_explicit(tmp_path, config_text, message):
    path = tmp_path / "missing-model.toml"
    path.write_text(config_text)

    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)

def test_decision_turn_limit_must_be_explicit(tmp_path):
    path = tmp_path / "missing-turn-limit.toml"
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    path.write_text(
        re.sub(r"^max_decision_turns_per_batch\s*=.*\n", "", text, count=1, flags=re.MULTILINE)
    )

    with pytest.raises(
        ValueError, match="max_decision_turns_per_batch"
    ):
        load_experiment_config(path)

def test_invalid_response_limit_must_be_explicit(tmp_path):
    path = tmp_path / "missing-invalid-response-limit.toml"
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    path.write_text(
        re.sub(r"^max_invalid_responses_per_turn\s*=.*\n", "", text, count=1, flags=re.MULTILINE)
    )

    with pytest.raises(ValueError, match="max_invalid_responses_per_turn"):
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
"""
    )

    with pytest.raises(ValueError, match="unknown experiment setting"):
        load_experiment_config(path)

def test_model_costs_must_be_configured_as_a_pair(tmp_path):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    path = tmp_path / "partial-cost.toml"
    path.write_text(
        re.sub(r"^output_cost_per_million\s*=.*\n", "", text, count=1, flags=re.MULTILINE)
    )

    with pytest.raises(ValueError, match="explicitly configure: output_cost_per_million"):
        load_experiment_config(path)

def test_pricing_model_map_must_target_configured_official_price(tmp_path):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    pricing_header = '[models.decision_agent.pricing."qwen3-coder:30b"]'
    path = tmp_path / "invalid-pricing-map.toml"
    path.write_text(text.replace(
        pricing_header,
        '[models.decision_agent.pricing_model_map]\n'
        '"qwen3-coder:30b" = "missing-official-model"\n\n'
        + pricing_header,
        1,
    ))

    with pytest.raises(ValueError, match="targets unknown pricing model"):
        load_experiment_config(path)

@pytest.mark.parametrize(
    "section",
    ["decision_agent", "social_llm"],
)
def test_every_model_output_limit_must_be_explicit(tmp_path, section):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    header = f"[models.{section}]"
    start = text.index(header)
    line_start = text.index("max_output_tokens = ", start)
    line_end = text.index("\n", line_start) + 1
    path = tmp_path / f"missing-{section}-limit.toml"
    path.write_text(text[:line_start] + text[line_end:])

    with pytest.raises(
        ValueError,
        match=rf"models\.{section} must explicitly configure: max_output_tokens",
    ):
        load_experiment_config(path)

def test_unknown_model_task_is_rejected(tmp_path):
    path = tmp_path / "unknown-task.toml"
    path.write_text(
        (PROJECT_ROOT / "experiments/smoke.toml").read_text()
        + "\n[models.social_llm.tasks.unknown]\nmax_output_tokens = 10\n"
    )

    with pytest.raises(ValueError, match="unknown models.social_llm.tasks entry"):
        load_experiment_config(path)

def test_request_options_must_match_the_selected_api(tmp_path):
    path = tmp_path / "wrong-request-options.toml"
    path.write_text(
        (PROJECT_ROOT / "experiments/smoke.toml").read_text()
        + "\n[models.social_llm.request_options.thinking]\ntype = 'adaptive'\n"
    )

    with pytest.raises(ValueError, match="unknown models.social_llm.request_options setting"):
        load_experiment_config(path)

def test_anthropic_uses_native_thinking_options_and_rejects_reasoning_effort(tmp_path):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    section_start = text.index("[models.social_llm]")
    section_end = text.index("[models.social_llm.pricing", section_start)
    social_section = text[section_start:section_end]
    replacements = {
        "provider": 'provider = "anthropic"',
        "api_type": 'api_type = "anthropic_messages"',
        "api_key_env": 'api_key_env = "ANTHROPIC_API_KEY"',
        "api_key_required": "api_key_required = true",
    }
    for field, replacement in replacements.items():
        social_section = re.sub(
            rf"^{field}\s*=.*$", replacement, social_section,
            count=1, flags=re.MULTILINE,
        )
    anthropic_text = text[:section_start] + social_section + text[section_end:]
    native_path = tmp_path / "anthropic-native.toml"
    native_path.write_text(
        anthropic_text
        + "\n[models.social_llm.request_options.thinking]\ntype = 'adaptive'\n"
        + "\n[models.social_llm.request_options.output_config]\neffort = 'medium'\n"
    )

    config = load_experiment_config(native_path)
    assert config.social_llm.request_options == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }

    invalid_path = tmp_path / "anthropic-reasoning.toml"
    invalid_path.write_text(
        text[:section_start]
        + social_section
        + "reasoning_effort = 'high'\n"
        + text[section_end:]
    )
    with pytest.raises(ValueError, match="reasoning_effort is not supported"):
        load_experiment_config(invalid_path)
