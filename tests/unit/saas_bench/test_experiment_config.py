"""实验 TOML 配置的快速单元测试。"""

import re
from pathlib import Path

import pytest

from saas_bench.experiment.experiment_config import load_experiment_config
from tests.support.harness import TEST_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUTODL_DEEPSEEK_CONFIG = PROJECT_ROOT / "config/analysis_autodl_deepseek_14d.toml"
FORMAL_BASELINE_CONFIG = (
    PROJECT_ROOT / "config/baseline_autodl_deepseek_497d.toml"
)
FORMAL_ANALYSIS_CONFIG = (
    PROJECT_ROOT / "config/analysis_autodl_deepseek_497d.toml"
)


def test_experiment_config_loads_all_experiment_and_model_fields():
    config = load_experiment_config(TEST_CONFIG)

    assert config.experiment.name == "test"
    assert config.experiment.seed == 42
    assert config.experiment.days == 7
    assert config.experiment.initial_cash == pytest.approx(1_000_000)
    assert config.experiment.max_decision_turns_per_batch == 100
    assert config.experiment.max_invalid_responses_per_turn == 3
    assert config.modules.analysis.enabled is False
    assert config.modules.analysis.max_schema_retries == 1
    assert config.modules.analysis.max_enterprise_threads == 50
    assert config.modules.analysis.role_report_concurrency == 1
    assert config.analysis is None
    assert config.decision_agent.model == "test-decision-model"
    assert config.decision_agent.tool_choice == "required"
    assert config.decision_agent.reasoning_effort is None
    assert config.decision_agent.temperature == pytest.approx(0.7)
    assert config.decision_agent.pricing["test-decision-model"] == {
        "currency": "USD",
        "uncached_input_cost_per_million": 0.0,
        "cached_input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
    }
    assert config.social_llm.model == "test-social-model"
    assert config.social_llm.base_url == "http://localhost:11434/v1"
    assert config.social_llm.max_output_tokens == 1000

def test_analysis_model_is_required_only_when_module_is_enabled(tmp_path):
    baseline_text = TEST_CONFIG.read_text()
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
    text = TEST_CONFIG.read_text()

    missing_modules_path = tmp_path / "missing-modules.toml"
    missing_modules_path.write_text(
        re.sub(
            r"^\[modules\.analysis\]\n.*?(?=^\[models\.)",
            "",
            text,
            count=1,
            flags=re.MULTILINE | re.DOTALL,
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

    missing_concurrency_path = tmp_path / "missing-analysis-concurrency.toml"
    missing_concurrency_path.write_text(
        text.replace("role_report_concurrency = 1", "", 1)
    )
    with pytest.raises(
        ValueError,
        match="modules.analysis must explicitly configure: role_report_concurrency",
    ):
        load_experiment_config(missing_concurrency_path)

@pytest.mark.parametrize("value", ["1", '"true"'])
def test_analysis_enabled_must_be_boolean(tmp_path, value):
    text = TEST_CONFIG.read_text()
    path = tmp_path / "invalid-analysis-switch.toml"
    path.write_text(text.replace("enabled = false", f"enabled = {value}", 1))

    with pytest.raises(ValueError, match="modules.analysis.enabled must be a boolean"):
        load_experiment_config(path)

@pytest.mark.parametrize("value", ["-1", "true"])
def test_analysis_schema_retries_must_be_non_negative_integer(tmp_path, value):
    text = TEST_CONFIG.read_text()
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
    text = TEST_CONFIG.read_text()
    path = tmp_path / "invalid-analysis-thread-limit.toml"
    path.write_text(
        text.replace("max_enterprise_threads = 50", f"max_enterprise_threads = {value}", 1)
    )

    with pytest.raises(ValueError, match="max_enterprise_threads must be a positive integer"):
        load_experiment_config(path)


@pytest.mark.parametrize("value", ["0", "5", "true"])
def test_analysis_role_concurrency_must_be_between_one_and_four(tmp_path, value):
    text = TEST_CONFIG.read_text()
    path = tmp_path / "invalid-analysis-role-concurrency.toml"
    path.write_text(
        text.replace(
            "role_report_concurrency = 1",
            f"role_report_concurrency = {value}",
            1,
        )
    )

    with pytest.raises(ValueError, match="must be an integer between 1 and 4"):
        load_experiment_config(path)

def test_experiment_config_path_is_required():
    with pytest.raises(ValueError, match="explicit experiment config path is required"):
        load_experiment_config(None)


@pytest.mark.parametrize("name", ["Baseline", "analysis_full", "../baseline", ""])
def test_experiment_name_must_be_a_safe_directory_name(tmp_path, name):
    path = tmp_path / "invalid-name.toml"
    path.write_text(TEST_CONFIG.read_text().replace('name = "test"', f'name = "{name}"'))

    with pytest.raises(ValueError, match="experiment.name"):
        load_experiment_config(path)


def test_compatible_endpoint_loads_explicit_generation_parameters(tmp_path):
    path = tmp_path / "compatible.toml"
    path.write_text(
        """
[experiment]
name = "compatible"
max_decision_turns_per_batch = 100
max_invalid_responses_per_turn = 3

[modules.analysis]
enabled = false
max_schema_retries = 1
max_enterprise_threads = 50
role_report_concurrency = 1

[models.decision_agent]
api_type = "openai_chat_completions"
tool_choice = "required"
model = "channel-model"
base_url = "https://www.autodl.art/api/v1"
api_key_env = "AUTODL_API_KEY"
reasoning_effort = "low"
max_output_tokens = 4096

[models.decision_agent.request_options.extra_body]
do_sample = false
stream = false
thinking = { type = "enabled", clear_thinking = true }
response_format = { type = "text" }

[models.decision_agent.pricing.channel-model]
currency = "CNY"
uncached_input_cost_per_million = 8.0
cached_input_cost_per_million = 2.0
output_cost_per_million = 28.0

[models.social_llm]
api_type = "openai_chat_completions"
model = "social-test"
base_url = "http://localhost:11434/v1"
api_key_required = false
max_output_tokens = 1000

[models.social_llm.pricing.social-test]
currency = "CNY"
uncached_input_cost_per_million = 0.0
cached_input_cost_per_million = 0.0
output_cost_per_million = 0.0
"""
    )

    config = load_experiment_config(path)

    assert config.decision_agent.model == "channel-model"
    assert config.decision_agent.reasoning_effort == "low"
    assert config.decision_agent.base_url == "https://www.autodl.art/api/v1"
    assert config.decision_agent.request_options["extra_body"]["thinking"] == {
        "type": "enabled",
        "clear_thinking": True,
    }


def test_autodl_deepseek_config_uses_verified_model_names_and_official_prices():
    config = load_experiment_config(AUTODL_DEEPSEEK_CONFIG)

    decision = config.decision_agent
    assert decision.model == "DeepSeek-V4-Pro"
    assert decision.pricing_model_map == {
        "DeepSeek-V4-Pro": "deepseek-v4-pro",
        "deepseek-v4-pro-0813": "deepseek-v4-pro",
    }
    assert decision.pricing["deepseek-v4-pro"] == {
        "currency": "USD",
        "uncached_input_cost_per_million": pytest.approx(1.32),
        "cached_input_cost_per_million": pytest.approx(0.044),
        "output_cost_per_million": pytest.approx(3.96),
    }

    for model in (config.social_llm, config.analysis):
        assert model is not None
        assert model.model == "DeepSeek-V4-Flash"
        assert model.pricing_model_map == {
            "DeepSeek-V4-Flash": "deepseek-v4-flash",
            "deepseek-v4-flash-0731": "deepseek-v4-flash",
        }
        assert model.pricing["deepseek-v4-flash"] == {
            "currency": "USD",
            "uncached_input_cost_per_million": pytest.approx(0.44),
            "cached_input_cost_per_million": pytest.approx(0.014),
            "output_cost_per_million": pytest.approx(1.32),
        }


def test_formal_baseline_and_analysis_configs_only_differ_by_ablation():
    baseline = load_experiment_config(FORMAL_BASELINE_CONFIG)
    analysis = load_experiment_config(FORMAL_ANALYSIS_CONFIG)

    assert baseline.experiment.name == "baseline"
    assert analysis.experiment.name == "analysis"
    assert {
        **baseline.experiment.__dict__,
        "name": "paired",
    } == {
        **analysis.experiment.__dict__,
        "name": "paired",
    }
    assert baseline.decision_agent == analysis.decision_agent
    assert baseline.social_llm == analysis.social_llm
    assert baseline.modules.analysis.__dict__ == {
        **analysis.modules.analysis.__dict__,
        "enabled": False,
    }
    assert baseline.analysis is None
    assert analysis.analysis is not None

@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        (
            "[experiment]\nname = 'baseline'\nseed = 42\nmax_decision_turns_per_batch = 100\nmax_invalid_responses_per_turn = 3\n",
            "models must be explicitly configured",
        ),
        (
            "[experiment]\nname = 'baseline'\nmax_decision_turns_per_batch = 100\nmax_invalid_responses_per_turn = 3\n"
            "[models.decision_agent]\napi_type = 'openai_responses'\ntool_choice = 'required'\nmodel = 'decision'\nmax_output_tokens = 100\napi_key_required = false\n[models.decision_agent.pricing.decision]\ncurrency = 'USD'\nuncached_input_cost_per_million = 0\ncached_input_cost_per_million = 0\noutput_cost_per_million = 0\n",
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
    text = TEST_CONFIG.read_text()
    path.write_text(
        re.sub(r"^max_decision_turns_per_batch\s*=.*\n", "", text, count=1, flags=re.MULTILINE)
    )

    with pytest.raises(
        ValueError, match="max_decision_turns_per_batch"
    ):
        load_experiment_config(path)


def test_decision_tool_choice_must_be_explicit_and_valid(tmp_path):
    text = TEST_CONFIG.read_text()
    missing_path = tmp_path / "missing-tool-choice.toml"
    missing_path.write_text(
        re.sub(r"^tool_choice\s*=.*\n", "", text, count=1, flags=re.MULTILINE)
    )

    with pytest.raises(
        ValueError,
        match=r"models\.decision_agent must explicitly configure: tool_choice",
    ):
        load_experiment_config(missing_path)

    invalid_path = tmp_path / "invalid-tool-choice.toml"
    invalid_path.write_text(text.replace('tool_choice = "required"', 'tool_choice = "always"', 1))

    with pytest.raises(
        ValueError,
        match=r"models\.decision_agent\.tool_choice must be one of: auto, required",
    ):
        load_experiment_config(invalid_path)

def test_invalid_response_limit_must_be_explicit(tmp_path):
    path = tmp_path / "missing-invalid-response-limit.toml"
    text = TEST_CONFIG.read_text()
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
model = "decision"
[models.social_llm]
model = "social"
"""
    )

    with pytest.raises(ValueError, match="unknown experiment setting"):
        load_experiment_config(path)

def test_model_costs_must_be_configured_as_a_pair(tmp_path):
    text = TEST_CONFIG.read_text()
    path = tmp_path / "partial-cost.toml"
    path.write_text(
        re.sub(r"^output_cost_per_million\s*=.*\n", "", text, count=1, flags=re.MULTILINE)
    )

    with pytest.raises(ValueError, match="explicitly configure: output_cost_per_million"):
        load_experiment_config(path)

def test_pricing_model_map_must_target_configured_official_price(tmp_path):
    text = TEST_CONFIG.read_text()
    pricing_header = '[models.decision_agent.pricing."test-decision-model"]'
    path = tmp_path / "invalid-pricing-map.toml"
    path.write_text(text.replace(
        pricing_header,
        '[models.decision_agent.pricing_model_map]\n'
        '"test-decision-model" = "missing-official-model"\n\n'
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
    text = TEST_CONFIG.read_text()
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
        TEST_CONFIG.read_text()
        + "\n[models.social_llm.tasks.unknown]\nmax_output_tokens = 10\n"
    )

    with pytest.raises(ValueError, match="unknown models.social_llm.tasks entry"):
        load_experiment_config(path)

def test_request_options_must_match_the_selected_api(tmp_path):
    path = tmp_path / "wrong-request-options.toml"
    path.write_text(
        TEST_CONFIG.read_text()
        + "\n[models.social_llm.request_options.thinking]\ntype = 'adaptive'\n"
    )

    with pytest.raises(ValueError, match="unknown models.social_llm.request_options setting"):
        load_experiment_config(path)
