import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent, Message
from saas_bench.agents.bash_agent.tools import (
    BashAgentToolExecutor,
    NextWeekExecutionError,
)
from saas_bench.agents.bash_agent import run_test
from saas_bench.agents.bash_agent.run_test import BashAgentRunner, _resume_runner
from saas_bench.config import BenchmarkConfig
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.database import add_api_cost, get_api_usage_summary
from saas_bench.experiment_config import load_experiment_config
from saas_bench.api_server import NovaMindAPIServer, _APIHandler
from saas_bench.event_logger import EventLogger
from saas_bench.server_entry import (
    _apply_simulator_llm_config,
    _apply_simulator_llm_env_overrides,
    _restore_simulator_llm_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMPTY_ENVIRONMENT_LLM_USAGE = {
    "input_tokens": 0,
    "cached_tokens": 0,
    "output_tokens": 0,
    "cost_by_currency": {},
    "by_purpose": {},
}


class RecordingResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="response",
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


class RecordingOpenAI:
    def __init__(self):
        self.responses = RecordingResponses()


def test_experiment_config_loads_all_experiment_and_model_fields():
    config = load_experiment_config(PROJECT_ROOT / "experiments/experiment.toml")

    assert config.experiment.seed == 42
    assert config.experiment.days == 70
    assert config.experiment.initial_cash == pytest.approx(1_000_000)
    assert config.experiment.max_decision_turns_per_batch == 100
    assert config.experiment.max_invalid_responses_per_turn == 3
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


def test_experiment_config_path_is_required():
    with pytest.raises(ValueError, match="explicit experiment config path is required"):
        load_experiment_config(None)


def test_runner_rejects_missing_decision_model_identity():
    with pytest.raises(ValueError, match="model must be explicitly configured"):
        BashAgentRunner(
            model=None,
            provider="openai",
            api_type="openai_responses",
            max_output_tokens=100,
            max_decision_turns_per_batch=100,
        )
    with pytest.raises(ValueError, match="provider must be explicitly configured"):
        BashAgentRunner(
            model="decision",
            provider=None,
            api_type="openai_responses",
            max_output_tokens=100,
            max_decision_turns_per_batch=100,
        )


def test_runner_rejects_missing_decision_request_limit():
    with pytest.raises(ValueError, match="max_output_tokens must be configured"):
        BashAgentRunner(
            model="decision",
            provider="openai",
            api_type="openai_responses",
            max_decision_turns_per_batch=100,
            max_invalid_responses_per_turn=3,
        )


def test_agent_rejects_missing_model_identity():
    with pytest.raises(ValueError, match="agent model must be explicitly configured"):
        BashAgent(tool_descriptions=[], client=object(), api_type="openai_responses")


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


def test_short_experiment_is_allowed_and_runner_rounds_to_zero(tmp_path):
    text = (PROJECT_ROOT / "experiments/smoke.toml").read_text()
    path = tmp_path / "short.toml"
    path.write_text(text.replace("days = 7", "days = 6", 1))

    config = load_experiment_config(path)
    runner = BashAgentRunner(
        model=config.decision_agent.model,
        provider=config.decision_agent.provider,
        api_type=config.decision_agent.api_type,
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


def test_resume_loads_the_saved_configuration_without_external_overrides(tmp_path):
    run_dir = tmp_path / "run_existing"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "format_version": 4,
        "run_id": "existing",
        "agent_type": "bash_agent",
        "model": "original-model",
        "provider": "openai_compatible",
        "api_type": "openai_responses",
        "base_url": "http://localhost:11434/v1",
        "reasoning_effort": None,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_output_tokens": 100,
        "timeout_seconds": 30.0,
        "pricing": {"original-model": {
            "currency": "USD",
            "uncached_input_cost_per_million": 1.0,
            "cached_input_cost_per_million": 0.1,
            "output_cost_per_million": 2.0,
        }},
        "request_options": {},
        "api_key_env": None,
        "api_key_required": False,
        "seed": 42,
        "scenario": "default",
        "total_days": 70,
        "initial_cash": 1_000_000.0,
        "max_decision_turns_per_batch": 100,
        "max_invalid_responses_per_turn": 3,
        "label": "saved",
        "simulator_llm": {},
        "public_bundle_sha256": "0" * 64,
        "harness_git_commit": "test-commit",
        "harness_git_dirty": False,
        "harness_source_sha256": "1" * 64,
    }))

    runner = _resume_runner(str(run_dir))

    assert runner.model == "original-model"
    assert runner.api_type == "openai_responses"
    assert runner.temperature == pytest.approx(0.7)
    assert runner.pricing["original-model"]["uncached_input_cost_per_million"] == pytest.approx(1.0)
    assert runner.workspace_dir == run_dir.resolve()


def test_resume_warns_when_current_harness_differs(tmp_path, capsys):
    run_dir = tmp_path / "run_existing"
    run_dir.mkdir()
    source = run_test._new_experiment_runner(
        PROJECT_ROOT / "experiments/smoke.toml"
    )
    config = source._run_config_payload()
    config["harness_source_sha256"] = "0" * 64
    (run_dir / "config.json").write_text(json.dumps(config))

    resumed = _resume_runner(str(run_dir))

    assert resumed.harness_source_sha256 != config["harness_source_sha256"]
    assert "Current Harness source differs" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.pop("pricing"),
        lambda config: config.update({"api_server_port": 12345}),
    ],
)
def test_resume_rejects_run_config_with_noncanonical_fields(tmp_path, mutation):
    run_dir = tmp_path / "run_existing"
    run_dir.mkdir()
    source = run_test._new_experiment_runner(
        PROJECT_ROOT / "experiments/smoke.toml"
    )
    config = source._run_config_payload()
    mutation(config)
    (run_dir / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="fields do not match"):
        run_test._load_saved_run_config(run_dir)


def test_resume_setup_never_rewrites_run_config(tmp_path, monkeypatch):
    runner = run_test._new_experiment_runner(
        PROJECT_ROOT / "experiments/smoke.toml"
    )
    run_dir = tmp_path / f"run_{runner.run_id}"
    run_dir.mkdir()
    config_file = run_dir / "config.json"
    config_file.write_text(json.dumps(runner._run_config_payload(), indent=4))
    original_bytes = config_file.read_bytes()
    resumed = _resume_runner(str(run_dir))
    resumed._load_checkpoint = lambda: {"session_id": "session-1"}
    resumed._launch_server_from_prepared_checkpoint = lambda: (
        _ for _ in ()
    ).throw(ValueError("stop after config check"))
    monkeypatch.setattr(
        "saas_bench.agents.bash_agent.tools.get_bash_agent_tool_descriptions",
        lambda: [],
    )

    with pytest.raises(ValueError, match="stop after config check"):
        resumed.setup()

    assert config_file.read_bytes() == original_bytes


def test_resume_restores_database_and_metadata_before_server_launch(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path / "run_existing"
    runner.agent_workspace = runner.workspace_dir / "agent_workspace"
    runner._session_id = None

    session_dir = runner.agent_workspace / "sessions" / "session-1"
    session_dir.mkdir(parents=True)
    checkpoint_db = runner.workspace_dir / ".checkpoint_dbs" / "world_day_35.nmdb"
    checkpoint_db.parent.mkdir()
    checkpoint_db.write_bytes(b"checkpoint-state")
    runner._git_init_workspace()
    runner._git_commit_workspace("checkpoint workspace")
    workspace_commit = runner._git("rev-parse", "HEAD", check=True).stdout.strip()
    runner._resume_checkpoint = {
        "day": 35,
        "session_id": "session-1",
        "database": {
            "file": ".checkpoint_dbs/world_day_35.nmdb",
            "sha256": runner._sha256_file(checkpoint_db),
        },
        "runtime": {
            "workspace_commit": workspace_commit,
            "runner_log_offsets": {},
            "server_log_offsets": {"history": 0, "event_log": 0},
        },
    }
    (session_dir / "world.nmdb").write_bytes(b"newer-uncommitted-state")
    (session_dir / "session.json").write_text(json.dumps({
        "current_day": 42,
        "status": "running",
        "port": 12345,
        "pid": 67890,
    }))

    observed = {}

    def fake_launch_server():
        observed["database"] = (session_dir / "world.nmdb").read_bytes()
        observed["metadata"] = json.loads((session_dir / "session.json").read_text())

    runner._launch_server = fake_launch_server
    runner._http_get = lambda path: {"day": 35}
    runner._preflight_checkpoint_restore = lambda checkpoint: SimpleNamespace(
        session_id="session-1", conversation_payload={}
    )
    runner._restore_logs_to_offsets = lambda offsets: None
    runner._launch_server_from_prepared_checkpoint()

    assert observed["database"] == b"checkpoint-state"
    assert observed["metadata"]["current_day"] == 35
    assert observed["metadata"]["status"] == "created"
    assert "port" not in observed["metadata"]
    assert "pid" not in observed["metadata"]


def test_resume_rejects_checkpoint_database_hash_mismatch(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path / "run_existing"
    runner.agent_workspace = runner.workspace_dir / "agent_workspace"
    runner._session_id = None

    checkpoint_db = runner.workspace_dir / ".checkpoint_dbs" / "world_day_35.nmdb"
    checkpoint_db.parent.mkdir(parents=True)
    checkpoint_db.write_bytes(b"tampered-state")
    session_dir = runner.agent_workspace / "sessions" / "session-1"
    session_dir.mkdir(parents=True)

    checkpoint = {
        "day": 35,
        "session_id": "session-1",
        "database": {
            "file": ".checkpoint_dbs/world_day_35.nmdb",
            "sha256": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        runner._restore_checkpoint_database_before_server(checkpoint)


def test_resume_stops_server_when_restored_day_does_not_match(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner._resume_checkpoint = {
        "day": 35,
        "runtime": {
            "workspace_commit": "checkpoint-commit",
            "runner_log_offsets": {},
            "server_log_offsets": {"history": 0, "event_log": 0},
        },
    }
    calls = []
    runner._preflight_checkpoint_restore = lambda checkpoint: (
        calls.append("preflight") or SimpleNamespace(
            session_id="session-1", conversation_payload={}
        )
    )
    runner._restore_workspace_commit = lambda commit: calls.append("workspace")
    runner._refresh_public_workspace_artifacts = lambda: calls.append("refresh")
    runner._restore_checkpoint_database_before_server = lambda checkpoint: calls.append("restore")
    runner._restore_logs_to_offsets = lambda offsets: calls.append("runner-logs")
    runner._restore_server_logs_before_server = lambda offsets: calls.append("logs")
    runner._launch_server = lambda: calls.append("launch")
    runner._http_get = lambda path: {"day": 28}
    runner._stop_server = lambda: calls.append("stop")

    with pytest.raises(RuntimeError, match="does not match checkpoint day 35"):
        runner._launch_server_from_prepared_checkpoint()

    assert calls == [
        "preflight", "workspace", "refresh", "restore",
        "runner-logs", "logs", "launch", "stop",
    ]


def _checkpoint_runner(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path / "run_test"
    runner.workspace_dir.mkdir()
    (runner.workspace_dir / "config.json").write_text(
        json.dumps({"test_config": True})
    )
    runner.agent_workspace = runner.workspace_dir / "agent_workspace"
    runner.agent_workspace.mkdir()
    runner._git_init_workspace()
    runner._session_id = "session-1"
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    session_dir.mkdir(parents=True)
    (session_dir / "world.nmdb").write_bytes(b"persisted-database")
    runner.run_id = "test"
    runner.logs_dir = runner.workspace_dir / "logs"
    runner.logs_dir.mkdir()
    runner.response_log_file = runner.logs_dir / "raw_responses_test.jsonl"
    runner.timing_log_file = runner.logs_dir / "timing_test.jsonl"
    runner.model = "model"
    runner.provider = "openai"
    runner.api_type = "openai_responses"
    runner.base_url = None
    runner.reasoning_effort = None
    runner.seed = 42
    runner.scenario = "default"
    runner.agent = None
    runner.total_decision_agent_cost_by_currency = {}
    return runner


def test_checkpoint_json_references_the_exact_hashed_database(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }

    runner._save_checkpoint(7)

    checkpoint = json.loads((runner.workspace_dir / "checkpoint.json").read_text())
    checkpoint_db = runner.workspace_dir / checkpoint["database"]["file"]
    assert checkpoint["format_version"] == runner.CHECKPOINT_FORMAT_VERSION
    assert checkpoint["run_config_sha256"] == runner._sha256_file(
        runner.workspace_dir / "config.json"
    )
    assert checkpoint["day"] == 7
    assert checkpoint["cash"] == pytest.approx(900_000.0)
    assert checkpoint_db.read_bytes() == b"persisted-database"
    assert checkpoint["database"]["sha256"] == runner._sha256_file(checkpoint_db)
    assert (runner.workspace_dir / "world.nmdb").read_bytes() == b"persisted-database"
    runtime = checkpoint["runtime"]
    conversation = runner.workspace_dir / runtime["conversation"]["file"]
    assert runtime["conversation"]["sha256"] == runner._sha256_file(conversation)
    assert runner._git("rev-parse", "HEAD", check=True).stdout.strip() == runtime["workspace_commit"]
    assert runtime["runner_log_offsets"] == {
        "tool_results": 0,
        "raw_responses": 0,
        "timing": 0,
    }
    assert runtime["server_log_offsets"] == {"history": 0, "event_log": 0}
    assert runtime["environment_llm"] == EMPTY_ENVIRONMENT_LLM_USAGE


def test_environment_llm_usage_is_summarized_by_purpose():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE api_costs (
            day INTEGER NOT NULL,
            model TEXT NOT NULL,
            purpose TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cost_amount REAL NOT NULL,
            currency TEXT NOT NULL
        )
    """)
    add_api_cost(conn, 7, "social", "customer_social_post", 11, 4, 7, 0.01, "CNY")
    add_api_cost(conn, 14, "social", "customer_social_post", 13, 5, 5, 0.02, "CNY")
    add_api_cost(conn, 14, "social", "macro_social_post", 17, 6, 3, 0.03, "CNY")

    assert get_api_usage_summary(conn) == {
        "input_tokens": 41,
        "cached_tokens": 15,
        "output_tokens": 15,
        "cost_by_currency": {"CNY": pytest.approx(0.06)},
        "by_purpose": {
            "customer_social_post": {
                "input_tokens": 24,
                "cached_tokens": 9,
                "output_tokens": 12,
                "cost_by_currency": {"CNY": pytest.approx(0.03)},
            },
            "macro_social_post": {
                "input_tokens": 17,
                "cached_tokens": 6,
                "output_tokens": 3,
                "cost_by_currency": {"CNY": pytest.approx(0.03)},
            },
        },
    }


def test_environment_llm_usage_rejects_inconsistent_totals():
    usage = {
        "input_tokens": 2,
        "cached_tokens": 0,
        "output_tokens": 1,
        "cost_by_currency": {"CNY": 0.01},
        "by_purpose": {
            "customer_social_post": {
                "input_tokens": 1,
                "cached_tokens": 0,
                "output_tokens": 1,
                "cost_by_currency": {"CNY": 0.01},
            }
        },
    }

    with pytest.raises(ValueError, match="input token total"):
        BashAgentRunner._validate_environment_llm_usage(usage)


def test_result_includes_environment_llm_usage_from_checkpoint(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.run_id = "test"
    runner.seed = 42
    runner.scenario = "default"
    runner.workspace_dir = tmp_path
    runner._harness_result_fields = lambda: {}
    environment_usage = {
        "input_tokens": 41,
        "cached_tokens": 15,
        "output_tokens": 15,
        "cost_by_currency": {"CNY": 0.06},
        "by_purpose": {
            "customer_social_post": {
                "input_tokens": 24,
                "cached_tokens": 9,
                "output_tokens": 12,
                "cost_by_currency": {"CNY": 0.03},
            },
            "macro_social_post": {
                "input_tokens": 17,
                "cached_tokens": 6,
                "output_tokens": 3,
                "cost_by_currency": {"CNY": 0.03},
            },
        },
    }
    checkpoint = {
        "day": 14,
        "cash": 900_000.0,
        "runtime": {
            "agent": {
                "total_turns": 3,
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_tokens": 10,
                "reasoning_tokens": 5,
                "decision_cost_by_currency": {"CNY": 0.1},
            },
            "environment_llm": environment_usage,
        },
    }

    result = runner._result_from_checkpoint(checkpoint, "completed")

    assert result["environment_llm_input_tokens"] == 41
    assert result["environment_llm_output_tokens"] == 15
    assert result["environment_llm_cached_tokens"] == 15
    assert result["environment_llm_cost_by_currency"] == {"CNY": pytest.approx(0.06)}
    assert result["environment_llm_usage_by_purpose"] == environment_usage["by_purpose"]


def test_checkpoint_load_rejects_tampered_run_config(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    (runner.workspace_dir / "config.json").write_text(
        json.dumps({"test_config": False})
    )

    with pytest.raises(ValueError, match="run config hash mismatch"):
        runner._load_checkpoint()


def test_failed_database_persistence_keeps_previous_checkpoint(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    checkpoint_file = runner.workspace_dir / "checkpoint.json"
    checkpoint_file.write_text('{"day": 0}')
    runner._http_post = lambda path, data, timeout: {
        "success": False,
        "error": "week_advance_not_stable",
    }

    with pytest.raises(RuntimeError, match="persistence failed"):
        runner._save_checkpoint(7)

    assert checkpoint_file.read_text() == '{"day": 0}'
    assert not (runner.workspace_dir / ".checkpoint_dbs").exists()


def test_runtime_snapshot_failure_keeps_previous_checkpoint_artifacts(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    old_db = runner.workspace_dir / ".checkpoint_dbs" / "old.nmdb"
    old_runtime = runner.workspace_dir / ".checkpoint_runtime" / "conversation_old.json"
    old_db.parent.mkdir()
    old_runtime.parent.mkdir()
    old_db.write_bytes(b"old-database")
    old_runtime.write_text("old-conversation")
    checkpoint_file = runner.workspace_dir / "checkpoint.json"
    checkpoint_file.write_text(json.dumps({
        "day": 0,
        "database_file": ".checkpoint_dbs/old.nmdb",
        "runtime": {"conversation_file": ".checkpoint_runtime/conversation_old.json"},
    }))
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._capture_workspace_commit = lambda day: (_ for _ in ()).throw(
        RuntimeError("git failed")
    )

    with pytest.raises(RuntimeError, match="git failed"):
        runner._save_checkpoint(7)

    assert json.loads(checkpoint_file.read_text())["day"] == 0
    assert old_db.read_bytes() == b"old-database"
    assert old_runtime.read_text() == "old-conversation"


def test_restore_preflight_rejects_tampered_conversation_before_mutation(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": runner._session_id,
        "current_day": 0,
        "status": "created",
    }))
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()
    conversation_path = (
        runner.workspace_dir / checkpoint["runtime"]["conversation"]["file"]
    )
    conversation_path.write_text("tampered")

    tracked = runner.agent_workspace / "MEMORY.md"
    tracked.write_text("future workspace state")
    session_db = session_dir / "world.nmdb"
    session_db.write_bytes(b"future database state")
    runner._resume_checkpoint = checkpoint

    with pytest.raises(ValueError, match="conversation hash mismatch"):
        runner._launch_server_from_prepared_checkpoint()

    assert tracked.read_text() == "future workspace state"
    assert session_db.read_bytes() == b"future database state"


def test_restore_preflight_rejects_invalid_conversation_schema_with_valid_hash(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": runner._session_id,
        "current_day": 0,
        "status": "created",
    }))
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()
    conversation_path = (
        runner.workspace_dir / checkpoint["runtime"]["conversation"]["file"]
    )
    payload = json.loads(conversation_path.read_text())
    payload.pop("tool_results_applied")
    conversation_path.write_text(json.dumps(payload))
    checkpoint["runtime"]["conversation"]["sha256"] = runner._sha256_file(
        conversation_path
    )

    with pytest.raises(ValueError, match="conversation fields"):
        runner._preflight_checkpoint_restore(checkpoint)


def test_restore_preflight_rejects_conversation_turn_count_mismatch(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": runner._session_id,
        "current_day": 0,
        "status": "created",
    }))
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 900_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()
    checkpoint["runtime"]["agent"]["total_turns"] = 1

    with pytest.raises(ValueError, match="total_turns mismatch"):
        runner._preflight_checkpoint_restore(checkpoint)


def test_api_rejects_checkpoint_while_week_state_is_unstable():
    calls = []
    server = NovaMindAPIServer(
        tools=SimpleNamespace(current_day=7),
        checkpoint_persist_callback=lambda day, fresh: calls.append((day, fresh)),
    )
    server._week_advance_in_progress = True

    result = server.persist_checkpoint(7)

    assert result["success"] is False
    assert result["error"] == "week_advance_not_stable"
    assert calls == []


def test_api_requests_fresh_snapshot_after_state_revision_changes():
    calls = []
    server = NovaMindAPIServer(
        tools=SimpleNamespace(current_day=7),
        checkpoint_persist_callback=lambda day, fresh: (
            calls.append((day, fresh)) or {
                "persisted_day": day,
                "server_log_offsets": {"history": 12, "event_log": 34},
            }
        ),
    )
    server._state_revision = 2
    server._checkpoint_snapshot_revision = 1

    result = server.persist_checkpoint(7)

    assert result == {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 0.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 12, "event_log": 34},
    }
    assert calls == [(7, True)]


def test_api_finalizes_completed_run_once_at_target_day():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE ledger (amount REAL NOT NULL)")
    conn.execute("INSERT INTO ledger VALUES (125.0)")
    finalized = []
    server = NovaMindAPIServer(
        tools=SimpleNamespace(
            current_day=14,
            config=SimpleNamespace(total_days=14),
        ),
        conn=conn,
        run_finalize_callback=lambda outcome, day, cash: finalized.append(
            (outcome, day, cash)
        ),
    )

    first = server.finalize_run("completed")
    second = server.finalize_run("completed")

    assert first == {
        "success": True,
        "outcome": "completed",
        "day": 14,
        "final_cash": 125.0,
        "already_finalized": False,
    }
    assert second["success"] is True
    assert second["already_finalized"] is True
    assert finalized == [("completed", 14, 125.0)]


def test_api_rejects_invalid_terminal_outcomes():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE ledger (amount REAL NOT NULL)")
    conn.execute("INSERT INTO ledger VALUES (100.0)")
    server = NovaMindAPIServer(
        tools=SimpleNamespace(
            current_day=7,
            config=SimpleNamespace(total_days=14),
        ),
        conn=conn,
        run_finalize_callback=lambda outcome, day, cash: None,
    )

    assert server.finalize_run("timeout")["error"] == "invalid_terminal_outcome"
    assert server.finalize_run("completed")["error"] == "completion_before_target_day"
    assert server.finalize_run("bankrupt")["error"] == "bankruptcy_without_negative_cash"

    server.tools.current_day = 14
    server.conn.execute("UPDATE ledger SET amount = -1")
    assert server.finalize_run("completed")["error"] == "completion_with_negative_cash"


def test_week_advance_exception_releases_stability_guard():
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0))
    server._advance_week_impl = lambda predictions, rationale: (_ for _ in ()).throw(
        RuntimeError("failed")
    )

    with pytest.raises(RuntimeError, match="failed"):
        server.advance_week()

    assert server._week_advance_in_progress is False
    assert server._week_advance_failed is True
    assert server.advance_week()["error"] == "week_advance_failed"
    blocked = server.execute_tool("unknown", {})
    assert blocked.success is False
    assert blocked.message == "week_advance_failed"


def test_week_advance_timeout_keeps_stability_guard_enabled():
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0))
    server._advance_week_impl = lambda predictions, rationale: {
        "success": False,
        "error": "step_week_timeout",
    }

    result = server.advance_week()

    assert result["error"] == "step_week_timeout"
    assert server._week_advance_in_progress is True


def test_week_advance_in_progress_blocks_concurrent_tools():
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0))
    server._week_advance_in_progress = True

    blocked = server.execute_tool("unknown", {})

    assert blocked.success is False
    assert blocked.message == "week_advance_in_progress"


def test_next_week_rejects_non_finite_predictions_before_advancing():
    advance_calls = []
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(
        _api_server=SimpleNamespace(
            advance_week=lambda **kwargs: advance_calls.append(kwargs)
        )
    )
    handler._read_body = lambda: {
        "rationale": "test",
        "predictions": {
            "cash_1wk": {"point": float("nan"), "lower": 0, "upper": 1},
            "cash_4wk": {"point": 1, "lower": 0, "upper": 2},
            "cash_12wk": {"point": 1, "lower": 0, "upper": 2},
            "cash_26wk": {"point": 1, "lower": 0, "upper": 2},
        },
    }
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_next_week()

    assert responses[0][1] == 400
    assert "finite numbers" in responses[0][0]["error"]
    assert advance_calls == []


def test_prediction_persistence_failure_prevents_world_advance(monkeypatch):
    step_calls = []
    server = NovaMindAPIServer(
        tools=SimpleNamespace(current_day=0),
        simulator=SimpleNamespace(step_week=lambda: step_calls.append(True)),
        conn=sqlite3.connect(":memory:"),
    )
    monkeypatch.setattr(
        "saas_bench.database.save_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )

    with pytest.raises(RuntimeError, match="write failed"):
        server.advance_week(predictions={7: {"cash": 1.0}})

    assert step_calls == []
    assert server._week_advance_failed is True


def test_main_starts_new_run_from_the_fixed_toml_without_cli_configuration(monkeypatch):
    calls = []
    assert run_test.DEFAULT_EXPERIMENT_CONFIG == PROJECT_ROOT / "experiments/experiment.toml"
    assert run_test.DEFAULT_EXPERIMENT_CONFIG.is_file()

    class Runner:
        def run(self, verbose):
            assert verbose is True
            return {
                "outcome": "completed",
                "final_cash": 1_000_000.0,
                "workspace_dir": "/tmp/run",
            }

    monkeypatch.setattr(
        run_test,
        "_new_experiment_runner",
        lambda path: calls.append(path) or Runner(),
    )
    monkeypatch.setattr(
        run_test,
        "_resume_runner",
        lambda value: pytest.fail("resume path should not be used"),
    )

    run_test.main([])

    assert calls == [run_test.DEFAULT_EXPERIMENT_CONFIG]


def test_main_resume_uses_only_saved_run_identity(monkeypatch):
    calls = []

    class Runner:
        def run(self, verbose):
            return {
                "outcome": "completed",
                "final_cash": 1_000_000.0,
                "workspace_dir": "/tmp/run",
            }

    monkeypatch.setattr(
        run_test,
        "_new_experiment_runner",
        lambda path: pytest.fail("current TOML should not be read during resume"),
    )
    monkeypatch.setattr(
        run_test,
        "_resume_runner",
        lambda value: calls.append(value) or Runner(),
    )

    run_test.main(["--resume", "existing"])

    assert calls == ["existing"]


@pytest.mark.parametrize(
    "args",
    [
        ["--config", "experiments/smoke.toml"],
        ["--model", "other-model"],
        ["--days", "7"],
        ["--temperature", "0.1"],
        ["--resume", "existing", "--config", "experiments/full.toml"],
    ],
)
def test_main_rejects_cli_configuration_overrides(args):
    with pytest.raises(SystemExit):
        run_test.main(args)


def test_simulator_settings_survive_environment_and_session_round_trip(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_KEY", "test-key")
    overrides = {
        "social_post_llm_provider": "openai",
        "social_post_llm_api_type": "openai_responses",
        "social_post_llm_model": "social-test",
        "social_post_llm_base_url": "http://social.test/v1",
        "social_post_llm_api_key_env": "LOCAL_LLM_KEY",
        "social_post_llm_reasoning_effort": "none",
        "social_post_llm_temperature": 0.31,
        "social_post_llm_top_p": 0.72,
        "social_post_llm_max_tokens": 123,
        "social_post_llm_timeout_seconds": 45.0,
        "social_post_llm_pricing": {"social-test": {
            "currency": "USD",
            "uncached_input_cost_per_million": 0.0,
            "cached_input_cost_per_million": 0.0,
            "output_cost_per_million": 0.0,
        }},
    }
    monkeypatch.setenv("CEOBENCH_SIMULATOR_LLM_CONFIG", json.dumps(overrides))

    created = BenchmarkConfig()
    _apply_simulator_llm_env_overrides(created)
    session_values = _apply_simulator_llm_config(created)
    restored = BenchmarkConfig()
    _restore_simulator_llm_config(restored, {"simulator_llm": session_values})

    for field, expected in overrides.items():
        assert getattr(restored, field) == expected


def test_social_openai_calls_receive_configured_api_parameters():
    social = RecordingOpenAI()
    config = BenchmarkConfig(
        social_post_llm_provider="openai",
        social_post_llm_api_type="openai_responses",
        social_post_llm_model="social-test",
        social_post_llm_reasoning_effort="none",
        social_post_llm_temperature=0.31,
        social_post_llm_top_p=0.72,
        social_post_llm_max_tokens=123,
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=config,
        social_client=social,
    )

    simulator.create_social_response("social system", "social user")

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


def test_simulator_text_model_rejects_empty_success_response():
    client = RecordingOpenAI()
    client.responses.create = lambda **kwargs: SimpleNamespace(
        output_text="",
        model="social-test",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=BenchmarkConfig(
            social_post_llm_provider="openai_compatible",
            social_post_llm_api_type="openai_responses",
            social_post_llm_model="social-test",
            social_post_llm_max_tokens=100,
        ),
        social_client=client,
    )

    with pytest.raises(RuntimeError, match="returned an empty response"):
        simulator.create_social_response("system", "user", task="customer_post")


def test_task_request_options_merge_without_dropping_model_options():
    social = RecordingOpenAI()
    config = BenchmarkConfig(
        social_post_llm_provider="openai_compatible",
        social_post_llm_api_type="openai_responses",
        social_post_llm_model="social-test",
        social_post_llm_max_tokens=100,
        social_post_llm_request_options={
            "extra_body": {"enable_thinking": False, "priority": "normal"},
            "extra_headers": {"x-base": "base"},
        },
        social_post_llm_task_parameters={
            "customer_post": {
                "max_output_tokens": 50,
                "request_options": {
                    "extra_body": {"priority": "high"},
                    "extra_query": {"trace": "1"},
                },
            }
        },
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=config,
        social_client=social,
    )

    simulator.create_social_response("system", "user", task="customer_post")

    assert social.responses.calls[0]["max_output_tokens"] == 50
    assert social.responses.calls[0]["extra_body"] == {
        "enable_thinking": False,
        "priority": "high",
    }
    assert social.responses.calls[0]["extra_headers"] == {"x-base": "base"}
    assert social.responses.calls[0]["extra_query"] == {"trace": "1"}


def test_customer_social_post_preserves_served_model_and_logs_cost(
    make_initialized_sim,
):
    social = RecordingOpenAI()
    config = BenchmarkConfig(
        social_post_llm_provider="openai",
        social_post_llm_api_type="openai_responses",
        social_post_llm_model="social-test",
        social_post_llm_max_tokens=100,
        social_post_llm_pricing={
            "social-test": {
                "currency": "CNY",
                "uncached_input_cost_per_million": 1.0,
                "cached_input_cost_per_million": 0.1,
                "output_cost_per_million": 2.0,
            }
        },
    )
    conn, simulation, _ = make_initialized_sim(config=config)
    customer_simulator = CustomerSimulator(
        conn=conn,
        config=config,
        social_client=social,
    )
    customer_simulator.simulator = simulation
    simulation.customer_simulator = customer_simulator
    simulation.current_day = 7

    customer = conn.execute(
        "SELECT customer_id, group_id FROM customers ORDER BY customer_id LIMIT 1"
    ).fetchone()
    async_state = simulation._submit_social_posts_async(
        regular_work=[{
            "customer_id": customer["customer_id"],
            "group_id": customer["group_id"],
            "satisfaction": 0.5,
            "is_churned": False,
            "post_type": "general_satisfaction",
            "event_context": None,
        }],
        influence_cache={},
        macro_work=[],
    )
    simulation._collect_social_posts_async(async_state)

    post = conn.execute(
        "SELECT content FROM social_media_posts ORDER BY post_id DESC LIMIT 1"
    ).fetchone()
    cost = conn.execute(
        """
        SELECT model, purpose, input_tokens, cached_tokens, output_tokens,
               cost_amount, currency
        FROM api_costs ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    assert post["content"] == "response"
    assert dict(cost) == {
        "model": "social-test",
        "purpose": "customer_social_post",
        "input_tokens": 11,
        "cached_tokens": 0,
        "output_tokens": 7,
        "cost_amount": pytest.approx(0.000025),
        "currency": "CNY",
    }


def test_successful_zero_token_social_call_is_still_logged(make_initialized_sim):
    conn, simulation, _config = make_initialized_sim()
    customer_id = conn.execute(
        "SELECT customer_id FROM customers ORDER BY customer_id LIMIT 1"
    ).fetchone()[0]
    customer_simulator = SimpleNamespace(
        _log_cost=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    simulation.customer_simulator = customer_simulator
    simulation.current_day = 7
    calls = []
    simulation._process_social_post_results(
        [{
            "type": "macro",
            "customer_id": customer_id,
            "pmi": 50.0,
            "trend": "flat",
            "text": "macro post",
            "success": True,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "model": "local-model",
        }],
        {},
    )

    assert calls == [((7, "macro_social_post", 0, 0), {
        "cached_tokens": 0,
        "model": "local-model",
    })]


def test_local_model_cost_is_zero_when_explicitly_configured():
    config = BenchmarkConfig(
        social_post_llm_pricing={"qwen3-coder:30b": {
            "currency": "CNY",
            "uncached_input_cost_per_million": 0.0,
            "cached_input_cost_per_million": 0.0,
            "output_cost_per_million": 0.0,
        }},
    )
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=config,
    )

    cost = simulator._calculate_cost(
        1_000_000, 1_000_000, 0,
        model="qwen3-coder:30b", purpose="customer_social_post"
    )
    assert cost.amount == 0.0
    assert cost.currency == "CNY"


def test_unknown_model_cost_requires_explicit_pricing():
    simulator = CustomerSimulator(
        conn=sqlite3.connect(":memory:"),
        config=BenchmarkConfig(),
    )

    with pytest.raises(ValueError, match="No token pricing configured"):
        simulator._calculate_cost(
            1, 1, 0, model="unknown-model", purpose="customer_social_post"
        )


def test_decision_response_cost_uses_the_served_model(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.model = "requested"
    runner.pricing = {
        "requested": {
            "currency": "USD",
            "uncached_input_cost_per_million": 1.0,
            "cached_input_cost_per_million": 0.1,
            "output_cost_per_million": 2.0,
        },
        "served": {
            "currency": "CNY",
            "uncached_input_cost_per_million": 3.0,
            "cached_input_cost_per_million": 0.25,
            "output_cost_per_million": 4.0,
        },
    }
    runner.total_decision_agent_cost_by_currency = {}
    runner.response_log_file = tmp_path / "responses.jsonl"
    runner.agent = SimpleNamespace(
        last_input_tokens=1_000_000,
        last_output_tokens=1_000_000,
        last_cached_tokens=250_000,
        last_reasoning_tokens=125_000,
        last_serving_model="served",
    )

    runner._log_response(1, 0, [], {"model": "served"})

    entry = json.loads(runner.response_log_file.read_text())
    assert entry["served_model"] == "served"
    assert entry["cached_tokens"] == 250_000
    assert entry["reasoning_tokens"] == 125_000
    assert entry["cost_amount"] == pytest.approx(6.3125)
    assert entry["currency"] == "CNY"
    assert runner.total_decision_agent_cost_by_currency == {
        "CNY": pytest.approx(6.3125)
    }


def test_decision_agent_request_builder_uses_config_without_hidden_defaults():
    agent = BashAgent.__new__(BashAgent)
    agent.model = "decision-test"
    agent.max_output_tokens = 345
    agent.temperature = 0.51
    agent.top_p = 0.92
    agent.reasoning_effort = "none"
    agent.request_options = {}
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


@pytest.mark.parametrize(
    ("call_method", "builder_name"),
    [
        ("_call_openai", "_build_openai_chat_kwargs"),
        ("_call_openai_responses", "_build_openai_responses_kwargs"),
    ],
)
def test_openai_agent_does_not_retry_local_errors(call_method, builder_name):
    agent = BashAgent.__new__(BashAgent)
    agent.timeout_seconds = 1
    agent.conversation = []
    agent.tool_descriptions = []
    setattr(
        agent,
        builder_name,
        lambda *args: (_ for _ in ()).throw(OSError("local failure")),
    )

    with pytest.raises(OSError, match="local failure"):
        getattr(agent, call_method)()


@pytest.mark.parametrize(
    ("call_method", "builder_name", "client"),
    [
        (
            "_call_openai",
            "_build_openai_chat_kwargs",
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace())),
        ),
        (
            "_call_openai_responses",
            "_build_openai_responses_kwargs",
            SimpleNamespace(responses=SimpleNamespace()),
        ),
    ],
)
def test_openai_agent_does_not_retry_bad_requests(call_method, builder_name, client):
    import httpx
    import openai

    error = openai.BadRequestError(
        "bad request",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "http://example.test"),
        ),
        body={},
    )
    endpoint = (
        client.chat.completions
        if call_method == "_call_openai"
        else client.responses
    )
    endpoint.create = lambda **kwargs: (_ for _ in ()).throw(error)

    agent = BashAgent.__new__(BashAgent)
    agent.timeout_seconds = 1
    agent.conversation = []
    agent.tool_descriptions = []
    agent.client = client
    setattr(agent, builder_name, lambda *args: {})

    with pytest.raises(openai.BadRequestError):
        getattr(agent, call_method)()


@pytest.mark.parametrize(
    ("call_method", "builder_name", "client"),
    [
        (
            "_call_openai",
            "_build_openai_chat_kwargs",
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace())),
        ),
        (
            "_call_openai_responses",
            "_build_openai_responses_kwargs",
            SimpleNamespace(responses=SimpleNamespace()),
        ),
    ],
)
def test_openai_agent_does_not_add_unbounded_provider_retries(
    call_method, builder_name, client
):
    import httpx
    import openai

    calls = 0

    def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise openai.APIConnectionError(
            request=httpx.Request("POST", "http://example.test")
        )

    endpoint = (
        client.chat.completions
        if call_method == "_call_openai"
        else client.responses
    )
    endpoint.create = fail
    agent = BashAgent.__new__(BashAgent)
    agent.timeout_seconds = 1
    agent.conversation = []
    agent.tool_descriptions = []
    agent.client = client
    setattr(agent, builder_name, lambda *args: {})

    with pytest.raises(openai.APIConnectionError):
        getattr(agent, call_method)()

    assert calls == 1


def test_failed_next_week_is_not_returned_to_the_decision_agent(tmp_path, monkeypatch):
    class FailedProcess:
        returncode = 1
        pid = 1

        def communicate(self, timeout):
            return "", "Error: internal_error\n"

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: FailedProcess())
    executor = BashAgentToolExecutor(tmp_path)

    with pytest.raises(NextWeekExecutionError):
        executor._exec_bash({
            "command": "./novamind-operation next-week required-arguments"
        })


def test_soft_sandbox_blocks_simulator_import(tmp_path, monkeypatch):
    executor = BashAgentToolExecutor(tmp_path)
    monkeypatch.setattr(executor, "_build_bwrap_cmd", lambda *args: None)

    output = executor._exec_bash({
        "command": "python -c 'import saas_bench'",
    })

    assert "is blocked inside the bash_agent sandbox" in output
    assert "[exit code: 1]" in output


def test_file_tools_reject_sibling_path_with_workspace_prefix(tmp_path):
    workspace = tmp_path / "workspace"
    sibling = tmp_path / "workspace_private"
    workspace.mkdir()
    sibling.mkdir()
    secret = sibling / "secret.txt"
    secret.write_text("hidden")
    executor = BashAgentToolExecutor(workspace)

    read_result = executor.execute("read_file", {"path": str(secret)})
    glob_result = executor.execute(
        "glob_files", {"pattern": "../workspace_private/*"}
    )

    assert "Path escapes workspace" in read_result
    assert "Path escapes workspace" in glob_result


def test_openai_responses_stops_after_configured_invalid_response_limit():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model="test-model", usage=None, output=[])

    agent = BashAgent.__new__(BashAgent)
    agent.timeout_seconds = 1
    agent.conversation = []
    agent.tool_descriptions = []
    agent.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    agent.model = "test-model"
    agent.max_invalid_responses_per_turn = 2
    agent.total_turns = 0
    agent._consecutive_errors = 0
    agent.total_input_tokens = 0
    agent.total_output_tokens = 0
    agent.total_cached_tokens = 0
    agent.total_reasoning_tokens = 0
    agent.response_callback = None
    agent.tool_result_callback = None
    agent._build_openai_responses_kwargs = lambda *args: {}

    with pytest.raises(RuntimeError, match="2 responses"):
        agent._call_openai_responses()

    assert len(calls) == 2


def test_anthropic_agent_does_not_retry_local_errors():
    agent = BashAgent.__new__(BashAgent)
    agent.conversation = []
    agent._get_system_prompt_with_memory = lambda: (
        _ for _ in ()
    ).throw(OSError("local failure"))

    with pytest.raises(OSError, match="local failure"):
        agent._call_anthropic()


def test_anthropic_agent_does_not_create_unpriced_prompt_cache():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model="served-anthropic",
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=7,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="tool-1",
                    name="bash",
                    input={"command": "pwd"},
                )
            ],
        )

    agent = BashAgent(
        tool_descriptions=[],
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="test-model",
        api_type="anthropic_messages",
        max_invalid_responses_per_turn=2,
        max_output_tokens=100,
    )
    # 模拟旧 checkpoint 遗留的缓存断点，新请求不应继续携带它。
    agent.conversation = [
        Message(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "dashboard",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
    ]

    action = agent._call_anthropic()

    assert action == run_test.Action(tool="bash", arguments={"command": "pwd"})
    assert len(calls) == 1
    assert "cache_control" not in json.dumps(calls[0])
    assert agent.last_input_tokens == 11
    assert agent.last_output_tokens == 7


def test_anthropic_agent_does_not_retry_bad_requests():
    import anthropic
    import httpx

    error = anthropic.BadRequestError(
        "bad request",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "http://example.test"),
        ),
        body={},
    )
    messages = SimpleNamespace(
        create=lambda **kwargs: (_ for _ in ()).throw(error)
    )
    agent = BashAgent.__new__(BashAgent)
    agent.conversation = []
    agent._get_system_prompt_with_memory = lambda: "system"
    agent.model = "test-model"
    agent.max_output_tokens = 100
    agent.temperature = None
    agent.top_p = None
    agent.request_options = {}
    agent.client = SimpleNamespace(messages=messages)

    with pytest.raises(anthropic.BadRequestError):
        agent._call_anthropic()


def test_game_status_is_the_authoritative_day_source():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner._http_get = lambda path: {
        "day": 7,
        "cash": 900_000,
        "subscribers": 10,
        "timed_out": False,
    }

    assert runner._get_game_status()["day"] == 7


@pytest.mark.parametrize("status", [{}, {"day": None}, {"day": "7"}, {"day": -1}])
def test_invalid_game_status_fails_instead_of_falling_back_to_day_zero(status):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner._http_get = lambda path: status

    with pytest.raises(RuntimeError, match="Invalid simulator status"):
        runner._get_game_status()


def test_tool_execution_does_not_parse_dashboard_text():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.tool_executor = SimpleNamespace(
        execute=lambda tool, arguments: "=== arbitrary future dashboard format ==="
    )
    runner.agent = SimpleNamespace(
        check_day_advanced=lambda output: pytest.fail(
            "dashboard text must not control week advancement"
        )
    )

    assert "arbitrary" in runner._execute_tool("bash", {"command": "next-week"})


def test_runtime_restore_truncates_logs_to_exact_same_day_offsets(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    files = runner._checkpoint_log_files()
    original = {}
    for name, path in files.items():
        content = f'{name}-checkpoint\n'.encode()
        path.write_bytes(content + b"same-day-future\n")
        original[name] = len(content)

    runner._restore_logs_to_offsets(original)

    for name, path in files.items():
        assert path.read_bytes() == f'{name}-checkpoint\n'.encode()


def test_server_runtime_restore_truncates_logs_before_launch(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    files = runner._server_log_files()
    offsets = {}
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_content = f"{name}-checkpoint\n".encode()
        path.write_bytes(checkpoint_content + b"same-day-future\n")
        offsets[name] = len(checkpoint_content)

    runner._restore_server_logs_before_server(offsets)

    for name, path in files.items():
        assert path.read_bytes() == f"{name}-checkpoint\n".encode()


def test_server_runtime_restore_removes_stale_terminal_metadata(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    files = runner._server_log_files()
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    event_log = files["event_log"]
    meta_file = event_log.with_name(event_log.stem + "_meta.json")
    meta_file.write_text('{"outcome":"completed"}')

    runner._restore_server_logs_before_server({"history": 0, "event_log": 0})

    assert not meta_file.exists()


def test_server_runtime_restore_rejects_offset_beyond_file_size(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    files = runner._server_log_files()
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"short")

    with pytest.raises(ValueError, match="exceeds file size"):
        runner._restore_server_logs_before_server({
            "history": 6,
            "event_log": 5,
        })


@pytest.mark.parametrize(
    "offsets",
    [None, {}, {"history": 0}, {"history": -1, "event_log": 0}],
)
def test_server_runtime_restore_requires_complete_valid_offsets(tmp_path, offsets):
    runner = _checkpoint_runner(tmp_path)

    with pytest.raises(ValueError, match="server log offset"):
        runner._restore_server_logs_before_server(offsets)


def test_event_logger_records_each_explicit_event_day(tmp_path):
    logger = EventLogger("days", tmp_path, 42, "default", {})
    logger.log_customer_signup(
        day=3,
        customer_id=1,
        group_id="S1",
        plan="A",
        price=10.0,
        is_enterprise=False,
    )
    logger.log_llm_call(
        day=7,
        purpose="customer_social_post",
        model="test-model",
        input_tokens=10,
        cached_tokens=4,
        output_tokens=5,
        cost_amount=0.25,
        currency="CNY",
    )
    logger.save_incremental()

    events = [json.loads(line) for line in logger.log_file.read_text().splitlines()]
    assert [event["day"] for event in events] == [3, 7]


def test_event_logger_preserves_session_start_time_across_server_restarts(tmp_path):
    logger = EventLogger(
        "start-time",
        tmp_path,
        42,
        "default",
        {},
        start_time="2026-01-02T03:04:05Z",
    )
    logger.log_run_end(day=7, final_cash=100.0, days_run=7, outcome="completed")
    logger.save()

    metadata = json.loads(logger.meta_file.read_text())
    assert metadata["start_time"] == "2026-01-02T03:04:05Z"


def test_event_logger_continues_llm_cost_from_restored_database_total(tmp_path):
    logger = EventLogger(
        "resume-cost",
        tmp_path,
        42,
        "default",
        {},
        starting_llm_cost_by_currency={"CNY": 1.25},
    )
    logger.log_llm_call(
        day=8,
        purpose="customer_negotiation",
        model="test-model",
        input_tokens=10,
        cached_tokens=4,
        output_tokens=5,
        cost_amount=0.75,
        currency="CNY",
    )
    logger.log_run_end(day=8, final_cash=100.0, days_run=8, outcome="completed")
    logger.save()

    metadata = json.loads(logger.meta_file.read_text())
    assert metadata["total_llm_cost_by_currency"] == {"CNY": pytest.approx(2.0)}


def test_event_logger_accepts_structured_agent_action_result(tmp_path):
    logger = EventLogger("structured-result", tmp_path, 42, "default", {})
    logger.log_agent_action(
        day=7,
        tool_name="log_rationale",
        arguments={"rationale": "hold prices"},
        result={"logged": True},
        success=True,
    )
    logger.save_incremental()

    event = json.loads(logger.log_file.read_text())
    assert event["day"] == 7
    assert event["details"]["result"] == {"logged": True}


def test_runner_releases_resources_when_experiment_fails():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = None
    calls = []
    runner._start_timing_poster = lambda: calls.append("timing-start")
    runner._run_experiment = lambda verbose: (_ for _ in ()).throw(
        RuntimeError("experiment failed")
    )
    runner._stop_server = lambda: calls.append("server-stop")
    runner._stop_timing_poster = lambda: calls.append("timing-stop")

    with pytest.raises(RuntimeError, match="experiment failed"):
        runner.run(verbose=False)

    assert calls == ["timing-start", "server-stop", "timing-stop"]


def test_stop_server_reaps_process_after_forced_kill():
    calls = []

    class Process:
        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is not None:
                raise run_test.subprocess.TimeoutExpired("server", timeout)

        def kill(self):
            calls.append("kill")

    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner._server_proc = Process()
    runner._server_port = 12345
    runner._server_stderr_file = None

    runner._stop_server()

    assert calls == ["terminate", ("wait", 210), "kill", ("wait", None)]
    assert runner._server_proc is None
    assert runner._server_port is None


def test_resume_setup_rejects_missing_checkpoint_instead_of_guessing_session(
    tmp_path, monkeypatch
):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = tmp_path
    runner.workspace_dir = tmp_path
    runner.agent_workspace = tmp_path / "agent_workspace"
    guessed_session = runner.agent_workspace / "sessions" / "latest-session"
    guessed_session.mkdir(parents=True)
    runner._session_id = None
    runner._load_checkpoint = lambda: None
    runner._verify_public_bundle = lambda: None
    monkeypatch.setattr(
        "saas_bench.agents.bash_agent.tools.get_bash_agent_tool_descriptions",
        lambda: [],
    )

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        runner.setup()

    assert runner._session_id is None


def test_resume_setup_rejects_changed_public_bundle_before_restore(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = tmp_path
    runner.public_bundle_sha256 = "0" * 64
    runner._current_public_bundle_sha256 = lambda: "1" * 64
    runner._load_checkpoint = lambda: pytest.fail(
        "checkpoint restore must not start after bundle mismatch"
    )

    with pytest.raises(ValueError, match="bundle hash does not match"):
        runner.setup()


def test_public_bundle_hash_includes_agent_documentation(tmp_path):
    public_dir = tmp_path / "public"
    docs = public_dir / "docs"
    docs.mkdir(parents=True)
    (public_dir / "novamind-operation").write_bytes(b"executable")
    reference = docs / "cli.md"
    reference.write_text("version one")
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner._public_dir = lambda: public_dir

    first = runner._current_public_bundle_sha256()
    reference.write_text("version two")

    assert runner._current_public_bundle_sha256() != first


def test_harness_hash_changes_with_main_agent_source(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    first = runner._current_harness_source_sha256()
    agent_file = PROJECT_ROOT / "src/saas_bench/agents/bash_agent/agent.py"
    original = agent_file.read_bytes()
    try:
        agent_file.write_bytes(original + b"\n")
        assert runner._current_harness_source_sha256() != first
    finally:
        agent_file.write_bytes(original)


def test_run_config_records_harness_identity(monkeypatch):
    runner = run_test._new_experiment_runner(
        PROJECT_ROOT / "experiments/smoke.toml"
    )
    monkeypatch.setattr(
        runner, "_current_public_bundle_sha256", lambda: "0" * 64
    )
    payload = runner._run_config_payload()

    assert payload["harness_git_commit"]
    assert isinstance(payload["harness_git_dirty"], bool)
    assert len(payload["harness_source_sha256"]) == 64


def test_new_setup_removes_run_directory_when_session_creation_fails(
    tmp_path, monkeypatch
):
    runner = run_test._new_experiment_runner(
        PROJECT_ROOT / "experiments/smoke.toml"
    )
    runner.workspace_base = tmp_path / "runs"
    runner.workspace_dir = runner.workspace_base / f"run_{runner.run_id}"
    runner.agent_workspace = runner.workspace_dir / "agent_workspace"
    runner.logs_dir = runner.workspace_dir / "logs"
    runner.response_log_file = runner.logs_dir / f"raw_responses_{runner.run_id}.jsonl"
    runner.timing_log_file = runner.logs_dir / f"timing_{runner.run_id}.jsonl"
    runner._verify_public_bundle = lambda: None
    runner._initialize_from_public_repo = lambda: (
        _ for _ in ()
    ).throw(RuntimeError("session creation failed"))
    monkeypatch.setattr(
        "saas_bench.agents.bash_agent.tools.get_bash_agent_tool_descriptions",
        lambda: [],
    )

    with pytest.raises(RuntimeError, match="session creation failed"):
        runner.setup()

    assert not runner.workspace_dir.exists()
    assert runner._session_id is None


def test_new_setup_removes_run_directory_when_initial_checkpoint_fails(
    tmp_path, monkeypatch
):
    runner = run_test._new_experiment_runner(
        PROJECT_ROOT / "experiments/smoke.toml"
    )
    runner.workspace_base = tmp_path / "runs"
    runner.workspace_dir = runner.workspace_base / f"run_{runner.run_id}"
    runner.agent_workspace = runner.workspace_dir / "agent_workspace"
    runner.logs_dir = runner.workspace_dir / "logs"
    runner.response_log_file = runner.logs_dir / f"raw_responses_{runner.run_id}.jsonl"
    runner.timing_log_file = runner.logs_dir / f"timing_{runner.run_id}.jsonl"
    runner._verify_public_bundle = lambda: None
    runner._initialize_from_public_repo = lambda: setattr(
        runner, "_session_id", "session-1"
    )
    runner._launch_server_from_prepared_checkpoint = lambda: setattr(
        runner, "_server_port", 12345
    )
    runner._save_checkpoint = lambda day: (
        _ for _ in ()
    ).throw(RuntimeError("checkpoint failed"))
    runner._stop_server = lambda: None
    monkeypatch.setattr(
        "saas_bench.agents.bash_agent.tools.get_bash_agent_tool_descriptions",
        lambda: [],
    )

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        runner.setup()

    assert not runner.workspace_dir.exists()
    assert runner._session_id is None
    assert runner.agent is None
    assert runner.tool_executor is None


def test_resume_setup_does_not_mutate_workspace_before_restore_preflight(
    tmp_path, monkeypatch
):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = tmp_path
    runner.workspace_dir = tmp_path
    runner.agent_workspace = tmp_path / "agent_workspace"
    legacy_marker = runner.agent_workspace / "_engine" / "marker"
    legacy_marker.parent.mkdir(parents=True)
    legacy_marker.write_text("unchanged")
    runner._session_id = None
    runner._load_checkpoint = lambda: {"session_id": "session-1"}
    runner._verify_public_bundle = lambda: None
    runner._launch_server_from_prepared_checkpoint = lambda: (
        _ for _ in ()
    ).throw(ValueError("preflight failed"))
    monkeypatch.setattr(
        "saas_bench.agents.bash_agent.tools.get_bash_agent_tool_descriptions",
        lambda: [],
    )

    with pytest.raises(ValueError, match="preflight failed"):
        runner.setup()

    assert legacy_marker.read_text() == "unchanged"


def test_runner_writes_machine_readable_result_atomically(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path
    result = {
        "run_id": "test",
        "outcome": "timeout",
        "final_cash": 123.0,
        "resumable": True,
    }

    runner._write_result(result)

    assert json.loads((tmp_path / "result.json").read_text()) == result
    assert not (tmp_path / "result.json.tmp").exists()


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        ({"day": 7, "cash": -1.0, "timed_out": True}, "timeout"),
        ({"day": 7, "cash": -1.0, "timed_out": False}, "bankrupt"),
        ({"day": 7, "cash": 1.0, "timed_out": False}, "completed"),
        ({"day": 0, "cash": 1.0, "timed_out": False}, None),
    ],
)
def test_runner_terminal_outcome_prioritizes_failures(status, outcome):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.total_days = 7

    assert runner._terminal_outcome(status) == outcome


def test_resume_validation_failure_preserves_previous_nonterminal_result(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = tmp_path
    runner.workspace_dir = tmp_path
    result_file = tmp_path / "result.json"
    previous_result = {"outcome": "incomplete", "resumable": True}
    result_file.write_text(json.dumps(previous_result))
    runner._load_checkpoint = lambda: (
        _ for _ in ()
    ).throw(ValueError("invalid checkpoint"))

    with pytest.raises(ValueError, match="invalid checkpoint"):
        runner._load_or_rebuild_terminal_result()

    assert json.loads(result_file.read_text()) == previous_result


def test_run_returns_existing_terminal_result_without_starting_resources(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()
    result = runner._result_from_checkpoint(checkpoint, "completed")
    (runner.workspace_dir / "result.json").write_text(json.dumps(result))
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": runner._session_id,
        "status": "completed",
        "current_day": 7,
        "final_cash": 850_000.0,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / f"run_{runner._session_id}.jsonl").write_text(json.dumps({
        "day": 7,
        "event_type": "lifecycle",
        "category": "run_end",
        "details": {"outcome": "completed", "final_cash": 850_000.0},
    }) + "\n")
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "completed",
        "days_run": 7,
        "final_cash": 850_000.0,
    }))
    runner._start_timing_poster = lambda: pytest.fail("must not start timing poster")
    runner._run_experiment = lambda verbose: pytest.fail("must not run experiment")

    assert runner.run(verbose=False) == result


def test_runner_rejects_terminal_result_that_disagrees_with_checkpoint(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()
    result = runner._result_from_checkpoint(checkpoint, "completed")
    result["final_cash"] = 1.0
    (runner.workspace_dir / "result.json").write_text(json.dumps(result))
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "status": "completed",
        "current_day": 7,
        "final_cash": 850_000.0,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / f"run_{runner._session_id}.jsonl").write_text(json.dumps({
        "day": 7,
        "category": "run_end",
        "details": {"outcome": "completed", "final_cash": 850_000.0},
    }) + "\n")
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "completed",
        "days_run": 7,
        "final_cash": 850_000.0,
    }))

    with pytest.raises(RuntimeError, match="authoritative artifacts"):
        runner._load_or_rebuild_terminal_result()


def test_runner_rejects_completed_terminal_result_with_negative_cash(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": -1.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()
    result = runner._result_from_checkpoint(checkpoint, "completed")
    (runner.workspace_dir / "result.json").write_text(json.dumps(result))
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "status": "completed",
        "current_day": 7,
        "final_cash": -1.0,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / f"run_{runner._session_id}.jsonl").write_text(json.dumps({
        "day": 7,
        "category": "run_end",
        "details": {"outcome": "completed", "final_cash": -1.0},
    }) + "\n")
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "completed",
        "days_run": 7,
        "final_cash": -1.0,
    }))

    with pytest.raises(RuntimeError, match="negative cash"):
        runner._load_or_rebuild_terminal_result()


def test_runner_rebuilds_missing_terminal_result_from_consistent_artifacts(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)

    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": runner._session_id,
        "status": "completed",
        "current_day": 7,
        "final_cash": 850_000.0,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    event_log = logs_dir / f"run_{runner._session_id}.jsonl"
    event_log.write_text(json.dumps({
        "day": 7,
        "event_type": "lifecycle",
        "category": "run_end",
        "details": {
            "outcome": "completed",
            "final_cash": 850_000.0,
        },
    }) + "\n")
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "completed",
        "days_run": 7,
        "final_cash": 850_000.0,
    }))

    result = runner._load_or_rebuild_terminal_result()

    assert result["outcome"] == "completed"
    assert result["days_run"] == 7
    assert result["final_cash"] == pytest.approx(850_000.0)
    assert result["resumable"] is False
    assert json.loads((runner.workspace_dir / "result.json").read_text()) == result


def test_runner_rejects_disagreeing_terminal_artifacts(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)

    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "status": "completed",
        "current_day": 7,
        "final_cash": 850_000.0,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "bankrupt",
        "days_run": 7,
        "final_cash": 850_000.0,
    }))

    with pytest.raises(RuntimeError, match="artifacts disagree"):
        runner._load_or_rebuild_terminal_result()


@pytest.mark.parametrize("terminal_artifact", ["session_meta", "event_meta"])
def test_runner_repairs_interrupted_terminal_finalization(
    tmp_path, terminal_artifact
):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": runner._session_id,
        "status": "running",
        "current_day": 7,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    event_log = logs_dir / f"run_{runner._session_id}.jsonl"
    event_log.write_text("")
    history_log = session_dir / "history.jsonl"
    history_log.write_text("")
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()

    # 模拟 finalize 在多文件提交之间崩溃，只留下其中一份终态证据。
    if terminal_artifact == "session_meta":
        (session_dir / "session.json").write_text(json.dumps({
            "session_id": runner._session_id,
            "status": "completed",
            "current_day": 7,
            "final_cash": 850_000.0,
        }))
    else:
        (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
            "outcome": "completed",
            "days_run": 7,
            "final_cash": 850_000.0,
        }))
    event_log.write_text(json.dumps({
        "day": 7,
        "event_type": "lifecycle",
        "category": "run_end",
        "details": {"outcome": "completed", "final_cash": 850_000.0},
    }) + "\n")

    assert runner._load_or_rebuild_terminal_result() is None

    runner._resume_checkpoint = checkpoint
    runner._refresh_public_workspace_artifacts = lambda: None
    runner._launch_server = lambda: None
    runner._http_get = lambda path: {"day": 7}
    runner._launch_server_from_prepared_checkpoint()

    # 恢复严格回到 checkpoint 边界，半完成的 run_end 与 meta 均被清除。
    assert event_log.read_text() == ""
    assert not (logs_dir / f"run_{runner._session_id}_meta.json").exists()
    restored_session_meta = json.loads((session_dir / "session.json").read_text())
    assert restored_session_meta["status"] == "created"

    finalize_calls = []
    runner._http_post = lambda path, data, timeout: (
        finalize_calls.append((path, data, timeout))
        or {"success": True, "outcome": "completed"}
    )
    result = runner._repair_terminal_checkpoint_after_setup()

    assert finalize_calls == [("/finalize-run", {"outcome": "completed"}, 30)]
    assert result["outcome"] == "completed"
    assert result["resumable"] is False
    assert json.loads((runner.workspace_dir / "result.json").read_text()) == result


def test_workspace_restore_removes_changes_after_checkpoint(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    tracked = runner.agent_workspace / "MEMORY.md"
    tracked.write_text("checkpoint memory")
    checkpoint_commit = runner._capture_workspace_commit(7)
    tracked.write_text("future memory")
    (runner.agent_workspace / "future.txt").write_text("future")
    ignored_session = runner.agent_workspace / "sessions" / "session-1" / "world.nmdb"
    ignored_session.parent.mkdir(parents=True, exist_ok=True)
    ignored_session.write_bytes(b"database")

    runner._restore_workspace_commit(checkpoint_commit)

    assert tracked.read_text() == "checkpoint memory"
    assert not (runner.agent_workspace / "future.txt").exists()
    assert ignored_session.read_bytes() == b"database"


def test_resume_rebuilds_week_commit_cursor_from_checkpoint_day():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.agent = None
    runner.total_decision_agent_cost_by_currency = {}
    runner._last_committed_week = 0
    checkpoint = {
        "day": 35,
        "runtime": {
            "agent": {
                "total_turns": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "decision_cost_by_currency": {},
            }
        },
    }
    restore_plan = run_test.CheckpointRestorePlan(
        session_id="session-1",
        conversation_payload={},
    )

    runner._restore_agent_state_after_launch(checkpoint, restore_plan)

    assert runner._last_committed_week == 5


def test_week_commit_cursor_does_not_advance_when_git_commit_fails():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner._last_committed_week = 0
    runner._git_commit_workspace = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("git failed"))

    with pytest.raises(RuntimeError, match="git failed"):
        runner._commit_weeks_up_to(7)

    assert runner._last_committed_week == 0


def test_agent_checkpoint_snapshot_applies_pending_tool_result(tmp_path):
    agent = BashAgent.__new__(BashAgent)
    agent.use_anthropic = False
    agent.conversation = [
        Message(
            role="assistant",
            content=[{
                "type": "function_call",
                "call_id": "call-1",
                "name": "bash",
                "arguments": '{"command":"query"}',
            }],
        )
    ]
    agent._pending_tool_calls = [{"id": "call-1", "name": "bash"}]
    agent.current_day = 7
    agent.turns_today = 2
    agent.total_turns = 2
    snapshot = tmp_path / "conversation.json"

    agent.save_checkpoint_snapshot(
        snapshot,
        resume_conversation=True,
        pending_observation="query result",
    )

    payload = json.loads(snapshot.read_text())
    assert payload["tool_results_applied"] is True
    assert payload["pending_tool_calls"] == []
    assert payload["conversation"][-1] == {
        "role": "tool",
        "content": "query result",
        "tool_calls": None,
        "tool_call_id": "call-1",
        "name": "bash",
    }


def test_restored_midweek_context_does_not_duplicate_first_observation():
    agent = BashAgent.__new__(BashAgent)
    agent.conversation = [Message(role="tool", content="saved tool result")]
    agent._pending_tool_calls = []
    agent.current_day = 7
    agent.turns_today = 3
    agent._last_observation = ""
    agent._skip_next_observation = True
    captured = []
    agent._call_llm = lambda: captured.append(list(agent.conversation)) or SimpleNamespace(
        tool="bash", arguments={"command": "query"}
    )
    agent._save_conversation_snapshot = lambda: None

    agent.act("fresh dashboard must be skipped", 0, False, {"day": 7})

    assert [message.content for message in captured[0]] == ["saved tool result"]
    assert agent._skip_next_observation is False


def test_day_zero_initializes_chat_context_with_system_prompt():
    agent = BashAgent.__new__(BashAgent)
    agent.use_anthropic = False
    agent.conversation = []
    agent._pending_tool_calls = []
    agent.current_day = 0
    agent.turns_today = 0
    agent._last_observation = ""
    agent._skip_next_observation = False
    agent._get_system_prompt_with_memory = lambda: "system prompt"
    captured = []
    agent._call_llm = lambda: captured.append(list(agent.conversation)) or SimpleNamespace(
        tool="bash", arguments={"command": "query"}
    )
    agent._save_conversation_snapshot = lambda: None

    agent.act("day zero dashboard", 0, False, {"day": 0})

    assert [(message.role, message.content) for message in captured[0]] == [
        ("system", "system prompt"),
        ("user", "day zero dashboard"),
    ]


def test_agent_does_not_hide_memory_read_failure(tmp_path):
    agent = BashAgent.__new__(BashAgent)
    agent.system_prompt = "system prompt"
    agent.workspace_path = tmp_path
    (tmp_path / "MEMORY.md").mkdir()

    with pytest.raises(OSError):
        agent._get_system_prompt_with_memory()


def test_turn_limit_saves_one_resumable_midweek_checkpoint(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = None
    runner.total_days = 7
    runner.max_decision_turns_per_batch = 1
    runner.total_decision_agent_cost_by_currency = {}
    runner.run_id = "turn-limit"
    runner.seed = 42
    runner.scenario = "default"
    runner.model = "test-model"
    runner.workspace_dir = tmp_path
    runner._server_port = 1
    runner.setup = lambda: None
    runner._repair_terminal_checkpoint_after_setup = lambda: None
    runner._get_game_status = lambda: {
        "day": 0,
        "cash": 1_000_000.0,
        "subscribers": 0,
        "timed_out": False,
    }
    runner._get_cash = lambda: 1_000_000.0
    runner._get_dashboard = lambda: "dashboard"
    runner._log_tool_result = lambda *args, **kwargs: None
    runner._log_timing = lambda *args, **kwargs: None
    runner._commit_weeks_up_to = lambda day: None
    runner._execute_tool = lambda tool, arguments: "query result"
    runner._http_get = lambda path: {
        "day": 0,
        "cash": 1_000_000.0,
        "subscribers": 0,
        "timed_out": False,
    }
    runner._NextWeekTimeoutError = RuntimeError
    runner._write_result = lambda result: None
    runner._harness_result_fields = lambda: {}
    checkpoint_calls = []

    def save_checkpoint(day, **kwargs):
        checkpoint_calls.append((day, kwargs))
        return {
            "day": day,
            "cash": 1_000_000.0,
            "runtime": {
                "agent": {
                    "total_turns": 1,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "decision_cost_by_currency": {},
                },
                "environment_llm": EMPTY_ENVIRONMENT_LLM_USAGE,
            },
        }

    runner._save_checkpoint = save_checkpoint

    agent = SimpleNamespace(
        total_turns=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cached_tokens=0,
        total_reasoning_tokens=0,
        last_serving_model="test-model",
    )

    def act(*args):
        agent.total_turns += 1
        return SimpleNamespace(tool="bash", arguments={"command": "query"})

    agent.act = act
    runner.agent = agent

    result = runner._run_experiment(verbose=False)

    assert result["outcome"] == "incomplete"
    assert result["resumable"] is True
    assert result["total_turns"] == 1
    assert result["decision_agent_input_tokens"] == 0
    assert result["decision_agent_output_tokens"] == 0
    assert result["decision_agent_cached_tokens"] == 0
    assert result["decision_agent_reasoning_tokens"] == 0
    assert result["environment_llm_input_tokens"] == 0
    assert result["environment_llm_output_tokens"] == 0
    assert result["environment_llm_cost_by_currency"] == {}
    assert result["environment_llm_usage_by_purpose"] == {}
    assert checkpoint_calls == [
        (0, {
            "resume_conversation": True,
            "pending_observation": "query result",
        }),
    ]
