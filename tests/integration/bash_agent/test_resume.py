"""按职责拆分的 Harness 回归测试。"""

import json

from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent.runner import BashAgentRunner
from saas_bench.agents.bash_agent.checkpoint import CheckpointStore
from saas_bench.agents.bash_agent.workspace import AgentWorkspaceRepository
from saas_bench.agents.bash_agent.run_config import (
    create_new_runner,
    create_resumed_runner,
    load_saved_run_config,
    resolve_run_directory,
)


from tests.support.harness import (
    EMPTY_ANALYSIS_USAGE,
    EMPTY_ENVIRONMENT_LLM_USAGE,
    TEST_CONFIG,
    make_checkpoint_runner as _checkpoint_runner,
)


def test_resume_loads_the_saved_configuration_without_external_overrides(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run_existing"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "existing",
        "experiment_name": "baseline",
        "agent_type": "bash_agent",
        "model": "original-model",
        "api_type": "openai_responses",
        "base_url": "http://localhost:11434/v1",
        "reasoning_effort": None,
        "temperature": 0.7,
        "top_p": 0.8,
        "tool_choice": "required",
        "max_output_tokens": 100,
        "timeout_seconds": 30.0,
        "pricing": {"original-model": {
            "currency": "USD",
            "uncached_input_cost_per_million": 1.0,
            "cached_input_cost_per_million": 0.1,
            "output_cost_per_million": 2.0,
        }},
        "pricing_model_map": {},
        "request_options": {},
        "api_key_env": None,
        "api_key_required": False,
        "seed": 42,
        "scenario": "default",
        "total_days": 70,
        "initial_cash": 1_000_000.0,
        "max_decision_turns_per_batch": 100,
        "max_invalid_responses_per_turn": 3,
        "simulator_llm": {},
        "analysis_module": {
            "enabled": False,
            "max_schema_retries": 1,
            "max_enterprise_threads": 50,
            "role_report_concurrency": 1,
        },
        "analysis_model": None,
        "git_commit": "test-commit",
    }))

    monkeypatch.setattr(
        BashAgentRunner,
        "_read_git_commit",
        staticmethod(lambda: "test-commit"),
    )
    runner = create_resumed_runner(str(run_dir))

    assert runner.model == "original-model"
    assert runner.experiment_name == "baseline"
    assert runner.api_type == "openai_responses"
    assert runner.tool_choice == "required"
    assert runner.temperature == pytest.approx(0.7)
    assert runner.pricing["original-model"]["uncached_input_cost_per_million"] == pytest.approx(1.0)
    assert runner.workspace_dir == run_dir.resolve()


def test_run_id_resolves_nested_experiment_directory(tmp_path):
    run_dir = tmp_path / "analysis" / "20260826-190015_seed-42_abc12345"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"run_id": "abc12345"}))

    assert resolve_run_directory("abc12345", tmp_path) == run_dir.resolve()


def test_resume_aborts_when_git_commit_differs(tmp_path, capsys, monkeypatch):
    run_dir = tmp_path / "run_existing"
    run_dir.mkdir()
    source = create_new_runner(
        TEST_CONFIG
    )
    source.git_commit = "original-commit"
    config = source._run_config_payload()
    config["git_commit"] = "original-commit"
    (run_dir / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        BashAgentRunner,
        "_read_git_commit",
        staticmethod(lambda: "current-commit"),
    )

    with pytest.raises(RuntimeError, match="Git commit differs"):
        create_resumed_runner(str(run_dir))
    assert "Current Git commit differs" in capsys.readouterr().err

def test_resume_rejects_run_config_missing_required_field(tmp_path):
    run_dir = tmp_path / "run_existing"
    run_dir.mkdir()
    source = create_new_runner(
        TEST_CONFIG
    )
    source.git_commit = "test-commit"
    config = source._run_config_payload()
    config.pop("pricing")
    (run_dir / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="missing fields"):
        load_saved_run_config(run_dir)


def test_resume_ignores_unrelated_run_config_fields(tmp_path):
    run_dir = tmp_path / "run_existing"
    run_dir.mkdir()
    source = create_new_runner(
        TEST_CONFIG
    )
    source.git_commit = "test-commit"
    config = source._run_config_payload()
    config["note"] = "does not affect the experiment"
    (run_dir / "config.json").write_text(json.dumps(config))

    assert load_saved_run_config(run_dir)["note"] == config["note"]

def test_resume_setup_never_rewrites_run_config(tmp_path, monkeypatch):
    runner = create_new_runner(
        TEST_CONFIG
    )
    runner.git_commit = "test-commit"
    run_dir = tmp_path / f"run_{runner.run_id}"
    run_dir.mkdir()
    config_file = run_dir / "config.json"
    config_file.write_text(json.dumps(runner._run_config_payload(), indent=4))
    original_bytes = config_file.read_bytes()
    monkeypatch.setattr(
        BashAgentRunner,
        "_read_git_commit",
        staticmethod(lambda: "test-commit"),
    )
    resumed = create_resumed_runner(str(run_dir))
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
    runner.workspace_repository = AgentWorkspaceRepository(runner.agent_workspace)
    runner.workspace_repository.initialize()
    runner.workspace_repository.commit("checkpoint workspace")
    runner.checkpoint_store = CheckpointStore(
        workspace_dir=runner.workspace_dir,
        agent_workspace=runner.agent_workspace,
        trajectory_log_file=runner.workspace_dir / "logs/trajectory.jsonl",
        performance_log_file=runner.workspace_dir / "logs/performance.jsonl",
        workspace_repository=runner.workspace_repository,
    )
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    runner.simulator_server = SimpleNamespace(public_dir=public_dir)
    workspace_commit = runner.workspace_repository.git(
        "rev-parse", "HEAD", check=True
    ).stdout.strip()
    runner._resume_checkpoint = {
        "day": 35,
        "session_id": "session-1",
        "database": {
            "file": ".checkpoint_dbs/world_day_35.nmdb",
        },
        "runtime": {
            "workspace_commit": workspace_commit,
            "runner_log_offsets": {},
            "server_log_offsets": {"history": 0, "event_log": 0},
            "analysis": EMPTY_ANALYSIS_USAGE,
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
    runner.checkpoint_store.preflight = lambda checkpoint: SimpleNamespace(
        session_id="session-1", conversation_payload={}
    )
    runner.checkpoint_store.restore_runner_logs = lambda offsets: None
    runner.analysis_pipeline = SimpleNamespace(
        prune_artifacts_after=lambda *args: None
    )
    runner._launch_server_from_prepared_checkpoint()

    assert observed["database"] == b"checkpoint-state"
    assert observed["metadata"]["current_day"] == 35
    assert observed["metadata"]["status"] == "created"
    assert "port" not in observed["metadata"]
    assert "pid" not in observed["metadata"]

def test_resume_stops_server_when_restored_day_does_not_match(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path
    runner._resume_checkpoint = {
        "day": 35,
        "runtime": {
            "workspace_commit": "checkpoint-commit",
            "runner_log_offsets": {},
            "server_log_offsets": {"history": 0, "event_log": 0},
            "analysis": EMPTY_ANALYSIS_USAGE,
        },
    }
    calls = []
    runner.workspace_repository = SimpleNamespace(
        restore_commit=lambda commit: calls.append("workspace"),
        refresh_public_artifacts=lambda public_dir: calls.append("refresh"),
    )
    runner.checkpoint_store = SimpleNamespace(
        preflight=lambda checkpoint: (
            calls.append("preflight")
            or SimpleNamespace(session_id="session-1", conversation_payload={})
        ),
        restore_database=lambda checkpoint: calls.append("restore"),
        restore_runner_logs=lambda offsets: calls.append("runner-logs"),
        restore_server_logs=lambda offsets, session_id: calls.append("logs"),
    )
    runner._public_dir = lambda: tmp_path / "public"
    runner.analysis_pipeline = SimpleNamespace(
        prune_artifacts_after=lambda *args: calls.append("analysis")
    )
    runner._launch_server = lambda: calls.append("launch")
    runner._http_get = lambda path: {"day": 28}
    runner._stop_server = lambda: calls.append("stop")

    with pytest.raises(RuntimeError, match="does not match checkpoint day 35"):
        runner._launch_server_from_prepared_checkpoint()

    assert calls == [
        "preflight", "workspace", "refresh", "restore",
        "runner-logs", "logs", "analysis", "launch", "stop",
    ]




def test_restore_preflight_rejects_invalid_conversation_before_mutation(tmp_path):
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

    with pytest.raises(ValueError, match="invalid JSON"):
        runner._launch_server_from_prepared_checkpoint()

    assert tracked.read_text() == "future workspace state"
    assert session_db.read_bytes() == b"future database state"
