"""按职责拆分的 Harness 回归测试。"""

import json
from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent import runner as runner_module
from saas_bench.agents.bash_agent.runner import BashAgentRunner
from saas_bench.agents.bash_agent.run_config import create_new_runner
from saas_bench.agents.bash_agent.simulator_server import SimulatorServer


from tests.support.harness import (
    TEST_CONFIG,
    make_checkpoint_runner as _checkpoint_runner,
)

def test_runtime_restore_truncates_logs_to_exact_same_day_offsets(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    files = runner.checkpoint_store.runner_log_files()
    original = {}
    for name, path in files.items():
        content = f'{name}-checkpoint\n'.encode()
        path.write_bytes(content + b"same-day-future\n")
        original[name] = len(content)

    runner.checkpoint_store.restore_runner_logs(original)

    for name, path in files.items():
        assert path.read_bytes() == f'{name}-checkpoint\n'.encode()

def test_server_runtime_restore_truncates_logs_before_launch(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    files = runner.checkpoint_store.server_log_files(runner._session_id)
    offsets = {}
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_content = f"{name}-checkpoint\n".encode()
        path.write_bytes(checkpoint_content + b"same-day-future\n")
        offsets[name] = len(checkpoint_content)

    runner.checkpoint_store.restore_server_logs(offsets, runner._session_id)

    for name, path in files.items():
        assert path.read_bytes() == f"{name}-checkpoint\n".encode()

def test_server_runtime_restore_removes_stale_terminal_metadata(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    files = runner.checkpoint_store.server_log_files(runner._session_id)
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    event_log = files["event_log"]
    meta_file = event_log.with_name(event_log.stem + "_meta.json")
    meta_file.write_text('{"outcome":"completed"}')

    runner.checkpoint_store.restore_server_logs(
        {"history": 0, "event_log": 0}, runner._session_id
    )

    assert not meta_file.exists()

def test_server_runtime_restore_rejects_offset_beyond_file_size(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    files = runner.checkpoint_store.server_log_files(runner._session_id)
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"short")

    with pytest.raises(ValueError, match="exceeds file size"):
        runner.checkpoint_store.restore_server_logs(
            {"history": 6, "event_log": 5}, runner._session_id
        )

@pytest.mark.parametrize(
    "offsets",
    [None, {}, {"history": 0}, {"history": -1, "event_log": 0}],
)
def test_server_runtime_restore_requires_complete_valid_offsets(tmp_path, offsets):
    runner = _checkpoint_runner(tmp_path)

    with pytest.raises(ValueError, match="server log offset"):
        runner.checkpoint_store.restore_server_logs(offsets, runner._session_id)

def test_runner_releases_resources_when_experiment_fails():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = None
    calls = []
    runner._run_experiment = lambda verbose: (_ for _ in ()).throw(
        RuntimeError("experiment failed")
    )
    runner._stop_server = lambda: calls.append("server-stop")

    with pytest.raises(RuntimeError, match="experiment failed"):
        runner.run(verbose=False)

    assert calls == ["server-stop"]

def test_stop_server_reaps_process_after_forced_kill(tmp_path):
    calls = []

    class Process:
        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is not None:
                raise runner_module.subprocess.TimeoutExpired("server", timeout)

        def kill(self):
            calls.append("kill")

    server = SimulatorServer(
        run_id="test",
        agent_workspace=tmp_path / "agent_workspace",
        logs_dir=tmp_path / "logs",
        public_dir=tmp_path / "public",
        simulator_llm_config={},
        env_vars={},
    )
    server.process = Process()
    server.port = 12345

    server.stop()

    assert calls == ["terminate", ("wait", 210), "kill", ("wait", None)]
    assert server.process is None
    assert server.port is None


def test_simulator_server_environment_forwards_explicit_llm_config(tmp_path):
    config = {
        "social_model": "small-model",
        "social_api_key_env": "SOCIAL_API_KEY",
    }
    server = SimulatorServer(
        run_id="test",
        agent_workspace=tmp_path / "agent_workspace",
        logs_dir=tmp_path / "logs",
        public_dir=tmp_path / "public",
        simulator_llm_config=config,
        env_vars={"SOCIAL_API_KEY": "secret"},
    )

    environment = server.environment()

    assert environment["NOVAMIND_SERVER_MODE"] == "1"
    assert json.loads(environment["CEOBENCH_SIMULATOR_LLM_CONFIG"]) == config
    assert environment["SOCIAL_API_KEY"] == "secret"

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
    runner.simulator_server = SimpleNamespace(stop=lambda: None)
    runner._load_checkpoint = lambda: None
    monkeypatch.setattr(
        "saas_bench.agents.bash_agent.tools.get_bash_agent_tool_descriptions",
        lambda: [],
    )

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        runner.setup()

    assert runner._session_id is None

def test_run_config_records_git_commit(monkeypatch):
    runner = create_new_runner(
        TEST_CONFIG
    )
    monkeypatch.setattr(runner, "_read_git_commit", lambda: "test-commit")
    payload = runner._run_config_payload()

    assert payload["git_commit"] == "test-commit"


def test_git_read_failure_warns_and_aborts(monkeypatch, capsys):
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        ),
    )

    with pytest.raises(RuntimeError, match="Unable to read Git commit"):
        BashAgentRunner._read_git_commit()
    warning = capsys.readouterr().err
    assert "Unable to read Git commit" in warning
    assert "not a git repository" in warning


def test_dirty_git_worktree_warns_and_aborts(monkeypatch, capsys):
    responses = iter([
        SimpleNamespace(returncode=0, stdout="test-commit\n", stderr=""),
        SimpleNamespace(returncode=0, stdout=" M changed.py\n", stderr=""),
    ])
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        BashAgentRunner._read_git_commit()
    assert "experiment aborted" in capsys.readouterr().err

def test_new_setup_removes_run_directory_when_session_creation_fails(
    tmp_path, monkeypatch
):
    runner = create_new_runner(
        TEST_CONFIG
    )
    runner.git_commit = "test-commit"
    runner.workspace_base = tmp_path / "runs"
    runner.workspace_dir = runner.workspace_base / f"run_{runner.run_id}"
    runner.agent_workspace = runner.workspace_dir / "agent_workspace"
    runner.logs_dir = runner.workspace_dir / "logs"
    runner.trajectory_log_file = runner.logs_dir / f"trajectory_{runner.run_id}.jsonl"
    runner.performance_log_file = runner.logs_dir / f"performance_{runner.run_id}.jsonl"
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
    runner = create_new_runner(
        TEST_CONFIG
    )
    runner.git_commit = "test-commit"
    runner.workspace_base = tmp_path / "runs"
    runner.workspace_dir = runner.workspace_base / f"run_{runner.run_id}"
    runner.agent_workspace = runner.workspace_dir / "agent_workspace"
    runner.logs_dir = runner.workspace_dir / "logs"
    runner.trajectory_log_file = runner.logs_dir / f"trajectory_{runner.run_id}.jsonl"
    runner.performance_log_file = runner.logs_dir / f"performance_{runner.run_id}.jsonl"
    runner._initialize_from_public_repo = lambda: setattr(
        runner, "_session_id", "session-1"
    )
    runner._launch_server_from_prepared_checkpoint = lambda: None
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
    runner.simulator_server = SimpleNamespace(stop=lambda: None)
    runner._load_checkpoint = lambda: {"session_id": "session-1"}
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
