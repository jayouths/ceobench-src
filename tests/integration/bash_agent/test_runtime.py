"""按职责拆分的 Harness 回归测试。"""

import pytest

from saas_bench.agents.bash_agent import run_test

from saas_bench.agents.bash_agent.run_test import BashAgentRunner, _resume_runner


from tests.support.harness import (
    PROJECT_ROOT,
    make_checkpoint_runner as _checkpoint_runner,
)

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

def test_runner_releases_resources_when_experiment_fails():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = None
    calls = []
    runner._start_performance_poster = lambda: calls.append("performance-start")
    runner._run_experiment = lambda verbose: (_ for _ in ()).throw(
        RuntimeError("experiment failed")
    )
    runner._stop_server = lambda: calls.append("server-stop")
    runner._stop_performance_poster = lambda: calls.append("performance-stop")

    with pytest.raises(RuntimeError, match="experiment failed"):
        runner.run(verbose=False)

    assert calls == ["performance-start", "server-stop", "performance-stop"]

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
    runner.trajectory_log_file = runner.logs_dir / f"trajectory_{runner.run_id}.jsonl"
    runner.performance_log_file = runner.logs_dir / f"performance_{runner.run_id}.jsonl"
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
    runner.trajectory_log_file = runner.logs_dir / f"trajectory_{runner.run_id}.jsonl"
    runner.performance_log_file = runner.logs_dir / f"performance_{runner.run_id}.jsonl"
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
