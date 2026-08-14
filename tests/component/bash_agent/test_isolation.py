"""按职责拆分的 Harness 回归测试。"""

import os

import shutil

import sqlite3

import tempfile

from pathlib import Path

from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent.tools import (
    BashAgentToolExecutor,
    NextWeekExecutionError,
)

from saas_bench.api_server import NovaMindAPIServer, _APIHandler


from tests.support.harness import (
    PROJECT_ROOT,
)

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

def test_bwrap_mounts_uv_python_alias_target(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    venv_bin = tmp_path / ".venv" / "bin"
    runtime = tmp_path / "uv" / "python" / "cpython-3.13-linux-aarch64-gnu"
    runtime_bin = runtime / "bin"
    workspace.mkdir()
    venv_bin.mkdir(parents=True)
    runtime_bin.mkdir(parents=True)
    (runtime_bin / "python3.13").touch()
    (venv_bin / "python").symlink_to(runtime_bin / "python3.13")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/bwrap")

    executor = BashAgentToolExecutor(workspace)
    command = executor._build_bwrap_cmd(
        "python --version",
        str(workspace),
        {"PATH": f"{venv_bin}:/usr/bin:/bin"},
    )

    mount_pair = ["--ro-bind", str(runtime), str(runtime)]
    assert any(
        command[index:index + len(mount_pair)] == mount_pair
        for index in range(len(command) - len(mount_pair) + 1)
    )

def test_bwrap_exposes_only_agent_api_socket_without_network(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    socket_dir = tmp_path / "socket"
    workspace.mkdir()
    socket_dir.mkdir()
    socket_path = socket_dir / "api.sock"
    socket_path.touch()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/bwrap")

    executor = BashAgentToolExecutor(workspace, api_socket_path=socket_path)
    command = executor._build_bwrap_cmd(
        "./novamind-operation status",
        str(workspace),
        {"PATH": "/usr/bin:/bin", "NOVAMIND_API_SOCKET": str(socket_path)},
    )

    assert "--share-net" not in command
    socket_mount_index = command.index(str(socket_dir))
    assert command[socket_mount_index - 1:socket_mount_index + 2] == [
        "--ro-bind", str(socket_dir), "/run/novamind"
    ]
    socket_env_index = command.index("NOVAMIND_API_SOCKET")
    assert command[socket_env_index + 1] == "/run/novamind/api.sock"

def test_isolated_bash_agent_requires_bubblewrap(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    socket_path = tmp_path / "api.sock"
    socket_path.touch()
    monkeypatch.setattr("shutil.which", lambda name: None)

    executor = BashAgentToolExecutor(workspace, api_socket_path=socket_path)

    with pytest.raises(RuntimeError, match="bubblewrap is required"):
        executor._build_bwrap_cmd("true", str(workspace), {"PATH": "/usr/bin"})

@pytest.mark.slow
@pytest.mark.linux
@pytest.mark.bwrap
@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").exists() or not shutil.which("bwrap"),
    reason="requires Linux and bubblewrap",
)
def test_real_bwrap_allows_only_unix_socket_api(tmp_path):
    workspace = tmp_path / "workspace"
    shutil.copytree(PROJECT_ROOT / "public", workspace)
    socket_dir = Path(tempfile.mkdtemp(prefix="ceobench-test-"))
    socket_path = socket_dir / "api.sock"
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    server = NovaMindAPIServer(
        tools=SimpleNamespace(current_day=13),
        conn=conn,
    )
    server.start(unix_socket_path=socket_path)
    try:
        executor = BashAgentToolExecutor(workspace, api_socket_path=socket_path)

        sdk_result = executor.execute(
            "bash",
            {"command": "./novamind-operation python-c 'import novamind_api as nm; print(nm.vars.current_day)'"},
        )
        cli_result = executor.execute(
            "bash",
            {"command": "./novamind-operation query 'SELECT 1 AS value'"},
        )
        internet_result = executor.execute(
            "bash",
            {"command": "python -c \"import socket; socket.create_connection(('1.1.1.1', 443), 1)\""},
        )
        tcp_result = executor.execute(
            "bash",
            {"command": f"python -c \"import socket; socket.create_connection(('127.0.0.1', {server.port}), 1)\""},
        )
        import_result = executor.execute(
            "bash",
            {"command": "python -c 'import saas_bench'"},
        )
        source_result = executor.execute(
            "bash",
            {"command": f"test -e {PROJECT_ROOT / 'src/saas_bench/simulation.py'}"},
        )
    finally:
        server.stop()
        shutil.rmtree(socket_dir, ignore_errors=True)

    assert sdk_result.strip() == "13"
    assert '"value": 1' in cli_result
    assert "[exit code:" in internet_result
    assert "[exit code:" in tcp_result
    assert "is blocked inside the bash_agent sandbox" in import_result
    assert "[exit code:" in source_result

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
