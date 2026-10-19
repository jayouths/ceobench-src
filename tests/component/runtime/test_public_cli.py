"""公开 CLI 的启动、传输和本地诊断边界，不启动真实子进程。"""

import http.client
import json
from types import SimpleNamespace

import pytest

from saas_bench.runtime import _public_cli as cli


@pytest.mark.parametrize("status", ["running", "stopped", "completed", "unrecoverable"])
def test_cli_does_not_implicitly_resume_stopped_sessions(tmp_path, monkeypatch, status):
    session = tmp_path / "one"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"status": status}))
    monkeypatch.delenv("NOVAMIND_API_SOCKET", raising=False)
    monkeypatch.delenv("NOVAMIND_API_PORT", raising=False)
    monkeypatch.setattr(cli, "_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_args, **_kw: pytest.fail("implicit restart"))
    with pytest.raises(SystemExit):
        cli._ensure_server_running("one")


@pytest.mark.parametrize("ready", [True, False])
def test_cli_launches_once_and_detects_startup_exit(tmp_path, monkeypatch, ready):
    session = tmp_path / "one"
    session.mkdir()
    (session / "session.json").write_text('{"status":"created"}')
    (session / ".server.port").write_text("999")  # 没有 PID 的旧端口不能当作新服务已就绪。
    launches = []
    monkeypatch.delenv("NOVAMIND_API_SOCKET", raising=False)
    monkeypatch.delenv("NOVAMIND_API_PORT", raising=False)
    monkeypatch.setattr(cli, "_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_base_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_server_cmd_prefix", lambda: ["test-server"])

    def launch(*args, **kwargs):
        launches.append(args)
        assert not (session / ".server.port").exists()
        if ready:
            (session / ".server.port").write_text("1234")
        return SimpleNamespace(poll=lambda: None if ready else 1)

    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    if ready:
        assert cli._ensure_server_running("one") == 1234
    else:
        with pytest.raises(SystemExit):
            cli._ensure_server_running("one")
    assert len(launches) == 1


def test_cli_http_disconnect_reports_request_outcome_unknown(monkeypatch, capsys):
    def disconnected(*args, **kwargs):
        raise http.client.RemoteDisconnected("Disconnected")

    monkeypatch.setattr(cli, "request_json", disconnected)
    with pytest.raises(SystemExit):
        cli._api_call(1, "POST", "/call", {"request_id": "same-request"})
    assert "Request outcome unknown; request_id=same-request" in capsys.readouterr().err


def test_cli_history_does_not_mix_with_host_checkpoint_log(tmp_path, monkeypatch, capsys):
    session = tmp_path / "one"
    session.mkdir()
    host_log = session / "history.jsonl"
    host_log.write_text('{"type":"host-week"}\n')
    monkeypatch.setattr(cli, "_resolve_session", lambda _session: "one")
    monkeypatch.setattr(cli, "_sessions_dir", lambda: tmp_path)
    cli._log_history("one", {"type": "query"})
    cli.cmd_history(SimpleNamespace(session="one", tail=10))
    assert json.loads(capsys.readouterr().out)["history"] == [{"type": "query"}]
    assert host_log.read_text() == '{"type":"host-week"}\n'
