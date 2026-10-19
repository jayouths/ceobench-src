"""State-changing HTTP requests must be replay-safe and fail closed."""

import json
import fcntl
import tempfile
import pytest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from saas_bench.novamind_api._transport import request_json
from saas_bench.novamind_api._transport import HTTPTransportError
from saas_bench.novamind_api import _client
from saas_bench.runtime.api_server import NovaMindAPIServer
from saas_bench.runtime.request_journal import RequestJournal
from saas_bench.runtime import server_entry
from saas_bench.runtime import _public_cli
from saas_bench.runtime.server_entry import cmd_start_server
from saas_bench.evaluation.request_audit import audit_run
from tests.support.harness import make_checkpoint_runner


def test_completed_request_reuses_result_across_restart(tmp_path):
    path = tmp_path / "request-journal.jsonl"
    journal = RequestJournal(path, 0)
    calls = []

    def act():
        calls.append(1)
        return {"success": True, "data": {"count": len(calls)}}

    expected = journal.run("r1", "call:set_prices", {"A": 10}, act)
    assert journal.run("r1", "call:set_prices", {"A": 10}, act) == expected
    assert journal.run("r1", "call:set_prices", {"A": 11}, act)["error"] == "request_id_payload_mismatch"
    assert calls == [1]
    assert len(path.read_text().splitlines()) == 2

    # The world snapshot was persisted at this exact journal boundary.
    restored = RequestJournal(path, journal.size)
    assert restored.run("r1", "call:set_prices", {"A": 10}, act) == expected
    assert calls == [1]


def test_unpersisted_or_interrupted_request_blocks_recovery(tmp_path):
    path = tmp_path / "request-journal.jsonl"
    journal = RequestJournal(path, 0)
    with pytest.raises(RuntimeError, match="action failed"):
        journal.run("r1", "next-week", {}, lambda: (_ for _ in ()).throw(RuntimeError("action failed")))
    assert journal.lookup("r1")["status"] == "pending"
    assert journal.run("r2", "call:set_prices", {}, lambda: None)["error"] == "unrecoverable_session"
    with pytest.raises(RuntimeError, match="interrupted request"):
        RequestJournal(path, journal.size)
    with pytest.raises(RuntimeError, match="exceeds persisted world"):
        RequestJournal(path, 0)


def test_http_write_retry_does_not_repeat_action(tmp_path):
    actions = []
    journal = RequestJournal(tmp_path / "request-journal.jsonl", 0)
    tools = SimpleNamespace(
        current_day=0,
        set_prices=lambda args: actions.append(args) or "ok",
    )
    server = NovaMindAPIServer(tools=tools, request_journal=journal)
    server.start()
    try:
        with pytest.raises(HTTPTransportError) as error:
            request_json("POST", "/call", {
                "request_id": "bad", "tool": "set_prices", "args": [],
            }, port=server.port)
        assert error.value.status == 400
        assert journal.size == 0
        body = {"request_id": "r1", "tool": "set_prices", "args": {"A": 10}}
        first = request_json("POST", "/call", body, port=server.port)
        assert request_json("POST", "/call", body, port=server.port) == first
        assert actions == [{"A": 10}]
        assert request_json("GET", "/request-status/r1", port=server.port) == {
            "request_id": "r1", "status": "completed", "result": first,
        }
        records = [json.loads(line) for line in journal.path.read_text().splitlines()]
        assert records[0]["day_before"] == 0
        assert records[1]["day_after"] == 0
        assert records[1]["success"] is True
    finally:
        server.stop()


@pytest.mark.parametrize("transport", ["tcp", "unix"])
@pytest.mark.parametrize("request_id", ["stable", "周 1/决策?cash=1#确认+", "literal%2Fid"])
def test_request_status_preserves_ids_across_http_sdk_and_cli(
    tmp_path, monkeypatch, capsys, transport, request_id,
):
    actions = []
    journal = RequestJournal(tmp_path / "request-journal.jsonl", 0)
    tools = SimpleNamespace(
        current_day=0,
        set_prices=lambda args: actions.append(args) or "ok",
    )
    server = NovaMindAPIServer(tools=tools, request_journal=journal)
    monkeypatch.delenv("NOVAMIND_API_SOCKET", raising=False)
    monkeypatch.delenv("NOVAMIND_API_PORT", raising=False)

    # macOS 的 Unix Socket 路径长度有限，不使用层级较深的 pytest 用例目录。
    with tempfile.TemporaryDirectory(prefix="nm-") as socket_dir:
        socket_path = Path(socket_dir) / "api.sock" if transport == "unix" else None
        server.start(unix_socket_path=socket_path)
        try:
            if socket_path:
                monkeypatch.setenv("NOVAMIND_API_SOCKET", str(socket_path))
            else:
                monkeypatch.setenv("NOVAMIND_API_PORT", str(server.port))

            assert _client.call("set_prices", {"A": 10}, request_id=request_id) == {"output": "ok"}
            persisted_bytes = journal.size
            expected = journal.lookup(request_id)
            assert expected["status"] == "completed"
            assert expected["result"]["success"] is True
            # 编码只作用于 URL；字面量 %2F 仍应是原 ID 的一部分，而不是斜杠。
            assert request_json("GET", f"/request-status/{quote(request_id, safe='')}") == expected
            assert _client.request_status(request_id) == expected
            _public_cli.cmd_request_status(SimpleNamespace(session=None, request_id=request_id))
            assert json.loads(capsys.readouterr().out) == expected
            assert _client.request_status(request_id + "-missing") == {
                "request_id": request_id + "-missing", "status": "unknown",
            }
            assert journal.size == persisted_bytes
            assert actions == [{"A": 10}]
        finally:
            server.stop()


def test_runner_refuses_to_roll_back_confirmed_requests(tmp_path):
    runner = make_checkpoint_runner(tmp_path)
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({"session_id": runner._session_id}))
    checkpoint = runner.checkpoint_store.save(
        day=0, cash=1, session_id=runner._session_id,
        environment_llm_usage={}, analysis_usage={}, agent=None,
        decision_cost_by_currency={},
        server_log_offsets={"history": 0, "event_log": 0},
        request_journal_bytes=0,
    )
    runner.checkpoint_store.preflight(checkpoint)
    (session_dir / "request-journal.jsonl").write_text("new request\n")
    with pytest.raises(ValueError, match="requests occurred after"):
        runner.checkpoint_store.preflight(checkpoint)


def test_second_server_cannot_own_same_session(tmp_path):
    session_dir = tmp_path / "sessions" / "one"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}")
    with (session_dir / ".server.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already has a running server"):
            cmd_start_server(SimpleNamespace(session="one"), tmp_path)


def test_server_refuses_explicitly_unrecoverable_session(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions" / "one"
    session_dir.mkdir(parents=True)
    metadata = {"status": "unrecoverable", "error": "checkpoint_persistence_failed"}
    (session_dir / "session.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(server_entry, "load_session_db", lambda *_args: pytest.fail("unsafe resume"))
    with pytest.raises(RuntimeError, match="checkpoint_persistence_failed"):
        cmd_start_server(SimpleNamespace(session="one"), tmp_path)
    assert json.loads((session_dir / "session.json").read_text()) == metadata


@pytest.mark.parametrize("missing_state", ["all", "shock_manager"])
def test_server_refuses_resume_without_saved_random_state(
    tmp_path, make_initialized_sim, monkeypatch, missing_state,
):
    conn, simulator, config = make_initialized_sim(seed=42)
    simulator.step_day()
    shocks = server_entry.ShockManager(
        conn, simulator.rng,
        server_entry.ScenarioPack(name="Default", description="Balanced scenario"),
    )
    shocks.save_rng_state()
    conn.execute("DELETE FROM _rng_states WHERE name = ?", (missing_state,))
    conn.commit()

    session_dir = tmp_path / "sessions" / "one"
    session_dir.mkdir(parents=True)
    meta = {
        "seed": 42, "total_days": config.total_days,
        "initial_cash": config.initial_cash, "current_day": 1, "created_at": 1,
    }
    metadata = session_dir / "session.json"
    metadata.write_text(json.dumps(meta))
    (session_dir / "world.nmdb").touch()
    monkeypatch.setattr(server_entry, "load_session_db", lambda _path: conn)
    monkeypatch.setattr(server_entry, "_apply_simulator_llm_config", lambda _config: {})
    monkeypatch.setattr(server_entry, "_create_simulator_llm_client", lambda *_args: None)

    def unexpected_start(**_kwargs):
        pytest.fail("A server with missing RNG state must not start")

    monkeypatch.setattr(server_entry, "NovaMindAPIServer", unexpected_start)
    try:
        with pytest.raises(RuntimeError, match="RNG state"):
            cmd_start_server(SimpleNamespace(session="one"), tmp_path)
        assert json.loads(metadata.read_text()) == meta
        assert not (session_dir / ".server.pid").exists()
    finally:
        conn.close()


@pytest.mark.parametrize("state, exited", [("gone", True), ("zombie", True), ("alive", False)])
def test_wait_for_exit_distinguishes_process_states(monkeypatch, state, exited):
    clock = iter([0.0, 0.0, 31.0])
    monkeypatch.setattr(server_entry.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(server_entry.time, "sleep", lambda _seconds: None)

    def check_process(_pid, _signal):
        if state == "gone":
            raise ProcessLookupError

    monkeypatch.setattr(server_entry.os, "kill", check_process)
    monkeypatch.setattr(
        server_entry.Path, "read_text",
        lambda *_args: f'424242 (simulation worker) {"Z" if state == "zombie" else "S"} 0',
    )
    assert server_entry._wait_for_exit(424242) is exited


@pytest.mark.parametrize("stopped", [True, False])
def test_stop_server_only_confirms_actual_exit(tmp_path, monkeypatch, capsys, stopped):
    session_dir = tmp_path / "sessions" / "one"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}")
    (session_dir / ".server.pid").write_text("424242")
    monkeypatch.setattr(server_entry.os, "kill", lambda *_args: None)
    monkeypatch.setattr(server_entry, "_wait_for_exit", lambda _pid: stopped)

    server_entry.cmd_stop_server(SimpleNamespace(session="one"), tmp_path)

    assert json.loads(capsys.readouterr().out) == {
        "success": stopped, "stopped_pid": 424242 if stopped else None,
        "error": None if stopped else "stop_pending",
    }


def test_public_stop_forwards_server_stop_result(tmp_path, monkeypatch, capsys):
    session_dir = tmp_path / "one"
    session_dir.mkdir()
    (session_dir / ".server.pid").write_text("424242")
    monkeypatch.setattr(_public_cli, "_resolve_session", lambda _session: "one")
    monkeypatch.setattr(_public_cli, "_sessions_dir", lambda: tmp_path)
    calls = []
    response = {"success": False, "stopped_pid": None, "error": "stop_pending"}

    def stop(args):
        calls.append(args)
        return SimpleNamespace(stdout=json.dumps(response), stderr="", returncode=0)

    monkeypatch.setattr(_public_cli, "_run_server_cmd", stop)
    _public_cli.cmd_stop(SimpleNamespace(session="one"))
    assert calls == [["stop-server", "--session", "one"]]
    assert json.loads(capsys.readouterr().out) == response


def test_public_stop_reports_externally_managed_server(monkeypatch, capsys):
    monkeypatch.setattr(_public_cli, "_resolve_session", lambda _session: "__env__")
    _public_cli.cmd_stop(SimpleNamespace(session=None))
    assert json.loads(capsys.readouterr().out) == {
        "success": False, "error": "server_managed_externally",
    }


def test_only_one_established_session_per_workspace(tmp_path, monkeypatch, capsys):
    attempts = []

    def create(_args, base):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("initialization failed")
        session_dir = base / "sessions" / "one"
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text("{}")

    monkeypatch.setattr(server_entry, "_create_new_session", create)
    with pytest.raises(RuntimeError, match="initialization failed"):
        server_entry.cmd_new_session(SimpleNamespace(), tmp_path)
    server_entry.cmd_new_session(SimpleNamespace(), tmp_path)
    server_entry.cmd_new_session(SimpleNamespace(), tmp_path)
    assert attempts == [1, 1]
    assert json.loads(capsys.readouterr().out) == {
        "success": False, "error": "one_game_only",
    }


def test_sdk_prints_request_id_before_submission(monkeypatch, capsys):
    def request(method, path, body):
        assert method == "POST" and path == "/call"
        assert json.loads(capsys.readouterr().err) == {"request_id": "stable"}
        assert body["request_id"] == "stable"
        return {"success": True, "data": {"ok": True}}

    monkeypatch.setattr(_client, "_request", request)
    assert _client.call("set_prices", {"A": 10}, request_id="stable") == {"ok": True}


@pytest.mark.parametrize("response", [None, [], {}, {"success": "true"}])
def test_sdk_malformed_write_result_is_unknown(monkeypatch, capsys, response):
    monkeypatch.setattr(_client, "_request", lambda *_args: response)
    with pytest.raises(_client.NovaMindAPIError) as error:
        _client.call("set_prices", {"A": 10}, request_id="stable")
    assert error.value.request_id == "stable" and error.value.outcome == "unknown"
    assert json.loads(capsys.readouterr().err) == {"request_id": "stable"}


def test_sdk_week_timeout_is_not_a_known_failure(monkeypatch):
    monkeypatch.setattr(
        _client, "_request",
        lambda *_args: {"success": False, "error": "step_week_timeout"},
    )
    with pytest.raises(_client.NovaMindAPIError) as error:
        _client.call("set_prices", {"A": 10}, request_id="stable")
    assert error.value.request_id == "stable" and error.value.outcome == "unknown"


def test_cli_5xx_write_is_unknown(monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise HTTPTransportError(500, b'{"success":false,"error":"internal_error"}')

    monkeypatch.setattr(_public_cli, "request_json", fail)
    with pytest.raises(SystemExit):
        _public_cli._api_call(1, "POST", "/next-week", {"request_id": "week"})
    assert "Request outcome unknown; request_id=week" in capsys.readouterr().err


def test_cli_prints_week_request_id_before_submission(monkeypatch, capsys):
    monkeypatch.setattr(_public_cli, "_resolve_session", lambda *_args: "__env__")
    monkeypatch.setattr(_public_cli, "_ensure_server_running", lambda *_args: 0)

    def request(*_args):
        assert json.loads(capsys.readouterr().err) == {"request_id": "week"}
        return {"success": False, "error": "rejected"}

    monkeypatch.setattr(_public_cli, "_api_call", request)
    forecasts = {
        f"cash_{horizon}_{field}": 1.0
        for horizon in ("1wk", "4wk", "12wk", "26wk")
        for field in ("point", "lower", "upper")
    }
    args = SimpleNamespace(session=None, rationale="plan", request_id="week", **forecasts)
    with pytest.raises(SystemExit):
        _public_cli.cmd_next_week(args)


def test_cli_python_inherits_streams_without_inner_timeout(tmp_path, monkeypatch):
    observed = {}

    def run(*_args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_public_cli, "_base_dir", lambda: tmp_path)
    monkeypatch.setattr(_public_cli.subprocess, "run", run)
    monkeypatch.setattr(_public_cli, "_log_history", lambda *_args: None)
    with pytest.raises(SystemExit) as exit_result:
        _public_cli._execute_python("one", 1, "print('ok')")
    assert exit_result.value.code == 0
    assert "capture_output" not in observed and "timeout" not in observed


def test_host_audit_distinguishes_execution_world_and_runner(tmp_path):
    run_dir = tmp_path / "run"
    session_dir = run_dir / "agent_workspace" / "sessions" / "one"
    session_dir.mkdir(parents=True)
    journal = RequestJournal(session_dir / "request-journal.jsonl", 0)
    journal.run("ok", "call:set_prices", {"A": 10},
                lambda: {"success": True, "data": {"PRIVATE": "secret"}},
                day_reader=lambda: 0)
    first_boundary = journal.size
    journal.run("rejected", "call:set_prices", {"A": 11},
                lambda: {"success": False, "error": "PRIVATE_ERROR"},
                day_reader=lambda: 0)
    with pytest.raises(RuntimeError, match="interrupted"):
        journal.run("unknown", "next-week", {},
                    lambda: (_ for _ in ()).throw(RuntimeError("interrupted")))
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": "one", "request_journal_bytes": first_boundary,
    }))
    (run_dir / "checkpoint.json").write_text(json.dumps({
        "session_id": "one", "runtime": {"request_journal_bytes": 0},
    }))

    audit = audit_run(run_dir)
    assert [(a["status"], a["success"], a["world_snapshot_covered"],
             a["runner_checkpoint_covered"]) for a in audit["actions"]] == [
        ("completed", True, True, False),
        ("completed", False, False, False),
        ("pending", None, False, False),
    ]
    assert audit["actions"][0]["input"] == {"A": 10}
    assert "PRIVATE" not in json.dumps(audit)
    assert "PRIVATE_ERROR" not in json.dumps(audit)


def test_host_audit_rejects_torn_log_and_preserves_old_unknown_fields(tmp_path):
    session_dir = tmp_path / "agent_workspace" / "sessions" / "one"
    session_dir.mkdir(parents=True)
    journal_path = session_dir / "request-journal.jsonl"
    journal_path.write_text(
        json.dumps({"request_id": "old", "digest": "d", "status": "pending"}) + "\n"
        + json.dumps({"request_id": "old", "digest": "d", "status": "completed",
                      "result": {"success": True}}) + "\n"
    )
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": "one", "request_journal_bytes": journal_path.stat().st_size,
    }))
    action = audit_run(tmp_path)["actions"][0]
    assert action["operation"] is None and action["success"] is None
    assert action["world_snapshot_covered"] is True
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": "one", "request_journal_bytes": 1,
    }))
    with pytest.raises(ValueError, match="Invalid world snapshot"):
        audit_run(tmp_path)
    with journal_path.open("ab") as file:
        file.write(b'{"request_id":"torn"')
    with pytest.raises(ValueError, match="Incomplete request journal"):
        audit_run(tmp_path)
