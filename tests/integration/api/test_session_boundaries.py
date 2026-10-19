"""单局、写入失败和连接中断不得绕过会话边界。"""

from types import SimpleNamespace
import sqlite3

import pytest

from saas_bench.novamind_api._transport import HTTPTransportError, request_json
from saas_bench.runtime.api_server import NovaMindAPIServer, _APIHandler
from saas_bench.runtime.request_journal import RequestJournal


def test_reset_and_daily_script_endpoints_cannot_change_world(tmp_path, monkeypatch):
    calls = []
    journal = RequestJournal(tmp_path / "requests.jsonl", 0)
    server = NovaMindAPIServer(
        tools=SimpleNamespace(current_day=0),
        simulator=SimpleNamespace(initialize=lambda: calls.append("reset")),
        request_journal=journal,
    )
    monkeypatch.delenv("NOVAMIND_API_SOCKET", raising=False)
    server.start()
    try:
        for method, path, status in [
            ("POST", "/reinitialize", 403),
            ("POST", "/daily-scripts", 404),
            ("GET", "/daily-scripts", 404),
            ("DELETE", "/daily-scripts", 404),
        ]:
            with pytest.raises(HTTPTransportError) as error:
                request_json(method, path, {}, port=server.port)
            assert error.value.status == status
        assert calls == [] and journal.size == 0
    finally:
        server.stop()


@pytest.mark.parametrize("failure_stage", ["tool", "journal"])
def test_failed_mutation_blocks_reads_checkpoint_and_finalization(
    tmp_path, monkeypatch, failure_stage,
):
    journal = RequestJournal(tmp_path / "requests.jsonl", 0)
    actions = []
    finalized = []
    server = NovaMindAPIServer(
        tools=SimpleNamespace(current_day=0), request_journal=journal,
        run_finalize_callback=lambda *args: finalized.append(args),
    )

    def broken_tool():
        actions.append("changed")
        raise RuntimeError("Partial tool mutation")

    def broken_append(_record):
        raise OSError("Journal storage unavailable")

    if failure_stage == "journal":
        monkeypatch.setattr(journal, "_append", broken_append)
    with pytest.raises((RuntimeError, OSError)):
        server.run_request("broken", "call:test", {}, broken_tool)
    assert actions == (["changed"] if failure_stage == "tool" else [])
    assert server.run_request("other", "call:test", {}, lambda: pytest.fail("replayed")) == {
        "success": False, "error": "unrecoverable_session",
    }
    assert server.persist_checkpoint(0)["error"] == "unrecoverable_session"
    assert server.finalize_run("completed")["error"] == "unrecoverable_session"
    assert finalized == []

    monkeypatch.delenv("NOVAMIND_API_SOCKET", raising=False)
    server.start()
    try:
        for method, path, body in [
            ("GET", "/vars", None), ("GET", "/dashboard", None),
            ("GET", "/game-status", None),
            ("POST", "/query", {"sql": "SELECT 1"}),
        ]:
            with pytest.raises(HTTPTransportError) as error:
                request_json(method, path, body, port=server.port, timeout=3)
            assert error.value.status == 409
            assert error.value.payload["error"] == "unrecoverable_session"
        assert request_json("GET", "/request-status/broken", port=server.port)["status"] == (
            "pending" if failure_stage == "tool" else "unknown"
        )
        assert request_json("GET", "/health", port=server.port)["status"] == "ok"
    finally:
        server.stop()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_payload_is_rejected_without_poisoning_session(tmp_path, value):
    journal = RequestJournal(tmp_path / "requests.jsonl", 0)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), request_journal=journal)
    assert server.run_request("invalid", "call:test", {"value": value}, lambda: pytest.fail("invalid action")) == {
        "success": False, "error": "invalid_request_payload",
    }
    assert journal.size == 0
    assert server.run_request("valid", "call:test", {}, lambda: {"success": True})["success"]


def test_request_commits_business_transaction_before_completion(tmp_path, monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE facts (amount REAL)")
    journal = RequestJournal(tmp_path / "requests.jsonl", 0)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn, request_journal=journal)
    append = journal._append

    def check_committed(record):
        if record["status"] == "completed":
            assert not conn.in_transaction
        append(record)

    def action():
        conn.execute("INSERT INTO facts VALUES (17)")
        return {"success": True}

    monkeypatch.setattr(journal, "_append", check_committed)
    try:
        assert server.run_request("write", "call:test", {}, action)["success"]
        assert conn.execute("SELECT amount FROM facts").fetchall() == [(17.0,)]
    finally:
        conn.close()


def test_worker_timeout_exception_is_not_confused_with_week_wait_deadline():
    def model_timeout():
        raise TimeoutError("Environment model call timed out")

    server = NovaMindAPIServer(
        tools=SimpleNamespace(current_day=0),
        simulator=SimpleNamespace(step_week=model_timeout),
    )
    with pytest.raises(TimeoutError, match="Environment model call timed out"):
        server.advance_week()
    assert server._week_advance_failed
    assert not server._step_week_timed_out


@pytest.mark.parametrize("stage", ["headers", "body"])
def test_disconnected_response_does_not_turn_completed_action_into_error(stage):
    handler = _APIHandler.__new__(_APIHandler)
    handler.send_response = lambda _status: None
    handler.send_header = lambda *_args: None

    def disconnected(*_args):
        raise BrokenPipeError("Client disconnected")

    handler.end_headers = disconnected if stage == "headers" else lambda: None
    handler.wfile = SimpleNamespace(write=disconnected)
    handler._send_json({"success": True})
