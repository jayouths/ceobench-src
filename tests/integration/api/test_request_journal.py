"""State-changing HTTP requests must be replay-safe and fail closed."""

import json
import fcntl
import pytest
from types import SimpleNamespace

from saas_bench.novamind_api._transport import request_json
from saas_bench.novamind_api._transport import HTTPTransportError
from saas_bench.runtime.api_server import NovaMindAPIServer
from saas_bench.runtime.request_journal import RequestJournal
from saas_bench.runtime.server_entry import cmd_start_server
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
