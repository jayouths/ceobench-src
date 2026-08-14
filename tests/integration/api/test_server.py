"""按职责拆分的 Harness 回归测试。"""

import shutil

import sqlite3

import tempfile

from pathlib import Path

from types import SimpleNamespace

import pytest

from saas_bench.api_server import NovaMindAPIServer, _APIHandler

from saas_bench import _public_cli

from saas_bench.novamind_api import _client as novamind_client

from saas_bench.novamind_api._transport import request_json


from tests.support.harness import (
    EMPTY_ENVIRONMENT_LLM_USAGE,
)

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

def test_api_server_serves_and_cleans_up_unix_socket():
    socket_dir = Path(tempfile.mkdtemp(prefix="ceobench-test-"))
    socket_path = socket_dir / "api.sock"
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=7))

    server.start(unix_socket_path=socket_path)
    try:
        assert request_json("GET", "/health", socket_path=str(socket_path)) == {
            "status": "ok"
        }
        assert request_json("GET", "/vars", socket_path=str(socket_path)) == {
            "current_day": 7
        }
        assert socket_path.stat().st_mode & 0o777 == 0o600
    finally:
        server.stop()
        shutil.rmtree(socket_dir, ignore_errors=True)

    assert not socket_path.exists()

def test_public_cli_and_sdk_prefer_unix_socket(monkeypatch):
    socket_dir = Path(tempfile.mkdtemp(prefix="ceobench-test-"))
    socket_path = socket_dir / "api.sock"
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=11))
    server.start(unix_socket_path=socket_path)
    monkeypatch.setenv("NOVAMIND_API_SOCKET", str(socket_path))
    monkeypatch.setenv("NOVAMIND_API_PORT", "1")
    try:
        assert _public_cli._api_call(1, "GET", "/health") == {"status": "ok"}
        assert novamind_client.get_vars() == {"current_day": 11}
    finally:
        server.stop()
        shutil.rmtree(socket_dir, ignore_errors=True)

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
