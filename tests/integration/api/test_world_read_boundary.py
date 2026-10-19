"""周推进中途或失败后，不得把半周状态作为正常观察交给 Agent。"""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from saas_bench.novamind_api._transport import HTTPTransportError, request_json
from saas_bench.runtime.api_server import NovaMindAPIServer
from saas_bench.runtime.request_journal import RequestJournal


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_world_reads_reject_partial_week_but_keep_request_status_available(
    tmp_path, make_initialized_sim, make_agent_tools, monkeypatch, outcome,
):
    original, simulator, config = make_initialized_sim(seed=42)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    original.backup(conn)
    original.close()
    simulator.conn = conn
    monkeypatch.delenv("NOVAMIND_API_SOCKET", raising=False)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    real_step_day = simulator.step_day
    real_step_week = simulator.step_week

    def paused_day():
        result = real_step_day()
        if result.day == 1:
            started.set()
            if not release.wait(10):
                raise RuntimeError("Test did not release the simulator worker")
            if outcome == "failure":
                raise RuntimeError("Simulated interruption after a partial week")
        return result

    def interrupted_week():
        try:
            return real_step_week()
        finally:
            finished.set()

    monkeypatch.setattr(simulator, "step_day", paused_day)
    monkeypatch.setattr(simulator, "step_week", interrupted_week)
    journal = RequestJournal(tmp_path / "request-journal.jsonl", 0)
    server = NovaMindAPIServer(
        tools=make_agent_tools(conn, config, seed=42), simulator=simulator,
        conn=conn, request_journal=journal,
    )
    server.STEP_WEEK_TIMEOUT = 0.01 if outcome == "timeout" else 10
    server.start()

    def request(method, path, body=None):
        return request_json(method, path, body, port=server.port, timeout=3)

    def assert_reads_blocked(error):
        for method, path, body in [
            ("GET", "/vars", None),
            ("GET", "/dashboard", None),
            ("GET", "/game-status", None),
            ("POST", "/query", {"sql": "SELECT SUM(amount) AS cash FROM ledger"}),
        ]:
            with pytest.raises(HTTPTransportError) as rejected:
                request(method, path, body)
            assert rejected.value.status == 409
            assert rejected.value.payload == {"success": False, "error": error}
        assert request("GET", "/health") == {"status": "ok"}
        assert request("GET", "/request-status/week") == {
            "request_id": "week", "status": "pending", "result": None,
        }

    def advance():
        try:
            return request("POST", "/next-week", {
                "request_id": "week", "rationale": "Test partial-week isolation",
                "predictions": {
                    key: {"point": 100, "lower": 0, "upper": 200}
                    for key in ("cash_1wk", "cash_4wk", "cash_12wk", "cash_26wk")
                },
            })
        except HTTPTransportError as exc:
            return exc.payload

    try:
        assert request("GET", "/vars") == {"current_day": 0}
        assert request("GET", "/dashboard")["day"] == 0
        assert request("GET", "/game-status")["day"] == 0
        assert request("POST", "/query", {"sql": "SELECT day FROM ledger"})["success"]
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(advance)
            try:
                assert started.wait(3)
                # 真实模拟器已完成一天，但周请求尚未完成，两个日期不能被混为一谈。
                assert simulator.current_day == 1
                assert server.tools.current_day == 0
                if outcome == "timeout":
                    assert pending.result(timeout=5)["error"] == "step_week_timeout"
                    assert not finished.is_set()
                assert_reads_blocked("week_advance_in_progress")
            finally:
                release.set()
            result = pending.result(timeout=5)
        assert finished.wait(3)
        if outcome == "success":
            assert result["success"] is True and result["day"] == 7
            assert request("GET", "/vars") == {"current_day": 7}
            dashboard = request("GET", "/dashboard")
            assert dashboard["day"] == dashboard["public_week_snapshot"]["day"] == 7
            assert request("GET", "/game-status")["day"] == 7
            assert request("POST", "/query", {
                "sql": "SELECT MAX(day) AS day FROM service_day",
            })["rows"] == [{"day": 7}]
            assert request("GET", "/request-status/week") == {
                "request_id": "week", "status": "completed", "result": result,
            }
        else:
            assert result["error"] == ("step_week_timeout" if outcome == "timeout" else "internal_error")
            if outcome == "timeout":
                assert simulator.current_day == 7 and server.tools.current_day == 0
                assert "do not save a new checkpoint" in result["message"]
            assert_reads_blocked(
                "week_advance_in_progress" if outcome == "timeout" else "week_advance_failed"
            )
            assert server.persist_checkpoint(0)["success"] is False
    finally:
        release.set()
        server.stop()
        if started.is_set():
            assert finished.wait(3)
        conn.close()
