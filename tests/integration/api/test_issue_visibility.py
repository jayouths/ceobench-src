"""失效工单统一退出公开入口，主库历史与 Oracle 读取不受影响。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlcipher3

from saas_bench.agents.bash_agent.analysis.signal_queries import issue_summary
from saas_bench.evaluation.run_metrics import _issue_metrics
from saas_bench.runtime.api_server import NovaMindAPIServer, _APIHandler
from saas_bench.runtime.db_protection import create_encrypted
from saas_bench.simulator.public_week_snapshot import build_public_week_snapshot


def _populate_issues(conn, simulator):
    active, lost = [simulator._create_customer(simulator._generate_customer_from_group("S1")) for _ in range(2)]
    for cid in (active, lost):
        simulator._create_subscription(cid, "A", 10.0)
    visible = []
    for cid, status, reason in [
        (active, "open", None),
        (active, "resolved", "ops_resolved"),
        (lost, "resolved", "ops_resolved"),
        (lost, "open", None),
    ]:
        issue_id = conn.execute(
            "INSERT INTO issues (customer_id, group_id, open_day, days_open, status, resolved_day, resolution_type) "
            "VALUES (?, 'S1', 2, 3, ?, ?, ?)",
            (cid, status, 6 if status == "resolved" else None, reason),
        ).lastrowid
        if status == "resolved" or cid == active:
            visible.append(issue_id)
    conn.execute("UPDATE subscriptions SET status = 'cancelled', end_day = 7 WHERE customer_id = ?", (lost,))
    # 模拟旧断点残留的开放工单；不能仅依靠新状态转移时的自动关闭。
    conn.execute(
        "INSERT INTO issues (customer_id, group_id, open_day, days_open, status) VALUES (?, 'S1', 1, 20, 'open')",
        (lost,),
    )
    conn.commit()
    return visible


@pytest.fixture(params=["sqlite", "sqlcipher"])
def issue_world(request, make_initialized_sim, tmp_path):
    conn, simulator, _ = make_initialized_sim(seed=42)
    visible = _populate_issues(conn, simulator)
    if request.param == "sqlite":
        yield conn, visible
    else:
        encrypted = create_encrypted(tmp_path / "issues.nmdb", key="issue-visibility-test")
        encrypted.row_factory = sqlcipher3.Row
        try:
            encrypted.executescript("\n".join(conn.iterdump()))
            yield encrypted, visible
        finally:
            encrypted.close()


def _query(conn, sql, monkeypatch, *, oracle=False):
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", oracle)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=7), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {"sql": sql}
    handler._send_json = lambda data, status=200: responses.append((data, status))
    handler._handle_query()
    return responses[0]


def test_public_issue_details_aggregates_and_analysis_agree(issue_world, monkeypatch):
    conn, visible = issue_world
    payload, status = _query(conn, "SELECT issue_id FROM issues ORDER BY issue_id", monkeypatch)
    assert status == 200, payload
    assert payload["rows"] == [{"issue_id": issue_id} for issue_id in visible]
    for sql in [
        "SELECT COUNT(*) AS n FROM issues",
        "WITH facts AS (SELECT * FROM issues) SELECT COUNT(*) AS n FROM facts",
        "SELECT COUNT(*) AS n FROM (SELECT issue_id FROM issues)",
        "SELECT COUNT(*) AS n FROM issues i JOIN customers c ON c.customer_id = i.customer_id",
    ]:
        payload, status = _query(conn, sql, monkeypatch)
        assert status == 200, payload
        assert payload["rows"] == [{"n": 3}]
    payload, status = _query(conn, issue_summary(1, 7, -6, 0), monkeypatch)
    assert status == 200, payload
    assert payload["rows"][0]["open_count"] == 1
    assert payload["rows"][0]["current_resolved"] == 2
    assert build_public_week_snapshot(conn, 7).current_state.open_issues == 1
    assert _issue_metrics(conn)["unresolved_issues"] == 1
    assert conn.execute("SELECT COUNT(*) FROM main.issues").fetchone()[0] == 5
    assert conn.execute("SELECT name FROM sqlite_temp_master").fetchall() == []


@pytest.mark.parametrize("sql", [
    "SELECT * FROM main.issues",
    'SELECT COUNT(*) FROM "main"."ISSUES"',
    "SELECT SUM(days_open) FROM main.issues",
    "WITH issues AS (SELECT * FROM main.issues) SELECT COUNT(*) FROM issues",
    "WITH issues AS (SELECT 1 FROM main.issues) SELECT COUNT(*) FROM issues",
    "WITH stolen AS (SELECT * FROM main.issues) SELECT * FROM stolen",
    "SELECT * FROM issues UNION ALL SELECT * FROM main.issues",
])
def test_raw_issue_access_cannot_bypass_projection(issue_world, monkeypatch, sql):
    conn, _ = issue_world
    payload, status = _query(conn, sql, monkeypatch)
    assert status == 403, payload
    assert payload["success"] is False
    assert conn.execute("SELECT COUNT(*) FROM main.issues").fetchone()[0] == 5
    assert conn.execute("SELECT name FROM sqlite_temp_master").fetchall() == []


def test_oracle_retains_all_issue_history(issue_world, monkeypatch):
    conn, _ = issue_world
    for table in ("issues", "main.issues"):
        payload, status = _query(conn, f"SELECT COUNT(*) AS n FROM {table}", monkeypatch, oracle=True)
        assert status == 200, payload
        assert payload["rows"] == [{"n": 5}]


def test_local_python_reader_uses_the_same_issue_projection(make_initialized_sim, make_agent_tools):
    conn, simulator, config = make_initialized_sim(seed=42)
    _populate_issues(conn, simulator)
    tools = make_agent_tools(conn, config)
    tools.db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    result = tools.python_exec("""
print(row('SELECT COUNT(*) FROM issues'))
for sql in ('SELECT COUNT(*) FROM main.issues',
            'WITH issues AS (SELECT * FROM main.issues) SELECT COUNT(*) FROM issues'):
    try:
        print(row(sql))
    except Exception:
        print('blocked')
""", timeout_seconds=30)
    assert result.success, result.message
    assert result.message.splitlines() == ["(3,)", "blocked", "blocked"]
