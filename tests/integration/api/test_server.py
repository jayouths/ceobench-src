"""按职责拆分的 Harness 回归测试。"""

import shutil

import sqlite3

import tempfile

from pathlib import Path

from types import SimpleNamespace

import pytest

from saas_bench.runtime.api_server import NovaMindAPIServer, _APIHandler
from saas_bench.novamind_api import analytics
from saas_bench.agents.bash_agent.analysis.signals import SignalCollector
from saas_bench.simulator.database import get_mrr
from saas_bench.simulator.public_week_snapshot import (
    build_public_week_snapshot,
    load_public_week_snapshot,
    save_public_week_snapshot,
)
from saas_bench.simulator.simulation import DayResult
from saas_bench.simulator.tools import _PUBLIC_CURRENT_MRR_SQL

from saas_bench.runtime import _public_cli

from saas_bench.novamind_api import _client as novamind_client

from saas_bench.novamind_api._transport import request_json


from tests.support.harness import (
    EMPTY_ENVIRONMENT_LLM_USAGE,
)


def _sample_week_result() -> DayResult:
    return DayResult(
        day=7,
        total_usage=1_080,
        overload=0.125,
        outage=True,
        downtime_minutes=15,
        p95_ms=284.4,
        error_rate=0.004,
        new_subscribers=5,
        new_leads=299,
        individual_cancellations=2,
        enterprise_cancellations=1,
        enterprise_cancelled_seats=12,
        individual_upgrades=1,
        enterprise_upgrades=2,
        individual_downgrades=3,
        enterprise_downgrades=4,
        payments_received=0,
        total_costs=0,
        cash=1_000_000,
        mrr=0,
        new_individual_leads=296,
        new_enterprise_leads=3,
        new_individual_subscribers=5,
        new_enterprise_subscriber_accounts=1,
        new_enterprise_subscribers_seats=20,
        total_individual_subscribers=5,
        total_enterprise_subscribers=1,
        total_enterprise_subscription_seats=20,
    )


def test_targeted_ops_sdk_sends_all_four_scopes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        analytics._client,
        "call",
        lambda tool, args: calls.append((tool, args)) or {"updated": True},
    )

    result = analytics.set_targeted_ops_spend(
        by_group={"E1": 300},
        by_plan={"A": 200},
        by_group_plan={"E2": {"B": 100}},
        by_customer={"42": 50},
    )

    assert result == {"updated": True}
    assert calls == [
        (
            "set_targeted_ops_spend",
            {
                "by_group": {"E1": 300},
                "by_plan": {"A": 200},
                "by_group_plan": {"E2": {"B": 100}},
                "by_customer": {"42": 50},
            },
        )
    ]


def test_targeted_ops_api_dispatches_all_four_scopes():
    calls = []
    tools = SimpleNamespace(
        current_day=0,
        set_targeted_ops_spend=lambda **kwargs: calls.append(kwargs) or "ok",
    )
    server = NovaMindAPIServer(tools=tools)

    result = server.execute_tool(
        "set_targeted_ops_spend",
        {
            "by_group": {"E1": 300},
            "by_plan": {"A": 200},
            "by_group_plan": {"E2": {"B": 100}},
            "by_customer": {"42": 50},
        },
    )

    assert result == "ok"
    assert calls == [
        {
            "by_group": {"E1": 300},
            "by_plan": {"A": 200},
            "by_group_plan": {"E2": {"B": 100}},
            "by_customer": {"42": 50},
        }
    ]

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


def test_dashboard_api_returns_text_and_public_snapshot(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    server = NovaMindAPIServer(
        tools=SimpleNamespace(current_day=0),
        conn=conn,
    )
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_dashboard_get()

    payload, status = responses[0]
    assert status == 200
    assert payload["day"] == 0
    assert payload["dashboard"].startswith("=== Week 0 Dashboard (Day 0) ===")
    assert payload["public_week_snapshot"]["current_state"]["cash"] == 1_000_000
    assert payload["public_week_snapshot"] == (
        server._last_public_week_snapshot.to_dict()
    )


def test_dashboard_api_restores_complete_week_snapshot(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    snapshot = build_public_week_snapshot(
        conn,
        day=7,
        day_result=_sample_week_result(),
        inbox_items=["研发项目已完成"],
    )
    save_public_week_snapshot(conn, snapshot)
    conn.commit()

    # 新服务实例没有任何内存快照，模拟从 Day 7 断点恢复。
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=7), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_dashboard_get()

    payload, status = responses[0]
    assert status == 200
    assert payload["public_week_snapshot"] == snapshot.to_dict()
    assert "Usage: 1,080 units" in payload["dashboard"]
    assert "  • 研发项目已完成" in payload["dashboard"]


@pytest.mark.parametrize("snapshot_day", [None, 14])
def test_dashboard_api_rejects_missing_or_stale_snapshot_after_day_zero(
    make_initialized_sim,
    snapshot_day,
):
    conn, _, _ = make_initialized_sim()
    if snapshot_day is not None:
        save_public_week_snapshot(
            conn,
            build_public_week_snapshot(conn, day=snapshot_day),
        )
        conn.commit()
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=7), conn=conn)
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._send_json = lambda data, status=200: None

    with pytest.raises(RuntimeError, match="[Pp]ublic week snapshot"):
        handler._handle_dashboard_get()


def test_week_advance_commits_public_snapshot_before_checkpoint_callback(
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim()
    tools = SimpleNamespace(current_day=0)
    tools.set_current_day = lambda day: setattr(tools, "current_day", day)
    callback_observations = []

    def day_callback(day, dashboard):
        callback_observations.append(
            (day, dashboard, conn.in_transaction, load_public_week_snapshot(conn))
        )

    server = NovaMindAPIServer(
        tools=tools,
        simulator=SimpleNamespace(step_week=_sample_week_result),
        conn=conn,
        day_callback=day_callback,
    )

    result = server.advance_week()

    assert result["success"] is True
    callback_day, callback_dashboard, in_transaction, persisted_snapshot = (
        callback_observations[0]
    )
    assert callback_day == 7
    assert callback_dashboard == result["dashboard"]
    assert in_transaction is False
    assert persisted_snapshot == server._last_public_week_snapshot


def test_analysis_signal_queries_pass_public_query_policy(
    make_initialized_sim,
    make_agent_tools,
):
    conn, simulator, config = make_initialized_sim(seed=42)
    tools = make_agent_tools(conn, config, seed=42)
    server = NovaMindAPIServer(tools=tools, conn=conn)

    def public_query(sql):
        responses = []
        handler = _APIHandler.__new__(_APIHandler)
        handler.server = SimpleNamespace(_api_server=server)
        handler._read_body = lambda: {"sql": sql}
        handler._send_json = lambda data, status=200: responses.append((data, status))
        handler._handle_query()
        payload, status = responses[0]
        assert status == 200, payload
        assert payload["success"] is True
        assert payload.get("truncated") is not True
        return payload["rows"]

    collector = SignalCollector(public_query)
    day_0 = collector.collect(build_public_week_snapshot(conn, 0))
    result = simulator.step_week()
    day_7 = collector.collect(
        build_public_week_snapshot(conn, 7, result),
        {0: day_0},
    )

    assert day_7.day == 7
    assert day_7.market.effective_leads.individual.current.value == (
        result.new_individual_leads
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "_runtime_state",
        "_eval_subscription_day",
        "_eval_subscription_event",
        "_eval_segment_day",
        "_eval_quality_day",
        "_eval_channel_effectiveness_event",
        "_eval_future_fact",
    ],
)
def test_api_blocks_all_evaluation_fact_tables(
    make_initialized_sim, table_name, monkeypatch
):
    conn, _, _ = make_initialized_sim()
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (secret TEXT)")
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", False)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {"sql": f"SELECT * FROM {table_name}"}
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 403
    assert payload == {
        "success": False,
        "error": f"Table '{table_name}' is not accessible.",
    }


def test_oracle_can_read_evaluation_fact_tables(make_initialized_sim, monkeypatch):
    conn, simulator, _ = make_initialized_sim()
    simulator.step_day()
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", True)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=1), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {
        "sql": "SELECT day, group_id FROM _eval_segment_day ORDER BY group_id"
    }
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 200
    assert payload["success"] is True
    assert payload["row_count"] > 0
    assert payload["columns"] == ["day", "group_id"]


@pytest.mark.parametrize(
    "table_name",
    ["_hidden_legacy_fact", "_hidden_future_fact"],
)
def test_api_blocks_all_hidden_prefix_tables(
    make_initialized_sim, table_name, monkeypatch
):
    conn, _, _ = make_initialized_sim()
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (secret TEXT)")
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", False)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {"sql": f"SELECT * FROM {table_name}"}
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 403
    assert payload == {
        "success": False,
        "error": f"Table '{table_name}' is not accessible.",
    }


@pytest.mark.parametrize(
    ("table_name", "sql"),
    [
        ("_rng_states", "SELECT COUNT(*) FROM _rng_states"),
        (
            "global_drift_state",
            (
                "WITH drift AS (SELECT global_q_bias_total AS leaked "
                "FROM global_drift_state) SELECT AVG(leaked) FROM drift"
            ),
        ),
        (
            "group_insight_snapshots",
            (
                "SELECT AVG(snapshot_c_max) AS real_budget "
                "FROM group_insight_snapshots"
            ),
        ),
    ],
)
def test_api_blocks_every_undocumented_internal_table(
    make_initialized_sim, monkeypatch, table_name, sql
):
    conn, simulator, _ = make_initialized_sim()
    simulator.save_rng_states()
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", False)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {"sql": sql}
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 403
    assert payload == {
        "success": False,
        "error": f"Table '{table_name}' is not accessible.",
    }


@pytest.mark.parametrize(
    ("sql", "result_column"),
    [
        ("SELECT c_max AS leaked_c_max FROM customers LIMIT 1", "leaked_c_max"),
        ("SELECT AVG(c_max) AS avg_hidden_budget FROM customers", "avg_hidden_budget"),
        (
            "SELECT leaked_c_max FROM "
            "(SELECT c_max AS leaked_c_max FROM customers LIMIT 1)",
            "leaked_c_max",
        ),
        (
            "WITH secret AS (SELECT c_max AS hidden_budget FROM customers LIMIT 1) "
            "SELECT hidden_budget FROM secret",
            "hidden_budget",
        ),
        (
            "SELECT seat_count AS leaked_seats FROM customers LIMIT 1",
            "leaked_seats",
        ),
    ],
)
def test_api_hides_source_columns_through_aliases_and_expressions(
    make_initialized_sim, monkeypatch, sql, result_column
):
    conn, _, _ = make_initialized_sim()
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", False)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {"sql": sql}
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 200
    assert payload["success"] is True
    assert payload["columns"] == [result_column]
    assert payload["rows"] == [{result_column: None}]
    # 查询结束后必须移除授权器，模拟器仍可读取自己的内部状态。
    assert conn.execute("SELECT c_max FROM customers LIMIT 1").fetchone()[0] is not None


def test_api_select_star_keeps_public_columns_and_removes_hidden_columns(
    make_initialized_sim, monkeypatch
):
    conn, _, _ = make_initialized_sim()
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", False)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {"sql": "SELECT customers.* FROM customers LIMIT 1"}
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 200
    assert payload["success"] is True
    assert "customer_id" in payload["columns"]
    assert "c_max" not in payload["columns"]
    assert "seat_count" not in payload["columns"]
    assert "c_max" not in payload["rows"][0]
    assert "seat_count" not in payload["rows"][0]


def test_public_mrr_query_counts_enterprise_subscription_seats(
    make_initialized_sim, monkeypatch
):
    conn, simulator, _ = make_initialized_sim()
    individual_id = simulator._create_customer(
        simulator._generate_customer_from_group("S1")
    )
    enterprise_id = simulator._create_customer(
        simulator._generate_customer_from_group("E1")
    )
    conn.execute(
        "UPDATE customers SET customer_type = 'large', seat_count = 37.8 "
        "WHERE customer_id = ?",
        (enterprise_id,),
    )
    conn.executemany(
        """
        INSERT INTO subscriptions (
            customer_id, plan, listed_price, effective_price,
            start_day, status, billing_day_mod30, seat_count
        ) VALUES (?, ?, ?, ?, 0, 'subscribed', 0, ?)
        """,
        [
            (individual_id, "A", 20.0, 20.0, 1),
            (enterprise_id, "B", 80.0, 80.0, 10),
        ],
    )
    conn.commit()

    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", False)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {"sql": _PUBLIC_CURRENT_MRR_SQL}
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 200
    assert payload["success"] is True
    # 个人订阅按账户计价，企业订阅按公开的订阅席位快照计价。
    assert payload["rows"] == [{"mrr": 20.0 + 80.0 * 10}]
    assert get_mrr(conn) == 20.0 + 80.0 * 10
    assert (
        build_public_week_snapshot(conn, 0).current_state.enterprise_subscribed_seats
        == 10
    )


def test_oracle_can_read_hidden_source_column_with_alias(
    make_initialized_sim, monkeypatch
):
    conn, _, _ = make_initialized_sim()
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", True)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {
        "sql": "SELECT c_max AS real_budget FROM customers LIMIT 1"
    }
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 200
    assert payload["success"] is True
    assert payload["columns"] == ["real_budget"]
    assert payload["rows"][0]["real_budget"] is not None


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE config_history SET price_A = 999 WHERE day = 0 RETURNING price_A",
        (
            "WITH chosen AS (SELECT 1) UPDATE config_history "
            "SET price_A = 999 WHERE day = 0 RETURNING price_A"
        ),
        "DELETE FROM config_history WHERE day = 0 RETURNING day",
        "REPLACE INTO config_history SELECT * FROM config_history WHERE day = 0",
        "CREATE TEMP TABLE leaked_state(value INTEGER)",
        "ATTACH DATABASE ':memory:' AS extra",
        "BEGIN",
    ],
)
def test_api_rejects_all_database_and_connection_changes(
    make_initialized_sim, monkeypatch, sql
):
    conn, _, _ = make_initialized_sim()
    initial_price = conn.execute(
        "SELECT price_A FROM config_history WHERE day = 0"
    ).fetchone()[0]
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", False)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {"sql": sql}
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 403
    assert payload == {
        "success": False,
        "error": "Write queries are not allowed. Use the novamind_api for all actions.",
    }
    assert conn.execute(
        "SELECT price_A FROM config_history WHERE day = 0"
    ).fetchone()[0] == initial_price
    assert conn.in_transaction is False
    assert conn.execute("PRAGMA query_only").fetchone()[0] == 0
    assert [row[1] for row in conn.execute("PRAGMA database_list")] == ["main"]


def test_api_keeps_complex_read_queries_available(make_initialized_sim, monkeypatch):
    conn, _, _ = make_initialized_sim()
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", False)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {
        "sql": (
            "WITH ranked AS ("
            "SELECT group_id, ROW_NUMBER() OVER (ORDER BY customer_id) AS rank "
            "FROM customers"
            ") SELECT group_id, COUNT(*) AS customers, MAX(rank) AS last_rank "
            "FROM ranked GROUP BY group_id ORDER BY group_id"
        )
    }
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 200
    assert payload["success"] is True
    assert payload["columns"] == ["group_id", "customers", "last_rank"]
    assert payload["row_count"] > 0
    assert conn.execute("PRAGMA query_only").fetchone()[0] == 0


def test_oracle_query_is_read_only(make_initialized_sim, monkeypatch):
    conn, _, _ = make_initialized_sim()
    initial_price = conn.execute(
        "SELECT price_A FROM config_history WHERE day = 0"
    ).fetchone()[0]
    monkeypatch.setattr("saas_bench.runtime.api_server._ORACLE_MODE", True)
    server = NovaMindAPIServer(tools=SimpleNamespace(current_day=0), conn=conn)
    responses = []
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=server)
    handler._read_body = lambda: {
        "sql": "UPDATE config_history SET price_A = 999 WHERE day = 0"
    }
    handler._send_json = lambda data, status=200: responses.append((data, status))

    handler._handle_query()

    payload, status = responses[0]
    assert status == 403
    assert payload["success"] is False
    assert conn.execute(
        "SELECT price_A FROM config_history WHERE day = 0"
    ).fetchone()[0] == initial_price
    assert conn.execute("PRAGMA query_only").fetchone()[0] == 0

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
        "saas_bench.simulator.database.save_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )

    with pytest.raises(RuntimeError, match="write failed"):
        server.advance_week(predictions={7: {"cash": 1.0}})

    assert step_calls == []
    assert server._week_advance_failed is True
