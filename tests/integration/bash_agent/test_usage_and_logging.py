"""按职责拆分的 Harness 回归测试。"""

import json

import sqlite3

import pytest

from saas_bench.agents.bash_agent.run_test import BashAgentRunner, _resume_runner

from saas_bench.database import add_api_cost, get_api_usage_summary

from saas_bench.event_logger import EventLogger

from tests.support.harness import EMPTY_ANALYSIS_USAGE


def test_environment_llm_usage_is_summarized_by_purpose():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE api_costs (
            day INTEGER NOT NULL,
            model TEXT NOT NULL,
            purpose TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cost_amount REAL NOT NULL,
            currency TEXT NOT NULL
        )
    """)
    add_api_cost(conn, 7, "social", "customer_social_post", 11, 4, 7, 0.01, "CNY")
    add_api_cost(conn, 14, "social", "customer_social_post", 13, 5, 5, 0.02, "CNY")
    add_api_cost(conn, 14, "social", "macro_social_post", 17, 6, 3, 0.03, "CNY")

    assert get_api_usage_summary(conn) == {
        "input_tokens": 41,
        "cached_tokens": 15,
        "output_tokens": 15,
        "cost_by_currency": {"CNY": pytest.approx(0.06)},
        "by_purpose": {
            "customer_social_post": {
                "input_tokens": 24,
                "cached_tokens": 9,
                "output_tokens": 12,
                "cost_by_currency": {"CNY": pytest.approx(0.03)},
            },
            "macro_social_post": {
                "input_tokens": 17,
                "cached_tokens": 6,
                "output_tokens": 3,
                "cost_by_currency": {"CNY": pytest.approx(0.03)},
            },
        },
    }

def test_environment_llm_usage_rejects_inconsistent_totals():
    usage = {
        "input_tokens": 2,
        "cached_tokens": 0,
        "output_tokens": 1,
        "cost_by_currency": {"CNY": 0.01},
        "by_purpose": {
            "customer_social_post": {
                "input_tokens": 1,
                "cached_tokens": 0,
                "output_tokens": 1,
                "cost_by_currency": {"CNY": 0.01},
            }
        },
    }

    with pytest.raises(ValueError, match="input token total"):
        BashAgentRunner._validate_environment_llm_usage(usage)


def test_analysis_usage_is_validated_against_role_and_state_totals():
    usage = {
        **EMPTY_ANALYSIS_USAGE,
        "role_report_days": [0],
        "state_portrait_days": [0],
        "call_count": 5,
        "input_tokens": 140,
        "output_tokens": 50,
        "cached_tokens": 18,
        "reasoning_tokens": 9,
        "cost_by_currency": {"USD": 0.06},
        "by_role": {
            role: {
                "call_count": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "cached_tokens": 2,
                "reasoning_tokens": 1,
                "cost_by_currency": {"USD": 0.01},
            }
            for role in ("market", "finance", "product", "customer")
        },
        "state_reconstruction": {
            "call_count": 1,
            "input_tokens": 100,
            "output_tokens": 30,
            "cached_tokens": 10,
            "reasoning_tokens": 5,
            "cost_by_currency": {"USD": 0.02},
        },
    }

    assert BashAgentRunner._validate_analysis_usage(usage, max_day=0) == usage

    usage["input_tokens"] = 139
    with pytest.raises(ValueError, match="input_tokens total"):
        BashAgentRunner._validate_analysis_usage(usage, max_day=0)

def test_result_includes_environment_llm_usage_from_checkpoint(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.run_id = "test"
    runner.seed = 42
    runner.scenario = "default"
    runner.workspace_dir = tmp_path
    runner._harness_result_fields = lambda: {}
    environment_usage = {
        "input_tokens": 41,
        "cached_tokens": 15,
        "output_tokens": 15,
        "cost_by_currency": {"CNY": 0.06},
        "by_purpose": {
            "customer_social_post": {
                "input_tokens": 24,
                "cached_tokens": 9,
                "output_tokens": 12,
                "cost_by_currency": {"CNY": 0.03},
            },
            "macro_social_post": {
                "input_tokens": 17,
                "cached_tokens": 6,
                "output_tokens": 3,
                "cost_by_currency": {"CNY": 0.03},
            },
        },
    }
    checkpoint = {
        "day": 14,
        "cash": 900_000.0,
        "runtime": {
            "agent": {
                "total_turns": 3,
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_tokens": 10,
                "reasoning_tokens": 5,
                "decision_cost_by_currency": {"CNY": 0.1},
            },
                "environment_llm": environment_usage,
                "analysis": EMPTY_ANALYSIS_USAGE,
        },
    }

    result = runner._result_from_checkpoint(checkpoint, "completed")

    assert result["environment_llm_input_tokens"] == 41
    assert result["environment_llm_output_tokens"] == 15
    assert result["environment_llm_cached_tokens"] == 15
    assert result["environment_llm_cost_by_currency"] == {"CNY": pytest.approx(0.06)}
    assert result["environment_llm_usage_by_purpose"] == environment_usage["by_purpose"]

def test_event_logger_records_each_explicit_event_day(tmp_path):
    logger = EventLogger("days", tmp_path, 42, "default", {})
    logger.log_customer_signup(
        day=3,
        customer_id=1,
        group_id="S1",
        plan="A",
        price=10.0,
        is_enterprise=False,
    )
    logger.log_llm_call(
        day=7,
        purpose="customer_social_post",
        model="test-model",
        input_tokens=10,
        cached_tokens=4,
        output_tokens=5,
        cost_amount=0.25,
        currency="CNY",
    )
    logger.save_incremental()

    events = [json.loads(line) for line in logger.log_file.read_text().splitlines()]
    assert [event["day"] for event in events] == [3, 7]

def test_event_logger_preserves_session_start_time_across_server_restarts(tmp_path):
    logger = EventLogger(
        "start-time",
        tmp_path,
        42,
        "default",
        {},
        start_time="2026-01-02T03:04:05Z",
    )
    logger.log_run_end(day=7, final_cash=100.0, days_run=7, outcome="completed")
    logger.save()

    metadata = json.loads(logger.meta_file.read_text())
    assert metadata["start_time"] == "2026-01-02T03:04:05Z"

def test_event_logger_continues_llm_cost_from_restored_database_total(tmp_path):
    logger = EventLogger(
        "resume-cost",
        tmp_path,
        42,
        "default",
        {},
        starting_llm_cost_by_currency={"CNY": 1.25},
    )
    logger.log_llm_call(
        day=8,
        purpose="customer_negotiation",
        model="test-model",
        input_tokens=10,
        cached_tokens=4,
        output_tokens=5,
        cost_amount=0.75,
        currency="CNY",
    )
    logger.log_run_end(day=8, final_cash=100.0, days_run=8, outcome="completed")
    logger.save()

    metadata = json.loads(logger.meta_file.read_text())
    assert metadata["total_llm_cost_by_currency"] == {"CNY": pytest.approx(2.0)}

def test_event_logger_accepts_structured_agent_action_result(tmp_path):
    logger = EventLogger("structured-result", tmp_path, 42, "default", {})
    logger.log_agent_action(
        day=7,
        tool_name="log_rationale",
        arguments={"rationale": "hold prices"},
        result={"logged": True},
        success=True,
    )
    logger.save_incremental()

    event = json.loads(logger.log_file.read_text())
    assert event["day"] == 7
    assert event["details"]["result"] == {"logged": True}
