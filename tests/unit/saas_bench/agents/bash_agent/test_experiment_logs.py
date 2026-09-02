"""实验轨迹与聚合性能日志测试。"""

import json

import pytest

from saas_bench.agents.bash_agent.experiment_logs import ExperimentLogWriter


def _writer(tmp_path):
    return ExperimentLogWriter(
        run_id="run-1",
        trajectory_file=tmp_path / "trajectory_run-1.jsonl",
        performance_file=tmp_path / "performance_run-1.jsonl",
    )


def test_writer_keeps_one_ordered_trajectory_with_explicit_week_boundaries(tmp_path):
    writer = _writer(tmp_path)

    writer.trajectory("week_start", 7, cash=100.0)
    writer.trajectory("dashboard", 7, dashboard="status")
    writer.trajectory("week_end", 7, end_sim_day=14, cash=120.0)

    events = [
        json.loads(line)
        for line in writer.trajectory_file.read_text().splitlines()
    ]
    assert [event["event_type"] for event in events] == [
        "week_start",
        "dashboard",
        "week_end",
    ]
    assert {event["week_index"] for event in events} == {1}
    assert writer.has_trajectory_event("week_start", 7) is True
    assert writer.has_trajectory_event("week_start", 14) is False


def test_performance_log_only_receives_explicit_summaries(tmp_path):
    writer = _writer(tmp_path)

    writer.trajectory("llm_call", 0, component="bash_agent")
    writer.performance("decision_batch", 0, elapsed_seconds=1.5)

    entries = [
        json.loads(line)
        for line in writer.performance_file.read_text().splitlines()
    ]
    assert len(entries) == 1
    assert entries[0]["event_type"] == "decision_batch"
    assert writer.has_performance_event("decision_batch", 0) is True
    assert writer.has_performance_event("week_summary", 0) is False


def test_week_summary_is_rebuilt_from_all_atomic_events_for_the_week(tmp_path):
    writer = _writer(tmp_path)
    writer.trajectory("dashboard", 7, elapsed_seconds=0.25)
    writer.trajectory(
        "llm_call",
        7,
        component="analysis",
        status="completed",
        input_tokens=100,
        output_tokens=20,
        cached_tokens=10,
        reasoning_tokens=5,
        elapsed_seconds=2.0,
        cost_amount=0.01,
        currency="USD",
    )
    writer.trajectory(
        "llm_call",
        7,
        component="bash_agent",
        status="invalid",
        input_tokens=50,
        output_tokens=10,
        cached_tokens=0,
        reasoning_tokens=2,
        elapsed_seconds=1.0,
        cost_amount=0.02,
        currency="USD",
    )
    writer.trajectory(
        "llm_call",
        7,
        component="bash_agent",
        status="valid",
        input_tokens=60,
        output_tokens=12,
        cached_tokens=5,
        reasoning_tokens=3,
        elapsed_seconds=1.5,
        cost_amount=0.03,
        currency="USD",
    )
    writer.trajectory(
        "tool_execution",
        7,
        status="completed",
        elapsed_seconds=0.5,
    )
    writer.trajectory("dashboard", 14, elapsed_seconds=99.0)

    summary = writer.summarize_week(7)

    assert summary["dashboard_seconds"] == pytest.approx(0.25)
    assert summary["tools"] == {
        "call_count": 1,
        "completed_count": 1,
        "error_count": 0,
        "elapsed_seconds": pytest.approx(0.5),
    }
    assert summary["modules"]["analysis"]["call_count"] == 1
    assert summary["modules"]["analysis"]["accepted_count"] == 0
    decision = summary["modules"]["bash_agent"]
    assert decision["call_count"] == 2
    assert decision["completed_count"] == 2
    assert decision["accepted_count"] == 1
    assert decision["invalid_count"] == 1
    assert decision["input_tokens"] == 110
    assert decision["cost_by_currency"]["USD"] == pytest.approx(0.05)


def test_week_summary_keeps_unreported_reasoning_tokens_unknown(tmp_path):
    writer = _writer(tmp_path)
    writer.trajectory(
        "llm_call",
        7,
        component="bash_agent",
        status="valid",
        input_tokens=10,
        output_tokens=3,
        cached_tokens=0,
        reasoning_tokens=None,
        elapsed_seconds=1.0,
    )

    summary = writer.summarize_week(7)

    assert summary["modules"]["bash_agent"]["reasoning_tokens"] is None
