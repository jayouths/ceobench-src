"""按职责拆分的 Harness 回归测试。"""

import json

from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent, Message

from saas_bench.agents.bash_agent import run_test

from saas_bench.agents.bash_agent.run_test import BashAgentRunner, _resume_runner


from tests.support.harness import (
    EMPTY_ANALYSIS_USAGE,
    EMPTY_ENVIRONMENT_LLM_USAGE,
    make_checkpoint_runner as _checkpoint_runner,
)

def test_workspace_restore_removes_changes_after_checkpoint(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    tracked = runner.agent_workspace / "MEMORY.md"
    tracked.write_text("checkpoint memory")
    checkpoint_commit = runner._capture_workspace_commit(7)
    tracked.write_text("future memory")
    (runner.agent_workspace / "future.txt").write_text("future")
    ignored_session = runner.agent_workspace / "sessions" / "session-1" / "world.nmdb"
    ignored_session.parent.mkdir(parents=True, exist_ok=True)
    ignored_session.write_bytes(b"database")

    runner._restore_workspace_commit(checkpoint_commit)

    assert tracked.read_text() == "checkpoint memory"
    assert not (runner.agent_workspace / "future.txt").exists()
    assert ignored_session.read_bytes() == b"database"

def test_resume_rebuilds_week_commit_cursor_from_checkpoint_day():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.agent = None
    runner.total_decision_agent_cost_by_currency = {}
    runner._last_committed_week = 0
    checkpoint = {
        "day": 35,
        "runtime": {
            "agent": {
                "total_turns": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "decision_cost_by_currency": {},
            }
        },
    }
    restore_plan = run_test.CheckpointRestorePlan(
        session_id="session-1",
        conversation_payload={},
    )

    runner._restore_agent_state_after_launch(checkpoint, restore_plan)

    assert runner._last_committed_week == 5

def test_week_commit_cursor_does_not_advance_when_git_commit_fails():
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner._last_committed_week = 0
    runner._git_commit_workspace = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("git failed"))

    with pytest.raises(RuntimeError, match="git failed"):
        runner._commit_weeks_up_to(7)

    assert runner._last_committed_week == 0

def test_agent_checkpoint_snapshot_applies_pending_tool_result(tmp_path):
    agent = BashAgent.__new__(BashAgent)
    agent.use_anthropic = False
    agent.conversation = [
        Message(
            role="assistant",
            content=[{
                "type": "function_call",
                "call_id": "call-1",
                "name": "bash",
                "arguments": '{"command":"query"}',
            }],
        )
    ]
    agent._pending_tool_calls = [{"id": "call-1", "name": "bash"}]
    agent.current_day = 7
    agent.turns_today = 2
    agent.total_turns = 2
    snapshot = tmp_path / "conversation.json"

    agent.save_checkpoint_snapshot(
        snapshot,
        resume_conversation=True,
        pending_observation="query result",
    )

    payload = json.loads(snapshot.read_text())
    assert payload["tool_results_applied"] is True
    assert payload["pending_tool_calls"] == []
    assert payload["conversation"][-1] == {
        "role": "tool",
        "content": "query result",
        "tool_calls": None,
        "tool_call_id": "call-1",
        "name": "bash",
    }

def test_restored_midweek_context_does_not_duplicate_first_observation():
    agent = BashAgent.__new__(BashAgent)
    agent.conversation = [Message(role="tool", content="saved tool result")]
    agent._pending_tool_calls = []
    agent.current_day = 7
    agent.turns_today = 3
    agent._last_observation = ""
    agent._skip_next_observation = True
    captured = []
    agent._call_llm = lambda: captured.append(list(agent.conversation)) or SimpleNamespace(
        tool="bash", arguments={"command": "query"}
    )
    agent._save_conversation_snapshot = lambda: None

    agent.act("fresh dashboard must be skipped", 0, False, {"day": 7})

    assert [message.content for message in captured[0]] == ["saved tool result"]
    assert agent._skip_next_observation is False
    assert agent.initial_observation_for_audit is None

def test_day_zero_initializes_chat_context_with_system_prompt():
    agent = BashAgent.__new__(BashAgent)
    agent.use_anthropic = False
    agent.conversation = []
    agent._pending_tool_calls = []
    agent.current_day = 0
    agent.turns_today = 0
    agent._last_observation = ""
    agent._skip_next_observation = False
    agent._get_system_prompt_with_memory = lambda: "system prompt"
    captured = []
    agent._call_llm = lambda: captured.append(list(agent.conversation)) or SimpleNamespace(
        tool="bash", arguments={"command": "query"}
    )
    agent._save_conversation_snapshot = lambda: None

    agent.act("day zero dashboard", 0, False, {"day": 0})

    assert [(message.role, message.content) for message in captured[0]] == [
        ("system", "system prompt"),
        ("user", "day zero dashboard"),
    ]
    assert agent.initial_observation_for_audit == "day zero dashboard"

def test_agent_does_not_hide_memory_read_failure(tmp_path):
    agent = BashAgent.__new__(BashAgent)
    agent.system_prompt = "system prompt"
    agent.workspace_path = tmp_path
    (tmp_path / "MEMORY.md").mkdir()

    with pytest.raises(OSError):
        agent._get_system_prompt_with_memory()

def test_turn_limit_saves_one_resumable_midweek_checkpoint(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = None
    runner.total_days = 7
    runner.max_decision_turns_per_batch = 1
    runner.total_decision_agent_cost_by_currency = {}
    runner.run_id = "turn-limit"
    runner.seed = 42
    runner.scenario = "default"
    runner.model = "test-model"
    runner.pricing_model_map = {}
    runner.workspace_dir = tmp_path
    runner.trajectory_log_file = tmp_path / "logs/trajectory_turn-limit.jsonl"
    runner.performance_log_file = tmp_path / "logs/performance_turn-limit.jsonl"
    runner._experiment_log_writer = None
    runner._performance_queue = None
    runner._pending_decision_context = None
    runner._server_port = 1
    runner.setup = lambda: None
    runner._repair_terminal_checkpoint_after_setup = lambda: None
    runner._get_game_status = lambda: {
        "day": 0,
        "cash": 1_000_000.0,
        "subscribers": 0,
        "timed_out": False,
    }
    runner._get_cash = lambda: 1_000_000.0
    runner.analysis_enabled = False
    runner._get_dashboard_payload = lambda: {"dashboard": "dashboard", "day": 0}
    runner._ensure_analysis_signals = lambda payload: None
    runner._commit_weeks_up_to = lambda day: None
    runner._execute_tool = lambda tool, arguments: "query result"
    runner._http_get = lambda path: {
        "day": 0,
        "cash": 1_000_000.0,
        "subscribers": 0,
        "timed_out": False,
    }
    runner._NextWeekTimeoutError = RuntimeError
    runner._write_result = lambda result: None
    runner._harness_result_fields = lambda: {}
    checkpoint_calls = []

    def save_checkpoint(day, **kwargs):
        checkpoint_calls.append((day, kwargs))
        return {
            "day": day,
            "cash": 1_000_000.0,
            "runtime": {
                "agent": {
                    "total_turns": 1,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "decision_cost_by_currency": {},
                },
                "environment_llm": EMPTY_ENVIRONMENT_LLM_USAGE,
                "analysis": EMPTY_ANALYSIS_USAGE,
            },
        }

    runner._save_checkpoint = save_checkpoint

    agent = SimpleNamespace(
        total_turns=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cached_tokens=0,
        total_reasoning_tokens=0,
        last_serving_model="test-model",
    )

    def act(*args):
        agent.total_turns += 1
        return SimpleNamespace(tool="bash", arguments={"command": "query"})

    agent.act = act
    runner.agent = agent

    result = runner._run_experiment(verbose=False)

    assert result["outcome"] == "incomplete"
    assert result["resumable"] is True
    assert result["total_turns"] == 1
    assert result["decision_agent_input_tokens"] == 0
    assert result["decision_agent_output_tokens"] == 0
    assert result["decision_agent_cached_tokens"] == 0
    assert result["decision_agent_reasoning_tokens"] == 0
    assert result["environment_llm_input_tokens"] == 0
    assert result["environment_llm_output_tokens"] == 0
    assert result["environment_llm_cost_by_currency"] == {}
    assert result["environment_llm_usage_by_purpose"] == {}
    assert result["analysis_role_report_days"] == []
    assert result["analysis_state_portrait_days"] == []
    assert result["analysis_llm_calls"] == 0
    assert result["analysis_cost_by_currency"] == {}
    trajectory_events = [
        json.loads(line)["event_type"]
        for line in runner.trajectory_log_file.read_text().splitlines()
    ]
    performance_events = [
        json.loads(line)["event_type"]
        for line in runner.performance_log_file.read_text().splitlines()
    ]
    assert trajectory_events == [
        "week_start",
        "dashboard",
        "tool_execution",
        "turn_cap_reached",
    ]
    assert performance_events == ["decision_batch", "run_summary"]
    assert checkpoint_calls == [
        (0, {
            "resume_conversation": True,
            "pending_observation": "query result",
        }),
    ]
