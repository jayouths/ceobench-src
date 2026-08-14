"""按职责拆分的 Harness 回归测试。"""

import json

import pytest

from saas_bench.agents.bash_agent.run_test import BashAgentRunner, _resume_runner


from tests.support.harness import (
    EMPTY_ENVIRONMENT_LLM_USAGE,
    make_checkpoint_runner as _checkpoint_runner,
)

def test_runner_writes_machine_readable_result_atomically(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path
    result = {
        "run_id": "test",
        "outcome": "timeout",
        "final_cash": 123.0,
        "resumable": True,
    }

    runner._write_result(result)

    assert json.loads((tmp_path / "result.json").read_text()) == result
    assert not (tmp_path / "result.json.tmp").exists()

@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        ({"day": 7, "cash": -1.0, "timed_out": True}, "timeout"),
        ({"day": 7, "cash": -1.0, "timed_out": False}, "bankrupt"),
        ({"day": 7, "cash": 1.0, "timed_out": False}, "completed"),
        ({"day": 0, "cash": 1.0, "timed_out": False}, None),
    ],
)
def test_runner_terminal_outcome_prioritizes_failures(status, outcome):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.total_days = 7

    assert runner._terminal_outcome(status) == outcome

def test_resume_validation_failure_preserves_previous_nonterminal_result(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.continue_from = tmp_path
    runner.workspace_dir = tmp_path
    result_file = tmp_path / "result.json"
    previous_result = {"outcome": "incomplete", "resumable": True}
    result_file.write_text(json.dumps(previous_result))
    runner._load_checkpoint = lambda: (
        _ for _ in ()
    ).throw(ValueError("invalid checkpoint"))

    with pytest.raises(ValueError, match="invalid checkpoint"):
        runner._load_or_rebuild_terminal_result()

    assert json.loads(result_file.read_text()) == previous_result

def test_run_returns_existing_terminal_result_without_starting_resources(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()
    result = runner._result_from_checkpoint(checkpoint, "completed")
    (runner.workspace_dir / "result.json").write_text(json.dumps(result))
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": runner._session_id,
        "status": "completed",
        "current_day": 7,
        "final_cash": 850_000.0,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / f"run_{runner._session_id}.jsonl").write_text(json.dumps({
        "day": 7,
        "event_type": "lifecycle",
        "category": "run_end",
        "details": {"outcome": "completed", "final_cash": 850_000.0},
    }) + "\n")
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "completed",
        "days_run": 7,
        "final_cash": 850_000.0,
    }))
    runner._start_timing_poster = lambda: pytest.fail("must not start timing poster")
    runner._run_experiment = lambda verbose: pytest.fail("must not run experiment")

    assert runner.run(verbose=False) == result

def test_runner_rejects_terminal_result_that_disagrees_with_checkpoint(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()
    result = runner._result_from_checkpoint(checkpoint, "completed")
    result["final_cash"] = 1.0
    (runner.workspace_dir / "result.json").write_text(json.dumps(result))
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "status": "completed",
        "current_day": 7,
        "final_cash": 850_000.0,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / f"run_{runner._session_id}.jsonl").write_text(json.dumps({
        "day": 7,
        "category": "run_end",
        "details": {"outcome": "completed", "final_cash": 850_000.0},
    }) + "\n")
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "completed",
        "days_run": 7,
        "final_cash": 850_000.0,
    }))

    with pytest.raises(RuntimeError, match="authoritative artifacts"):
        runner._load_or_rebuild_terminal_result()

def test_runner_rebuilds_missing_terminal_result_from_consistent_artifacts(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)

    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": runner._session_id,
        "status": "completed",
        "current_day": 7,
        "final_cash": 850_000.0,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    event_log = logs_dir / f"run_{runner._session_id}.jsonl"
    event_log.write_text(json.dumps({
        "day": 7,
        "event_type": "lifecycle",
        "category": "run_end",
        "details": {
            "outcome": "completed",
            "final_cash": 850_000.0,
        },
    }) + "\n")
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "completed",
        "days_run": 7,
        "final_cash": 850_000.0,
    }))

    result = runner._load_or_rebuild_terminal_result()

    assert result["outcome"] == "completed"
    assert result["days_run"] == 7
    assert result["final_cash"] == pytest.approx(850_000.0)
    assert result["resumable"] is False
    assert json.loads((runner.workspace_dir / "result.json").read_text()) == result

def test_runner_rejects_disagreeing_terminal_artifacts(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    runner._save_checkpoint(7)

    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "status": "completed",
        "current_day": 7,
        "final_cash": 850_000.0,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "bankrupt",
        "days_run": 7,
        "final_cash": 850_000.0,
    }))

    with pytest.raises(RuntimeError, match="artifacts disagree"):
        runner._load_or_rebuild_terminal_result()

def test_runner_repairs_interrupted_terminal_finalization(tmp_path):
    runner = _checkpoint_runner(tmp_path)
    runner.continue_from = runner.workspace_dir
    runner.seed = 42
    runner.scenario = "default"
    runner.total_days = 7
    runner._http_post = lambda path, data, timeout: {
        "success": True,
        "persisted_day": 7,
        "checkpoint_cash": 850_000.0,
        "environment_llm_usage": EMPTY_ENVIRONMENT_LLM_USAGE,
        "server_log_offsets": {"history": 0, "event_log": 0},
    }
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    (session_dir / "session.json").write_text(json.dumps({
        "session_id": runner._session_id,
        "status": "running",
        "current_day": 7,
    }))
    logs_dir = session_dir / "logs"
    logs_dir.mkdir()
    event_log = logs_dir / f"run_{runner._session_id}.jsonl"
    event_log.write_text("")
    history_log = session_dir / "history.jsonl"
    history_log.write_text("")
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()

    # 模拟 finalize 在多文件提交之间崩溃，只留下一份终态证据。
    (logs_dir / f"run_{runner._session_id}_meta.json").write_text(json.dumps({
        "outcome": "completed",
        "days_run": 7,
        "final_cash": 850_000.0,
    }))
    event_log.write_text(json.dumps({
        "day": 7,
        "event_type": "lifecycle",
        "category": "run_end",
        "details": {"outcome": "completed", "final_cash": 850_000.0},
    }) + "\n")

    assert runner._load_or_rebuild_terminal_result() is None

    runner._resume_checkpoint = checkpoint
    runner._refresh_public_workspace_artifacts = lambda: None
    runner._launch_server = lambda: None
    runner._http_get = lambda path: {"day": 7}
    runner._launch_server_from_prepared_checkpoint()

    # 恢复严格回到 checkpoint 边界，半完成的 run_end 与 meta 均被清除。
    assert event_log.read_text() == ""
    assert not (logs_dir / f"run_{runner._session_id}_meta.json").exists()
    restored_session_meta = json.loads((session_dir / "session.json").read_text())
    assert restored_session_meta["status"] == "created"

    finalize_calls = []
    runner._http_post = lambda path, data, timeout: (
        finalize_calls.append((path, data, timeout))
        or {"success": True, "outcome": "completed"}
    )
    result = runner._repair_terminal_checkpoint_after_setup()

    assert finalize_calls == [("/finalize-run", {"outcome": "completed"}, 30)]
    assert result["outcome"] == "completed"
    assert result["resumable"] is False
    assert json.loads((runner.workspace_dir / "result.json").read_text()) == result
