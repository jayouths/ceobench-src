"""Harness 测试共享的固定值、Fake 和运行目录构造器。"""

import json
from pathlib import Path
from types import SimpleNamespace

from saas_bench.agents.bash_agent.run_test import BashAgentRunner


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMPTY_ENVIRONMENT_LLM_USAGE = {
    "input_tokens": 0,
    "cached_tokens": 0,
    "output_tokens": 0,
    "cost_by_currency": {},
    "by_purpose": {},
}


class RecordingResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="response",
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


class RecordingOpenAI:
    def __init__(self):
        self.responses = RecordingResponses()


def make_checkpoint_runner(tmp_path):
    """创建只包含断点持久化所需状态的 Runner。"""
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path / "run_test"
    runner.workspace_dir.mkdir()
    (runner.workspace_dir / "config.json").write_text(
        json.dumps({"test_config": True})
    )
    runner.agent_workspace = runner.workspace_dir / "agent_workspace"
    runner.agent_workspace.mkdir()
    runner._git_init_workspace()
    runner._session_id = "session-1"
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    session_dir.mkdir(parents=True)
    (session_dir / "world.nmdb").write_bytes(b"persisted-database")
    runner.run_id = "test"
    runner.logs_dir = runner.workspace_dir / "logs"
    runner.logs_dir.mkdir()
    runner.response_log_file = runner.logs_dir / "raw_responses_test.jsonl"
    runner.timing_log_file = runner.logs_dir / "timing_test.jsonl"
    runner.model = "model"
    runner.provider = "openai"
    runner.api_type = "openai_responses"
    runner.base_url = None
    runner.reasoning_effort = None
    runner.seed = 42
    runner.scenario = "default"
    runner.agent = None
    runner.total_decision_agent_cost_by_currency = {}
    return runner
