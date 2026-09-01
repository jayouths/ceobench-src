"""Harness 测试共享的固定值、Fake 和运行目录构造器。"""

import json
from pathlib import Path
from types import SimpleNamespace

from saas_bench.agents.bash_agent.runner import BashAgentRunner
from saas_bench.agents.bash_agent.analysis.pipeline import AnalysisPipeline
from saas_bench.agents.bash_agent.checkpoint import CheckpointStore
from saas_bench.agents.bash_agent.simulator_server import SimulatorServer
from saas_bench.agents.bash_agent.workspace import AgentWorkspaceRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_CONFIG = PROJECT_ROOT / "tests/fixtures/config/base.toml"


def make_analysis_pipeline(
    tmp_path,
    *,
    enabled=True,
    module_config=None,
    model_config=None,
    client=None,
    query_public_rows=None,
    log_trajectory=None,
):
    """构造只包含 Analysis 职责的测试 Pipeline。"""
    return AnalysisPipeline(
        enabled=enabled,
        module_config=module_config or {
            "max_schema_retries": 1,
            "max_enterprise_threads": 50,
        },
        model_config=model_config,
        client=client,
        workspace_dir=tmp_path,
        query_public_rows=query_public_rows or (
            lambda sql: (_ for _ in ()).throw(
                AssertionError(f"unexpected Analysis query: {sql}")
            )
        ),
        log_trajectory=log_trajectory or (lambda *args, **kwargs: None),
    )

EMPTY_ENVIRONMENT_LLM_USAGE = {
    "input_tokens": 0,
    "cached_tokens": 0,
    "output_tokens": 0,
    "cost_by_currency": {},
    "by_purpose": {},
}

EMPTY_ANALYSIS_USAGE = {
    "role_report_days": [],
    "state_portrait_days": [],
    "call_count": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cached_tokens": 0,
    "reasoning_tokens": 0,
    "cost_by_currency": {},
    "by_role": {
        role: {
            "call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "cost_by_currency": {},
        }
        for role in ("market", "finance", "product", "customer")
    },
    "state_reconstruction": {
        "call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cost_by_currency": {},
    },
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
    runner.workspace_dir = tmp_path / "run_fixture"
    runner.workspace_dir.mkdir()
    (runner.workspace_dir / "config.json").write_text(
        json.dumps({"test_config": True})
    )
    runner.agent_workspace = runner.workspace_dir / "agent_workspace"
    runner.agent_workspace.mkdir()
    runner.workspace_repository = AgentWorkspaceRepository(runner.agent_workspace)
    runner.workspace_repository.initialize()
    runner._session_id = "session-1"
    session_dir = runner.agent_workspace / "sessions" / runner._session_id
    session_dir.mkdir(parents=True)
    (session_dir / "world.nmdb").write_bytes(b"persisted-database")
    runner.run_id = "test"
    runner.experiment_name = "test"
    runner.logs_dir = runner.workspace_dir / "logs"
    runner.logs_dir.mkdir()
    runner.trajectory_log_file = runner.logs_dir / "trajectory_test.jsonl"
    runner.performance_log_file = runner.logs_dir / "performance_test.jsonl"
    runner._experiment_log_writer = None
    runner.checkpoint_store = CheckpointStore(
        workspace_dir=runner.workspace_dir,
        agent_workspace=runner.agent_workspace,
        trajectory_log_file=runner.trajectory_log_file,
        performance_log_file=runner.performance_log_file,
        workspace_repository=runner.workspace_repository,
    )
    runner.simulator_server = SimulatorServer(
        run_id=runner.run_id,
        agent_workspace=runner.agent_workspace,
        logs_dir=runner.logs_dir,
        public_dir=PROJECT_ROOT / "public",
        simulator_llm_config={},
        env_vars={},
    )
    runner.model = "model"
    runner.provider = "openai"
    runner.api_type = "openai_responses"
    runner.base_url = None
    runner.reasoning_effort = None
    runner.tool_choice = "required"
    runner.seed = 42
    runner.scenario = "default"
    runner.agent = None
    runner.total_decision_agent_cost_by_currency = {}
    runner.analysis_pipeline = make_analysis_pipeline(runner.workspace_dir)
    return runner
