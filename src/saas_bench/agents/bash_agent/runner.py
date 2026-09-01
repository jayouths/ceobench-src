"""Bash Agent 主实验编排、断点恢复和产物记录。

This script runs a simulation using the bash_agent with any supported LLM provider.
The agent uses bash/file tools and interacts with the simulator via
novamind_api (Python library) and ./novamind-operation (CLI).

The simulation engine runs as a separate subprocess (novamind-server start-server).
The harness communicates with it exclusively via HTTP — no direct DB or simulator
access. This ensures the harness and the public repo have identical interfaces.

All configured endpoints use the OpenAI SDK and an explicit API protocol.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time as _time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add package to path
package_root = Path(__file__).parent.parent.parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from saas_bench.experiment.llm_provider import (
    create_llm_client,
    model_token_cost,
    validate_provider_api_type,
    validate_reasoning_effort,
    validate_tool_choice,
)
from saas_bench.experiment.experiment_config import validate_experiment_name

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.analysis.signals import (
    parse_public_week_snapshot,
)
from saas_bench.agents.bash_agent.analysis.pipeline import AnalysisPipeline
from saas_bench.agents.bash_agent.checkpoint import (
    CheckpointRestorePlan,
    CheckpointStore,
)
from saas_bench.agents.bash_agent.experiment_logs import ExperimentLogWriter
from saas_bench.agents.bash_agent.run_config import (
    load_saved_run_config,
)
from saas_bench.agents.bash_agent.simulator_server import SimulatorServer
from saas_bench.agents.bash_agent.workspace import AgentWorkspaceRepository
from saas_bench.experiment.json_io import write_json_atomic

def load_env_file(env_path: Path) -> Dict[str, str]:
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    return env_vars


class BashAgentRunner:
    """Runner for bash_agent with SaaS Bench.

    The simulation runs in a separate subprocess (novamind-server start-server).
    This harness only handles: agent LLM calls, tool execution, timing, and
    checkpoint management. All simulation state is queried via HTTP.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_type: Optional[str] = None,
        base_url: Optional[str] = None,
        seed: int = 42,
        scenario: str = "default",
        total_days: int = 3650,
        initial_cash: float = 1_000_000.0,
        max_decision_turns_per_batch: Optional[int] = None,
        max_invalid_responses_per_turn: Optional[int] = None,
        workspace_base: Optional[Path] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tool_choice: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        timeout_seconds: float = 600.0,
        request_options: Optional[Dict[str, Any]] = None,
        pricing: Optional[Dict[str, Dict[str, Any]]] = None,
        pricing_model_map: Optional[Dict[str, str]] = None,
        api_key_env: Optional[str] = None,
        api_key_required: bool = True,
        simulator_llm_config: Optional[Dict[str, Any]] = None,
        analysis_module_config: Optional[Dict[str, Any]] = None,
        analysis_model_config: Optional[Dict[str, Any]] = None,
        git_commit: Optional[str] = None,
        continue_from: Optional[Path] = None,
        experiment_name: Optional[str] = None,
    ):
        if not model:
            raise ValueError("decision-agent model must be explicitly configured")
        if not provider:
            raise ValueError("decision-agent provider must be explicitly configured")
        if not api_type:
            raise ValueError("decision-agent api_type must be explicitly configured")
        validate_provider_api_type(provider, api_type, "models.decision_agent")
        validate_reasoning_effort(api_type, reasoning_effort, "models.decision_agent")
        validate_tool_choice(tool_choice, "models.decision_agent")
        self.model = model
        self.provider = provider
        self.api_type = api_type
        self.seed = seed
        self.scenario = scenario
        # 实验只能按周推进，因此总天数会向下取整为 7 的倍数。
        self.total_days = (total_days // 7) * 7
        self.initial_cash = initial_cash
        if (
            not isinstance(max_decision_turns_per_batch, int)
            or isinstance(max_decision_turns_per_batch, bool)
            or max_decision_turns_per_batch <= 0
        ):
            raise ValueError(
                "max_decision_turns_per_batch must be explicitly configured as a positive integer"
            )
        self.max_decision_turns_per_batch = max_decision_turns_per_batch
        if (
            not isinstance(max_invalid_responses_per_turn, int)
            or isinstance(max_invalid_responses_per_turn, bool)
            or max_invalid_responses_per_turn <= 0
        ):
            raise ValueError(
                "max_invalid_responses_per_turn must be explicitly configured as a positive integer"
            )
        self.max_invalid_responses_per_turn = max_invalid_responses_per_turn
        # None means "not configured" and omits the API parameter. Explicit
        # values, including "none", must be forwarded unchanged.
        self.reasoning_effort = reasoning_effort
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if top_p is not None and not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if max_output_tokens is None:
            raise ValueError("decision-agent max_output_tokens must be configured")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.temperature = temperature
        self.top_p = top_p
        self.tool_choice = tool_choice
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = float(timeout_seconds)
        self.request_options = dict(request_options or {})
        self.pricing = dict(pricing or {})
        self.pricing_model_map = dict(pricing_model_map or {})
        configured_pricing_model = self.pricing_model_map.get(self.model, self.model)
        if configured_pricing_model not in self.pricing:
            raise ValueError(
                f"decision-agent model {self.model!r} resolves to missing pricing "
                f"model {configured_pricing_model!r}"
            )
        self.total_decision_agent_cost_by_currency: Dict[str, float] = {}
        self.api_key_env = api_key_env
        self.api_key_required = api_key_required
        self.simulator_llm_config = dict(simulator_llm_config or {})
        self.analysis_module_config = dict(analysis_module_config or {
            "enabled": False,
            "max_schema_retries": 0,
            "max_enterprise_threads": 50,
        })
        self.analysis_enabled = self.analysis_module_config.get("enabled") is True
        max_threads = self.analysis_module_config.get("max_enterprise_threads")
        if not isinstance(max_threads, int) or isinstance(max_threads, bool) or max_threads <= 0:
            raise ValueError("analysis max_enterprise_threads must be a positive integer")
        self.analysis_model_config = (
            dict(analysis_model_config) if analysis_model_config is not None else None
        )
        if self.analysis_enabled and self.analysis_model_config is None:
            raise ValueError("analysis model config is required when analysis is enabled")
        self.git_commit = git_commit
        if continue_from:
            current_commit = self._read_git_commit()
            if (
                git_commit is not None
                and current_commit is not None
                and current_commit != git_commit
            ):
                print(
                    "WARNING: Current Git commit differs from the original run "
                    f"({current_commit} != {git_commit}); refusing to resume.",
                    file=sys.stderr,
                    flush=True,
                )
                raise RuntimeError(
                    "Current Git commit differs from the original run"
                )
        self.continue_from = continue_from
        self.experiment_name = validate_experiment_name(experiment_name)
        self._resume_checkpoint: Optional[Dict[str, Any]] = None
        if continue_from:
            self.workspace_dir = Path(continue_from).resolve()
            if not self.workspace_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {self.workspace_dir}")
            self.run_id = load_saved_run_config(self.workspace_dir)['run_id']
            self.workspace_base = self.workspace_dir.parent
        else:
            self.run_id = str(uuid.uuid4())[:8]
            self.workspace_base = (workspace_base or Path("./outputs/runs")).resolve()
            # 目录使用北京时间，便于国内实验人员直接按时间浏览和排序。
            started_at = datetime.now(timezone(timedelta(hours=8)))
            directory_name = (
                f"{started_at:%Y%m%d-%H%M%S}_seed-{self.seed}_{self.run_id}"
            )
            self.workspace_dir = (
                self.workspace_base / self.experiment_name / directory_name
            )

        # Agent working directory (inside the run directory)
        self.agent_workspace = self.workspace_dir / "agent_workspace"
        self.workspace_repository = AgentWorkspaceRepository(self.agent_workspace)

        # Logs directory
        self.logs_dir = self.workspace_dir / "logs"

        # 原子过程与聚合性能分开保存，避免同一次调用在多个文件重复。
        self.trajectory_log_file = self.logs_dir / f"trajectory_{self.run_id}.jsonl"
        self.performance_log_file = self.logs_dir / f"performance_{self.run_id}.jsonl"
        self._experiment_log_writer = None
        self._pending_decision_context: Optional[Dict[str, Any]] = None
        self.checkpoint_store = CheckpointStore(
            workspace_dir=self.workspace_dir,
            agent_workspace=self.agent_workspace,
            trajectory_log_file=self.trajectory_log_file,
            performance_log_file=self.performance_log_file,
            workspace_repository=self.workspace_repository,
        )

        # Load API key
        env_file = Path(__file__).parent.parent.parent.parent.parent / ".env"
        env_vars = load_env_file(env_file)
        self._env_vars = env_vars

        for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION',
                    'AWS_SESSION_TOKEN', 'NMDB_KEY']:
            if key in env_vars and key not in os.environ:
                os.environ[key] = env_vars[key]

        self.simulator_server = SimulatorServer(
            run_id=self.run_id,
            agent_workspace=self.agent_workspace,
            logs_dir=self.logs_dir,
            public_dir=package_root.parent / "public",
            simulator_llm_config=self.simulator_llm_config,
            env_vars=self._env_vars,
        )

        # The .nmdb session database is SQLCipher-encrypted. The engine resolves
        # the key from saas_bench.runtime._embedded_key (committed in the source tree
        # and compiled into the zipapp) or, failing that, the NMDB_KEY env var.
        # Fail fast here only if neither source is available.
        try:
            from saas_bench.runtime.db_protection import _get_key
            _get_key()
        except RuntimeError as exc:
            raise RuntimeError(
                "No SQLCipher key available for the .nmdb session database: "
                "neither saas_bench.runtime._embedded_key nor the NMDB_KEY env var is "
                "set. Restore src/saas_bench/runtime/_embedded_key.py, or set NMDB_KEY "
                "in .env or the environment."
            ) from exc

        self.api_key = (
            env_vars.get(self.api_key_env) or os.environ.get(self.api_key_env)
            if self.api_key_env
            else None
        )

        if not self.api_key and not self.api_key_required:
            self.api_key = "not-required"
        if not self.api_key:
            raise ValueError(f"No API key found for provider {self.provider}")

        self.base_url = base_url

        self.client = create_llm_client(
            provider=self.provider,
            api_type=self.api_type,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
        )
        self.analysis_client = self._create_analysis_client() if self.analysis_enabled else None
        self.analysis_pipeline = AnalysisPipeline(
            enabled=self.analysis_enabled,
            module_config=self.analysis_module_config,
            model_config=self.analysis_model_config,
            client=self.analysis_client,
            workspace_dir=self.workspace_dir,
            query_public_rows=self._query_public_rows,
            log_trajectory=self._log_trajectory,
        )

        # Components (initialized in setup)
        self.agent = None
        self.tool_executor = None
        self._session_id = None

    def _create_analysis_client(self):
        """为 Analysis 创建独立客户端，避免与决策 Agent 配置串用。"""

        config = self.analysis_model_config
        if config is None:
            raise ValueError("analysis model config is required")
        provider = config.get("provider")
        api_type = config.get("api_type")
        model = config.get("model")
        if not all(isinstance(value, str) and value for value in (provider, api_type, model)):
            raise ValueError("analysis provider, api_type, and model must be configured")
        validate_provider_api_type(provider, api_type, "models.analysis")
        validate_reasoning_effort(
            api_type,
            config.get("reasoning_effort"),
            "models.analysis",
        )
        for task, values in config.get("tasks", {}).items():
            validate_reasoning_effort(
                api_type,
                values.get("reasoning_effort", config.get("reasoning_effort")),
                f"models.analysis.tasks.{task}",
            )

        pricing = config.get("pricing") or {}
        pricing_model_map = config.get("pricing_model_map") or {}
        pricing_model = pricing_model_map.get(model, model)
        if pricing_model not in pricing:
            raise ValueError(
                f"analysis model {model!r} resolves to missing pricing model "
                f"{pricing_model!r}"
            )

        api_key_env = config.get("api_key_env")
        api_key = (
            self._env_vars.get(api_key_env) or os.environ.get(api_key_env)
            if api_key_env else None
        )
        api_key_required = config.get("api_key_required", True)
        if not api_key and not api_key_required:
            api_key = "not-required"
        if not api_key:
            raise ValueError(
                f"No API key found for analysis provider {provider!r} "
                f"from environment variable {api_key_env!r}"
            )
        return create_llm_client(
            provider=provider,
            api_type=api_type,
            api_key=api_key,
            base_url=config.get("base_url"),
            timeout_seconds=float(config.get("timeout_seconds", 600.0)),
        )

    # =========================================================================
    # HTTP helpers — all simulation interaction goes through these
    # =========================================================================

    def _http_get(self, path: str, timeout: float = 30) -> Dict:
        return self.simulator_server.get(path, timeout)

    def _http_post(self, path: str, data: Optional[Dict] = None, timeout: float = 1800) -> Dict:
        return self.simulator_server.post(path, data, timeout)

    def _get_game_status(self) -> Dict:
        """Get and validate the authoritative simulator status."""
        status = self._http_get('/game-status')
        day = status.get('day')
        cash = status.get('cash')
        subscribers = status.get('subscribers')
        timed_out = status.get('timed_out')
        week_advance_failed = status.get('week_advance_failed', False)
        if (
            not isinstance(day, int)
            or isinstance(day, bool)
            or day < 0
            or not isinstance(cash, (int, float))
            or isinstance(cash, bool)
            or not math.isfinite(cash)
            or not isinstance(subscribers, int)
            or isinstance(subscribers, bool)
            or subscribers < 0
            or not isinstance(timed_out, bool)
            or not isinstance(week_advance_failed, bool)
        ):
            raise RuntimeError(f"Invalid simulator status: {status!r}")
        if week_advance_failed:
            raise RuntimeError(
                "Simulator week advancement failed; resume from the previous stable checkpoint"
            )
        return status

    def _get_dashboard_payload(self) -> Dict[str, Any]:
        """读取 Dashboard 及其同源结构化经营快照。"""
        result = self._http_get('/dashboard')
        dashboard = result.get('dashboard')
        if not isinstance(dashboard, str) or not dashboard.strip():
            raise RuntimeError(f"Invalid simulator dashboard: {result!r}")
        if self.analysis_enabled:
            snapshot = parse_public_week_snapshot(result.get('public_week_snapshot'))
            if snapshot.day != result.get('day'):
                raise RuntimeError("Dashboard day does not match public week snapshot")
        return result

    def _query_public_rows(self, sql: str) -> list[dict[str, Any]]:
        """通过与 Baseline 相同的受限 `/query` 接口读取公开数据。"""
        result = self._http_post('/query', {'sql': sql}, timeout=180)
        if not result.get('success'):
            raise RuntimeError(f"Analysis public query failed: {result}")
        if result.get('truncated'):
            raise RuntimeError("Analysis public query was truncated")
        rows = result.get('rows')
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RuntimeError(f"Analysis public query returned invalid rows: {result!r}")
        return rows


    def _terminal_outcome(self, status: Dict[str, Any]) -> Optional[str]:
        """Classify one validated simulator status, with failure states first."""
        # 推进超时时使用上一个稳定断点；负现金在任何日期都表示破产。
        if status['timed_out']:
            return 'timeout'
        if status['cash'] < 0:
            return 'bankrupt'
        if status['day'] >= self.total_days:
            return 'completed'
        return None

    # =========================================================================
    # Logging
    # =========================================================================

    def _experiment_logs(self) -> ExperimentLogWriter:
        """延迟创建日志器，便于断点恢复和轻量单元测试共用。"""
        writer = getattr(self, "_experiment_log_writer", None)
        if writer is None:
            writer = ExperimentLogWriter(
                run_id=self.run_id,
                trajectory_file=self.trajectory_log_file,
                performance_file=self.performance_log_file,
            )
            self._experiment_log_writer = writer
        return writer

    def _log_trajectory(self, event_type: str, day: int, **fields: Any) -> None:
        self._experiment_logs().trajectory(event_type, day, **fields)

    def _log_performance(self, event_type: str, day: int, **fields: Any) -> None:
        self._experiment_logs().performance(event_type, day, **fields)

    def _log_analysis_artifacts(self, day: int) -> None:
        """记录 Analysis 产物索引，产物正文仍以独立文件为准。"""
        if self._experiment_logs().has_trajectory_event("analysis_artifacts", day):
            return
        self._log_trajectory(
            "analysis_artifacts",
            day,
            component="analysis",
            signals=str(
                self.analysis_pipeline.signal_path(day).relative_to(self.workspace_dir)
            ),
            role_reports=str(
                self.analysis_pipeline.role_reports_path(day).relative_to(self.workspace_dir)
            ),
            state_portrait=str(
                self.analysis_pipeline.state_portrait_path(day).relative_to(self.workspace_dir)
            ),
            strategy_brief=str(
                self.analysis_pipeline.brief_path(day).relative_to(self.workspace_dir)
            ),
        )

    @staticmethod
    def _decision_response_status(raw_response: Any) -> tuple[str, int, Optional[str]]:
        """根据原始响应判断 Harness 是否会接受这一轮工具调用。"""
        if not isinstance(raw_response, dict):
            return "invalid", 0, "unstructured_response"

        tool_calls: List[Any] = []
        choices = raw_response.get("choices") or []
        if choices:
            message = (choices[0] or {}).get("message") or {}
            tool_calls = message.get("tool_calls") or []
        elif isinstance(raw_response.get("output"), list):
            tool_calls = [
                item
                for item in raw_response["output"]
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]
        elif isinstance(raw_response.get("content"), list):
            tool_calls = [
                item
                for item in raw_response["content"]
                if isinstance(item, dict) and item.get("type") == "tool_use"
            ]

        if not tool_calls:
            return "invalid", 0, "missing_tool_call"
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments", tool_call.get("arguments"))
            if isinstance(arguments, str) and arguments:
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    return "invalid", len(tool_calls), "invalid_tool_arguments"
        return "valid", len(tool_calls), None

    def _log_decision_llm_call(
        self,
        turn: int,
        day: int,
        messages: List[Dict],
        raw_response: Any,
        elapsed_seconds: float = 0.0,
    ) -> None:
        input_tokens = self.agent.last_input_tokens if self.agent else 0
        output_tokens = self.agent.last_output_tokens if self.agent else 0
        cached_tokens = self.agent.last_cached_tokens if self.agent else 0
        reasoning_tokens = self.agent.last_reasoning_tokens if self.agent else 0
        served_model = self.agent.last_serving_model if self.agent else self.model
        cost = model_token_cost(
            served_model,
            input_tokens,
            output_tokens,
            cached_tokens,
            self.pricing,
            self.pricing_model_map,
        )
        self.total_decision_agent_cost_by_currency[cost.currency] = (
            self.total_decision_agent_cost_by_currency.get(cost.currency, 0.0)
            + cost.amount
        )
        initial_observation = (
            getattr(self.agent, "initial_observation_for_audit", None)
            if self.agent else None
        )
        if initial_observation is not None:
            context = self._pending_decision_context
            if context is None or context["decision_observation"] != initial_observation:
                raise RuntimeError("Logged decision observation does not match Agent input")
            # Dashboard 和创新模块产物分字段保存，同时保留真实发送文本。
            self._log_trajectory(
                "decision_observation",
                day,
                dashboard=context["dashboard"],
                strategy_brief=context["strategy_brief"],
                strategy_brief_artifact=context["strategy_brief_artifact"],
                rendered_observation=initial_observation,
            )
            self._pending_decision_context = None

        status, tool_call_count, invalid_reason = self._decision_response_status(
            raw_response
        )
        entry = {
            "component": "bash_agent",
            "react_round": turn,
            "messages_count": len(messages),
            "status": status,
            "returned_tool_call_count": tool_call_count,
            "elapsed_seconds": elapsed_seconds,
            "provider": self.provider,
            "api_type": self.api_type,
            "requested_model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "served_model": served_model,
            "pricing_model": cost.pricing_model,
            "cost_amount": cost.amount,
            "currency": cost.currency,
            "cumulative_cost_by_currency": dict(
                self.total_decision_agent_cost_by_currency
            ),
            "raw_response": raw_response,
        }
        if invalid_reason is not None:
            entry["invalid_reason"] = invalid_reason
        self._log_trajectory("llm_call", day, **entry)

    def _log_tool_execution(
        self,
        *,
        react_round: int,
        day: int,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        elapsed_seconds: float,
        status: str,
    ) -> None:
        self._log_trajectory(
            "tool_execution",
            day,
            component="bash_agent",
            react_round=react_round,
            tool_index=0,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            elapsed_seconds=elapsed_seconds,
            status=status,
        )

    # =========================================================================
    # Workspace setup
    # =========================================================================


    def _initialize_from_public_repo(self):
        """Copy the published layout into the agent workspace and create a session.

        After the zipapp refactor the published repo is just two artifacts:

            novamind-operation    # zipapp (engine + CLI)
            docs/                 # reference material (incl. SDK source)

        Flow:
        1. Copy those two into agent_workspace.
        2. Create a session via the HOST-SIDE zipapp invoked in server mode,
           so the agent never sees simulator bytecode directly.
        3. Return the session metadata.

        public/ must be built first via `uv run python scripts/build_public.py`.
        """
        public_dir = self._public_dir()
        source_cli = self.workspace_repository.install_public_artifacts(public_dir)
        self._session_id = self.simulator_server.create_session(
            source_cli=source_cli,
            total_days=self.total_days,
            seed=self.seed,
            initial_cash=self.initial_cash,
            scenario=self.scenario,
        )
        print(f"  Session created via CLI: {self._session_id}")
        self.workspace_repository.commit("Initial workspace setup (day 0)")

    def _public_dir(self) -> Path:
        """Return the current repository's host-side public bundle.

        The bash_agent runner.py lives at:
            <root>/src/saas_bench/agents/bash_agent/runner.py
        public/ lives at <root>/public/. parent^5 = <root>.
        """
        public_dir = self.simulator_server.public_dir
        if not public_dir.exists():
            raise FileNotFoundError(
                f"public/ directory not found at {public_dir}. "
                f"Run 'uv run python scripts/build_public.py' first."
            )
        return public_dir

    def _launch_server(self):
        """启动 host 侧模拟器服务。"""
        if not self._session_id:
            raise RuntimeError("Cannot launch simulator without a session")
        self.simulator_server.start(self._session_id)

    def _stop_server(self):
        """幂等关闭 host 侧模拟器服务。"""
        self.simulator_server.stop()

    # =========================================================================
    # Checkpoint
    # =========================================================================

    def _save_checkpoint(
        self,
        day: int,
        *,
        resume_conversation: bool = False,
        pending_observation: Optional[str] = None,
    ):
        """Save checkpoint for resume capability."""
        # 先让服务器确认内存数据库已持久化，再生成 checkpoint，避免“新日期 + 旧数据库”。
        persisted = self._http_post(
            '/persist-checkpoint', {"expected_day": day}, timeout=360
        )
        if not persisted.get("success") or persisted.get("persisted_day") != day:
            raise RuntimeError(
                f"Database checkpoint persistence failed for day {day}: {persisted}"
            )
        checkpoint_cash = persisted.get("checkpoint_cash")
        if (
            not isinstance(checkpoint_cash, (int, float))
            or isinstance(checkpoint_cash, bool)
            or not math.isfinite(checkpoint_cash)
        ):
            raise ValueError(
                f"Checkpoint persistence returned invalid cash: {checkpoint_cash!r}"
            )
        server_log_offsets = self.checkpoint_store.validate_server_log_offsets(
            persisted.get("server_log_offsets"), self._session_id
        )
        environment_llm_usage = persisted.get("environment_llm_usage")
        if not isinstance(environment_llm_usage, dict):
            raise ValueError("Checkpoint persistence did not return environment LLM usage")
        analysis_usage = self.analysis_pipeline.usage_summary(day)
        if not self._session_id:
            raise RuntimeError("Cannot save checkpoint without a simulator session")
        # 模型配置由 config.json 唯一管理；断点只保存恢复所需的运行状态。
        return self.checkpoint_store.save(
            day=day,
            cash=float(checkpoint_cash),
            session_id=self._session_id,
            environment_llm_usage=environment_llm_usage,
            analysis_usage=analysis_usage,
            agent=self.agent,
            decision_cost_by_currency=self.total_decision_agent_cost_by_currency,
            server_log_offsets=server_log_offsets,
            resume_conversation=resume_conversation,
            pending_observation=pending_observation,
        )

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load and validate the state required to resume from disk."""
        return self.checkpoint_store.load()

    def _restore_agent_state_after_launch(
        self, checkpoint: Dict, restore_plan: CheckpointRestorePlan
    ):
        """Restore Agent counters and conversation after its client is created."""
        runtime = checkpoint.get('runtime')
        if not isinstance(runtime, dict):
            raise ValueError("Checkpoint lacks exact runtime state and cannot be safely resumed")

        agent_state = runtime['agent']
        if self.agent:
            self.agent.total_turns = agent_state['total_turns']
            self.agent.total_input_tokens = agent_state['input_tokens']
            self.agent.total_output_tokens = agent_state['output_tokens']
            self.agent.total_cached_tokens = agent_state['cached_tokens']
            self.agent.total_reasoning_tokens = agent_state['reasoning_tokens']
        self.total_decision_agent_cost_by_currency = dict(
            agent_state['decision_cost_by_currency']
        )
        # Git 周节点不需要单独持久化，由可信断点日期即可唯一恢复。
        self.workspace_repository.last_committed_week = checkpoint['day'] // 7

        conversation_payload = restore_plan.conversation_payload
        if self.agent:
            self.agent.restore_checkpoint_snapshot(conversation_payload)

    def _launch_server_from_prepared_checkpoint(self):
        """Restore persistent simulator state, then launch the server."""
        if self._resume_checkpoint:
            runtime = self._resume_checkpoint.get('runtime')
            if not isinstance(runtime, dict):
                raise ValueError("Checkpoint lacks exact runtime state and cannot be safely resumed")
            # 预检只读：所有文件、Git commit 和日志边界都通过后才应用恢复。
            restore_plan = self.checkpoint_store.preflight(self._resume_checkpoint)
            self._checkpoint_restore_plan = restore_plan
            self._session_id = restore_plan.session_id
            self.workspace_repository.restore_commit(runtime['workspace_commit'])
            # Git 回退只负责 Agent 产物；静态客户端必须与当前 host 端 bundle 对齐。
            self.workspace_repository.refresh_public_artifacts(self._public_dir())
            self.checkpoint_store.restore_database(self._resume_checkpoint)
            self.checkpoint_store.restore_runner_logs(runtime['runner_log_offsets'])
            # EventLogger 启动后会以 append 模式打开文件，必须在启动前回退。
            self.checkpoint_store.restore_server_logs(
                runtime['server_log_offsets'], self._session_id
            )
            self.analysis_pipeline.prune_artifacts_after(
                self._resume_checkpoint['day'],
                set(
                    self._resume_checkpoint['runtime']['analysis'][
                        'role_report_days'
                    ]
                ),
                set(
                    self._resume_checkpoint['runtime']['analysis'][
                        'state_portrait_days'
                    ]
                ),
            )
        self._launch_server()
        if self._resume_checkpoint:
            expected_day = self._resume_checkpoint['day']
            status = self._http_get('/game-status')
            actual_day = status.get('day')
            if actual_day != expected_day:
                self._stop_server()
                raise RuntimeError(
                    f"Restored server day {actual_day!r} does not match checkpoint day {expected_day}"
                )

    # =========================================================================
    # Setup
    # =========================================================================

    def _create_new_run_directory(self) -> None:
        """Allocate an empty directory owned exclusively by this new run."""
        self.workspace_dir.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(exist_ok=False)
        self.logs_dir.mkdir()

    def _discard_failed_new_run(self) -> None:
        """Remove artifacts that never reached the initial checkpoint."""
        # 新实验在 Day 0 断点落盘前不是可恢复状态，失败后整体回滚。
        shutil.rmtree(self.workspace_dir)
        self._session_id = None
        self.agent = None
        self.tool_executor = None

    def setup(self):
        """Initialize the simulation environment.

        Flow:
        1. Create a new session, or validate and restore an exact checkpoint
        2. Launch the simulator server subprocess
        3. Create the Agent and HTTP tool executor

        The simulator bytecode (_engine/) and server launcher NEVER enter the
        workspace — they stay in public/ on the host side.
        """
        from .tools import get_bash_agent_tool_descriptions, BashAgentToolExecutor, NextWeekTimeoutError
        self._NextWeekTimeoutError = NextWeekTimeoutError

        new_run_directory_created = False
        try:
            # ── Step 1: Create a new session, or validate an exact checkpoint ──
            if not self.continue_from:
                self._create_new_run_directory()
                new_run_directory_created = True
                self._initialize_from_public_repo()
                # 会话创建成功后才提交实验身份，避免留下虚假的可恢复目录。
                self._write_new_run_config()
            else:
                # 恢复只能使用 checkpoint 明确记录的会话，禁止按目录修改时间猜测。
                checkpoint = self._load_checkpoint()
                if checkpoint is None:
                    raise FileNotFoundError(
                        f"Resume checkpoint not found: {self.workspace_dir / 'checkpoint.json'}"
                    )
                self._resume_checkpoint = checkpoint
                self._session_id = checkpoint['session_id']

            if not self._session_id:
                raise RuntimeError("No session ID found. Cannot proceed.")

            # ── Step 2: Launch server subprocess ──
            self._launch_server_from_prepared_checkpoint()

            # ── Step 3: Create tool executor + agent ──
            self.tool_executor = BashAgentToolExecutor(
                workspace_path=self.agent_workspace,
                api_socket_path=self.simulator_server.api_socket_path,
            )

            tool_descriptions = get_bash_agent_tool_descriptions()

            self.agent = BashAgent(
                tool_descriptions=tool_descriptions,
                client=self.client,
                model=self.model,
                api_type=self.api_type,
                max_invalid_responses_per_turn=self.max_invalid_responses_per_turn,
                response_callback=self._log_decision_llm_call,
                reasoning_effort=self.reasoning_effort,
                temperature=self.temperature,
                top_p=self.top_p,
                tool_choice=self.tool_choice,
                max_output_tokens=self.max_output_tokens,
                timeout_seconds=self.timeout_seconds,
                request_options=self.request_options,
                # 推理文本和非法响应已包含在原始 LLM 事件中，不伪装成工具结果。
                tool_result_callback=None,
                workspace_path=self.agent_workspace,
                total_days=self.total_days,
            )

            # 可读快照用于诊断；恢复仍只读取 checkpoint 指向的不可变对话版本。
            self.agent._snapshot_path = (
                self.agent_workspace / "sessions" / self._session_id / "conversation.json"
            )

            if self._resume_checkpoint:
                # Agent 创建后再恢复对话、Token、成本和日志计数。
                restore_plan = getattr(self, '_checkpoint_restore_plan', None)
                if restore_plan is None:
                    raise RuntimeError("Checkpoint was not preflighted before Agent creation")
                self._restore_agent_state_after_launch(
                    self._resume_checkpoint, restore_plan
                )
            else:
                # Day 0 断点是新实验的事务提交点，此后目录才具备完整恢复条件。
                self._save_checkpoint(0)
        except Exception:
            self._stop_server()
            if new_run_directory_created:
                self._discard_failed_new_run()
            raise

    def _run_config_payload(self) -> Dict[str, Any]:
        """Return the immutable experiment identity stored for future resumes."""
        if self.git_commit is None:
            self.git_commit = self._read_git_commit()
        return {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'agent_type': 'bash_agent',
            'model': self.model,
            'provider': self.provider,
            'api_type': self.api_type,
            'base_url': self.base_url,
            'reasoning_effort': self.reasoning_effort,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'tool_choice': self.tool_choice,
            'max_output_tokens': self.max_output_tokens,
            'timeout_seconds': self.timeout_seconds,
            'request_options': self.request_options,
            'pricing': self.pricing,
            'pricing_model_map': self.pricing_model_map,
            'api_key_env': self.api_key_env,
            'api_key_required': self.api_key_required,
            'seed': self.seed,
            'scenario': self.scenario,
            'total_days': self.total_days,
            'initial_cash': self.initial_cash,
            'max_decision_turns_per_batch': self.max_decision_turns_per_batch,
            'max_invalid_responses_per_turn': self.max_invalid_responses_per_turn,
            'simulator_llm': self.simulator_llm_config,
            'analysis_module': self.analysis_module_config,
            'analysis_model': self.analysis_model_config,
            'git_commit': self.git_commit,
        }

    def _write_new_run_config(self) -> None:
        config_file = self.workspace_dir / "config.json"
        if config_file.exists():
            raise FileExistsError(f"New run config already exists: {config_file}")
        # 配置是实验身份，只在新实验启动时提交一次；恢复过程禁止覆盖。
        payload = self._run_config_payload()
        write_json_atomic(config_file, payload)

    @staticmethod
    def _read_git_commit() -> str:
        """读取唯一版本依据；版本不明确时先告警，再中断实验。"""
        repo_root = package_root.parent
        commit_result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if commit_result.returncode != 0 or not commit_result.stdout.strip():
            reason = commit_result.stderr.strip() or (
                f"git rev-parse exited with code {commit_result.returncode}"
            )
            print(
                f"WARNING: Unable to read Git commit: {reason}",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError("Unable to read Git commit")

        status_result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
        )
        if status_result.returncode != 0:
            reason = status_result.stderr.strip() or (
                f"git status exited with code {status_result.returncode}"
            )
            print(
                f"WARNING: Unable to inspect Git working tree status: {reason}",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError("Unable to inspect Git working tree status")
        elif status_result.stdout.strip():
            print(
                "WARNING: Git working tree has uncommitted changes; experiment aborted.",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError("Git working tree has uncommitted changes")
        return commit_result.stdout.strip()

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a bash_agent tool.

        Raises NextWeekTimeoutError if ./novamind-operation next-week times out,
        which triggers run checkpoint + kill in the run loop.
        """
        return self.tool_executor.execute(tool_name, arguments)

    def _write_result(self, result: Dict[str, Any]) -> None:
        """Atomically publish one machine-readable outcome for this invocation."""
        write_json_atomic(self.workspace_dir / "result.json", result)

    def _result_from_checkpoint(
        self, checkpoint: Dict[str, Any], outcome: str
    ) -> Dict[str, Any]:
        agent_state = checkpoint['runtime']['agent']
        environment_state = checkpoint['runtime']['environment_llm']
        analysis_state = checkpoint['runtime']['analysis']
        return {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'scenario': self.scenario,
            'final_cash': checkpoint['cash'],
            'days_run': checkpoint['day'],
            'outcome': outcome,
            'total_turns': agent_state['total_turns'],
            'decision_agent_input_tokens': agent_state['input_tokens'],
            'decision_agent_output_tokens': agent_state['output_tokens'],
            'decision_agent_cached_tokens': agent_state['cached_tokens'],
            'decision_agent_reasoning_tokens': agent_state['reasoning_tokens'],
            'decision_agent_cost_by_currency': agent_state['decision_cost_by_currency'],
            'environment_llm_input_tokens': environment_state['input_tokens'],
            'environment_llm_output_tokens': environment_state['output_tokens'],
            'environment_llm_cached_tokens': environment_state['cached_tokens'],
            'environment_llm_cost_by_currency': environment_state['cost_by_currency'],
            'environment_llm_usage_by_purpose': environment_state['by_purpose'],
            'analysis_role_report_days': analysis_state['role_report_days'],
            'analysis_state_portrait_days': analysis_state['state_portrait_days'],
            'analysis_llm_calls': analysis_state['call_count'],
            'analysis_input_tokens': analysis_state['input_tokens'],
            'analysis_output_tokens': analysis_state['output_tokens'],
            'analysis_cached_tokens': analysis_state['cached_tokens'],
            'analysis_reasoning_tokens': analysis_state['reasoning_tokens'],
            'analysis_cost_by_currency': analysis_state['cost_by_currency'],
            'analysis_usage_by_role': analysis_state['by_role'],
            'analysis_state_reconstruction_usage': analysis_state[
                'state_reconstruction'
            ],
            'resumable': outcome in {'timeout', 'incomplete'},
            'workspace_dir': str(self.workspace_dir),
        }

    def _load_or_rebuild_terminal_result(self) -> Optional[Dict[str, Any]]:
        """Rebuild a finalized result from the authoritative checkpoint."""
        if not self.continue_from:
            return None

        checkpoint = self._load_checkpoint()
        if checkpoint is None:
            return None
        outcome = self._checkpoint_terminal_outcome(checkpoint)
        if outcome is None:
            return None

        session_id = checkpoint['session_id']
        session_dir = self.agent_workspace / "sessions" / session_id
        session_meta_file = session_dir / "session.json"
        if not session_meta_file.is_file():
            return None

        session_meta = json.loads(session_meta_file.read_text())
        if session_meta.get('status') != outcome:
            # checkpoint 已到终态，但模拟器尚未完成 finalize；恢复后重做即可。
            return None

        result = self._result_from_checkpoint(checkpoint, outcome)
        # result.json 是便于分析的索引，丢失或过期时直接由 checkpoint 重建。
        self._write_result(result)
        return result

    def _checkpoint_terminal_outcome(
        self, checkpoint: Dict[str, Any]
    ) -> Optional[str]:
        """Infer whether a durable checkpoint has reached the experiment end."""
        if checkpoint['cash'] < 0:
            return 'bankrupt'
        if checkpoint['day'] >= self.total_days:
            return 'completed'
        return None

    def _repair_terminal_checkpoint_after_setup(self) -> Optional[Dict[str, Any]]:
        """Finish a terminal checkpoint whose prior finalization was interrupted."""
        checkpoint = self._resume_checkpoint
        if checkpoint is None:
            return None
        outcome = self._checkpoint_terminal_outcome(checkpoint)
        if outcome is None:
            return None

        finalized = self._http_post('/finalize-run', {'outcome': outcome}, timeout=30)
        if not finalized.get('success'):
            raise RuntimeError(f"Run finalization repair failed: {finalized}")
        result = self._result_from_checkpoint(checkpoint, outcome)
        self._write_result(result)
        return result

    # =========================================================================
    # Main run loop
    # =========================================================================

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the experiment while owning all subprocess and thread resources."""
        terminal_result = self._load_or_rebuild_terminal_result()
        if terminal_result is not None:
            return terminal_result
        try:
            return self._run_experiment(verbose)
        finally:
            # 主循环任何位置失败，都必须关闭模拟器。
            self._stop_server()

    def _run_experiment(self, verbose: bool = True) -> Dict[str, Any]:
        """Execute the experiment; resource ownership stays in run()."""

        # 准备实验环境
        self.setup()

        repaired_terminal = self._repair_terminal_checkpoint_after_setup()
        if repaired_terminal is not None:
            return repaired_terminal

        if self.continue_from and verbose:
            checkpoint = self._resume_checkpoint
            cash = self._get_game_status()['cash']
            print(f"\n{'='*60}")
            print(f"RESUMING Bash Agent Run at Sim Day {checkpoint['day']}")
            print(f"Run ID: {self.run_id}")
            print(f"Model: {self.model}")
            print(f"Cash balance: ${cash:,.2f}")
            print(f"Workspace: {self.workspace_dir}")
            print(f"{'='*60}\n")
        elif verbose:
            print(f"\n{'='*60}")
            print(f"Starting Bash Agent Run")
            print(f"Run ID: {self.run_id}")
            print(f"Model: {self.model}")
            print(f"Provider: {self.provider}")
            print(f"Seed: {self.seed}")
            print(f"API Server Port: {self.simulator_server.port}")
            print(f"Agent Workspace: {self.agent_workspace}")
            print(f"Workspace: {self.workspace_dir}")
            print(f"{'='*60}\n")

        status = self._get_game_status()
        last_status: Dict[str, Any] = status
        sim_day = status['day']
        game_outcome = self._terminal_outcome(status)

        # 外层表示决策批次，不代表自然日。模拟日期始终以服务端状态为准。
        decision_batch = 0
        while game_outcome is None:
            decision_batch += 1

            batch_started_at = _time.monotonic()

            # 查询模拟器真实状态
            status = self._get_game_status()
            last_status = status
            sim_day = status['day']
            batch_start_day = sim_day

            if not self._experiment_logs().has_trajectory_event("week_start", sim_day):
                self._log_trajectory(
                    "week_start",
                    sim_day,
                    cash=status["cash"],
                    subscribers=status["subscribers"],
                )

            if verbose:
                print(f"\n{'='*40}")
                print(f"DECISION BATCH {decision_batch} (sim day {sim_day})")
                print(f"{'='*40}")

            # Dashboard 文本和 Analysis 使用同一个公开结构化快照。
            _t0 = _time.monotonic()
            dashboard_payload = self._get_dashboard_payload()
            dashboard = dashboard_payload['dashboard']
            _dashboard_elapsed = _time.monotonic() - _t0
            self._log_trajectory(
                "dashboard",
                sim_day,
                dashboard=dashboard,
                elapsed_seconds=round(_dashboard_elapsed, 3),
            )

            _analysis_started = _time.monotonic()
            signals = self.analysis_pipeline.ensure_signals(dashboard_payload)
            role_reports_generated = False
            state_portrait_generated = False
            brief_generated = False
            analysis_brief = None
            if signals is not None:
                role_reports, role_reports_generated = (
                    self.analysis_pipeline.ensure_role_reports(signals)
                )
                if role_reports is None:
                    raise RuntimeError("Analysis role reports were not generated")
                if role_reports_generated:
                    # 四个角色调用完成后先提交断点；状态重构失败时无需重复付费。
                    stable_checkpoint = self._save_checkpoint(sim_day)
                state_portrait, state_portrait_generated = (
                    self.analysis_pipeline.ensure_state_portrait(role_reports)
                )
                if state_portrait is None:
                    raise RuntimeError("Analysis state portrait was not generated")
                analysis_brief, brief_generated = self.analysis_pipeline.ensure_brief(
                    role_reports,
                    state_portrait,
                )
            if self.analysis_enabled:
                if not self._experiment_logs().has_performance_event(
                    "analysis_week", sim_day
                ):
                    self._log_performance(
                        "analysis_week",
                        sim_day,
                        component="analysis",
                        elapsed_seconds=round(
                            _time.monotonic() - _analysis_started, 3
                        ),
                        role_reports_generated=role_reports_generated,
                        state_portrait_generated=state_portrait_generated,
                        brief_generated=brief_generated,
                    )
                self._log_analysis_artifacts(sim_day)
            if state_portrait_generated:
                # 状态画像和本周汇总日志完成后再次提交，形成完整 Analysis 断点。
                stable_checkpoint = self._save_checkpoint(sim_day)

            # Agent Loop：只要本周的决策尚未结束，就持续执行
            observation = self.analysis_pipeline.decision_observation(
                dashboard, analysis_brief
            )
            brief_path = self.analysis_pipeline.brief_path(sim_day)
            self._pending_decision_context = {
                "dashboard": dashboard,
                "strategy_brief": analysis_brief,
                "strategy_brief_artifact": (
                    str(brief_path.relative_to(self.workspace_dir))
                    if analysis_brief is not None and brief_path.is_file()
                    else None
                ),
                "decision_observation": observation,
            }
            info = {'day': sim_day, 'cash': status['cash']}
            turns_in_batch = 0
            week_advanced = False
            batch_llm_s = 0.0
            batch_tool_s = 0.0
            environment_advance_s = 0.0
            batch_input_tokens = 0
            batch_output_tokens = 0
            batch_cached_tokens = 0
            batch_reasoning_tokens = 0
            batch_api_calls = 0

            while (
                not week_advanced
                and turns_in_batch < self.max_decision_turns_per_batch
            ):
                turns_in_batch += 1

                # 将 observation（可能是 DashBoard，也可能是 Tool Use） 传给 LLM，获取下一步的 action
                _before_total_turns = self.agent.total_turns
                _before_input_tokens = self.agent.total_input_tokens
                _before_output_tokens = self.agent.total_output_tokens
                _before_cached_tokens = self.agent.total_cached_tokens
                _before_reasoning_tokens = self.agent.total_reasoning_tokens
                _t0 = _time.monotonic()
                action = self.agent.act(observation, 0, False, info)
                _llm_elapsed = _time.monotonic() - _t0
                _call_count = self.agent.total_turns - _before_total_turns
                _turn_input_tokens = self.agent.total_input_tokens - _before_input_tokens
                _turn_output_tokens = self.agent.total_output_tokens - _before_output_tokens
                _turn_cached_tokens = self.agent.total_cached_tokens - _before_cached_tokens
                _turn_reasoning_tokens = self.agent.total_reasoning_tokens - _before_reasoning_tokens
                batch_api_calls += _call_count
                batch_llm_s += _llm_elapsed
                batch_input_tokens += _turn_input_tokens
                batch_output_tokens += _turn_output_tokens
                batch_cached_tokens += _turn_cached_tokens
                batch_reasoning_tokens += _turn_reasoning_tokens

                # 若 action 为 None，说明 LLM 返回有误，此时直接报错
                if action is None:
                    # With the agent's retry-with-feedback loop, _call_* should no
                    # longer return None. If we still get here, something is very
                    # wrong — raise so the run fails loudly instead of silently
                    # spamming a broken next-week command.
                    raise RuntimeError(
                        "Agent.act() returned None despite retry-with-feedback loop. "
                        "This indicates a bug in the agent scaffold — please investigate."
                    )

                # 解析 action 为工具调用指令
                tool_name = action.tool
                tool_args_preview = ""
                if tool_name == 'bash':
                    tool_args_preview = (action.arguments or {}).get('command', '')[:120]
                else:
                    tool_args_preview = json.dumps(action.arguments or {})[:120]

                if verbose:
                    if tool_name == 'bash':
                        print(f"    [Turn {turns_in_batch}] bash: {tool_args_preview[:100]}")
                    else:
                        print(f"    [Turn {turns_in_batch}] {tool_name}({tool_args_preview[:100]})")

                # 执行工具调用
                day_before_tool = sim_day
                _t0 = _time.monotonic()
                try:
                    result = self._execute_tool(action.tool, action.arguments or {})
                except self._NextWeekTimeoutError as e:
                    _tool_elapsed = _time.monotonic() - _t0
                    self._log_tool_execution(
                        react_round=self.agent.total_turns,
                        day=sim_day,
                        tool_name=action.tool,
                        arguments=action.arguments or {},
                        result={"error": str(e)},
                        elapsed_seconds=round(_tool_elapsed, 3),
                        status="timeout",
                    )
                    print(f"\n⚠️  next_week timed out on sim day {sim_day} ({e})")
                    print("Auto-quitting. Keeping the previous completed checkpoint.")
                    game_outcome = 'timeout'
                    break
                _tool_elapsed = _time.monotonic() - _t0
                batch_tool_s += _tool_elapsed
                observation = result if isinstance(result, str) else json.dumps(result)

                self._log_tool_execution(
                    react_round=self.agent.total_turns,
                    day=sim_day,
                    tool_name=action.tool,
                    arguments=action.arguments or {},
                    result=observation,
                    elapsed_seconds=round(_tool_elapsed, 3),
                    status="completed",
                )

                if verbose:
                    print(f"      → {observation[:200]}")
                    print(f"      ⏱ llm={_llm_elapsed:.1f}s tool={_tool_elapsed:.1f}s")

                # 以服务端日期为唯一依据，不再解析可能变化的 Dashboard 文本格式。
                status = self._get_game_status()
                last_status = status
                sim_day = status['day']
                if sim_day < day_before_tool:
                    raise RuntimeError(
                        f"Simulator day moved backwards: {day_before_tool} -> {sim_day}"
                    )
                week_advanced = sim_day > day_before_tool
                if week_advanced:
                    # next-week 被封装在工具调用内，日期变化才是环境真正推进的信号。
                    environment_advance_s += _tool_elapsed
                # next-week 后，将 Agent 工具目录的改动提交到 Git。
                self.workspace_repository.commit_weeks_up_to(sim_day)

                _cash_inner = status['cash']
                info = {'day': sim_day, 'cash': _cash_inner}
                game_outcome = self._terminal_outcome(status)
                if game_outcome is not None:
                    if verbose:
                        print(
                            f"\nSimulation ended: outcome={game_outcome}, "
                            f"day={sim_day}, cash=${_cash_inner:,.0f}"
                        )
                    break

            if game_outcome == 'timeout':
                break

            terminal_batch = game_outcome in {'completed', 'bankrupt'}
            resumable_batch = not week_advanced and not terminal_batch

            # 达到配置上限仍未推进时保留同一模拟日，不伪造缺少理由和预测参数的 next-week。
            # 当前上下文会随 checkpoint 一起保存，下一轮可以继续决策。
            if resumable_batch:
                print(
                    f"\n⚠️  Turn cap reached on sim day {sim_day} without next-week; "
                    "saving a resumable checkpoint and ending this invocation."
                )
                self._log_trajectory(
                    "turn_cap_reached", sim_day, turns=turns_in_batch
                )

            # 内层每次工具执行后已经校验服务端状态，这里直接复用。
            subscribers = status['subscribers']
            cash = status['cash']

            # 决策批次可能停留在同一模拟日，日志必须按 batch 而不是 day 命名。
            batch_elapsed_s = _time.monotonic() - batch_started_at
            agent_tool_s = max(batch_tool_s - environment_advance_s, 0.0)
            batch_other_s = (
                batch_elapsed_s - batch_llm_s - batch_tool_s - _dashboard_elapsed
            )
            if week_advanced or terminal_batch:
                self._log_trajectory(
                    "week_end",
                    batch_start_day,
                    end_sim_day=sim_day,
                    cash=cash,
                    subscribers=subscribers,
                    outcome=game_outcome,
                )

            self._log_performance(
                "decision_batch",
                batch_start_day,
                decision_batch=decision_batch,
                end_sim_day=sim_day,
                elapsed_seconds=round(batch_elapsed_s, 1),
                llm_seconds=round(batch_llm_s, 1),
                agent_tool_seconds=round(agent_tool_s, 1),
                environment_advance_seconds=round(environment_advance_s, 1),
                dashboard_seconds=round(_dashboard_elapsed, 2),
                other_seconds=round(max(batch_other_s, 0), 1),
                turns=turns_in_batch,
                api_calls=batch_api_calls,
                subs=subscribers,
                cash=cash,
                batch_input_tokens=batch_input_tokens,
                batch_output_tokens=batch_output_tokens,
                batch_cached_tokens=batch_cached_tokens,
                batch_reasoning_tokens=batch_reasoning_tokens,
                total_input_tokens=self.agent.total_input_tokens,
                total_output_tokens=self.agent.total_output_tokens,
                total_cached_tokens=self.agent.total_cached_tokens,
                total_reasoning_tokens=self.agent.total_reasoning_tokens,
            )
            if week_advanced or terminal_batch:
                self._log_performance(
                    "week_summary",
                    batch_start_day,
                    end_sim_day=sim_day,
                    cash=cash,
                    subscribers=subscribers,
                    **self._experiment_logs().summarize_week(batch_start_day),
                )

            pct_llm = (batch_llm_s / batch_elapsed_s * 100) if batch_elapsed_s > 0 else 0
            pct_environment = (environment_advance_s / batch_elapsed_s * 100) if batch_elapsed_s > 0 else 0
            pct_tool = (agent_tool_s / batch_elapsed_s * 100) if batch_elapsed_s > 0 else 0
            cache_pct = (batch_cached_tokens / batch_input_tokens * 100) if batch_input_tokens > 0 else 0
            print(f"\n⏱ BATCH {decision_batch} (DAY {sim_day}): total={batch_elapsed_s:.0f}s | "
                  f"llm={batch_llm_s:.0f}s ({pct_llm:.0f}%) | "
                  f"environment={environment_advance_s:.0f}s ({pct_environment:.0f}%) | "
                  f"tools={agent_tool_s:.0f}s ({pct_tool:.0f}%) | "
                  f"dashboard={_dashboard_elapsed:.1f}s | "
                  f"turns={turns_in_batch} | "
                  f"tokens={batch_input_tokens:,}in/{batch_output_tokens:,}out "
                  f"cached={batch_cached_tokens:,}({cache_pct:.0f}%) "
                  f"reasoning={batch_reasoning_tokens:,} "
                  f"(cumul: {self.agent.total_input_tokens:,}in/{self.agent.total_output_tokens:,}out)",
                  file=sys.stderr, flush=True)

            if verbose:
                print(f"  📊 End of batch: Cash=${cash:,.0f}, Subs={subscribers}")

            # 保存当前 sim_day 的 checkpoint
            stable_checkpoint = self._save_checkpoint(
                sim_day,
                resume_conversation=resumable_batch,
                pending_observation=observation if resumable_batch else None,
            )

            if resumable_batch:
                game_outcome = 'incomplete'
                break

        if game_outcome is None:
            raise RuntimeError("Experiment loop exited without a terminal outcome")

        # 所有结果字段都从同一个稳定断点读取，避免现金、日期和 Token 用量错位。
        if game_outcome == 'timeout':
            # 超时后的后台 step_week 仍可能改变内存数据库，结果必须对齐上一个可恢复断点。
            stable_checkpoint = self._load_checkpoint()
            if not stable_checkpoint:
                raise RuntimeError("Timeout occurred without a stable checkpoint")
        else:
            sim_day = last_status['day']
            # incomplete 已保存带完整对话的断点；终态需要补存最终环境状态。
            if game_outcome in {'completed', 'bankrupt'}:
                # 终态之后不再调用 Analysis LLM：事后产物不参与决策，
                # 否则会把无法改善最终现金的调用错误计入创新模块成本。
                self._save_checkpoint(sim_day)
                stable_checkpoint = self._load_checkpoint()
            if not stable_checkpoint:
                raise RuntimeError("Experiment ended without a stable checkpoint")

        if game_outcome in {'completed', 'bankrupt'}:
            finalized = self._http_post(
                '/finalize-run', {'outcome': game_outcome}, timeout=30
            )
            if not finalized.get('success'):
                raise RuntimeError(f"Run finalization failed: {finalized}")

        result = self._result_from_checkpoint(stable_checkpoint, game_outcome)
        result_agent_state = stable_checkpoint['runtime']['agent']
        sim_day = stable_checkpoint['day']
        final_cash = stable_checkpoint['cash']
        self._log_performance(
            "run_summary",
            sim_day,
            outcome=game_outcome,
            final_cash=final_cash,
            modules={
                "bash_agent": {
                    "call_count": result_agent_state["total_turns"],
                    "input_tokens": result_agent_state["input_tokens"],
                    "output_tokens": result_agent_state["output_tokens"],
                    "cached_tokens": result_agent_state["cached_tokens"],
                    "reasoning_tokens": result_agent_state["reasoning_tokens"],
                    "cost_by_currency": result_agent_state[
                        "decision_cost_by_currency"
                    ],
                },
                "analysis": stable_checkpoint["runtime"]["analysis"],
                "social_llm": stable_checkpoint["runtime"]["environment_llm"],
            },
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"RUN COMPLETE")
            print(f"{'='*60}")
            print(f"Final Cash: ${final_cash:,.0f}")
            print(f"Sim Days Run: {sim_day}")
            print(f"Outcome: {game_outcome}")
            print(f"Total Turns: {result_agent_state['total_turns']}")
            total_input_tokens = result_agent_state['input_tokens']
            cache_pct = (
                result_agent_state['cached_tokens'] / total_input_tokens * 100
                if total_input_tokens > 0 else 0
            )
            print(
                f"Total Tokens: {total_input_tokens:,} input / "
                f"{result_agent_state['output_tokens']:,} output"
            )
            print(
                f"Cached Tokens: {result_agent_state['cached_tokens']:,} "
                f"({cache_pct:.0f}% of input)"
            )
            print(f"Reasoning Tokens: {result_agent_state['reasoning_tokens']:,}")
            print(
                "Decision Agent Cost: "
                f"{result_agent_state['decision_cost_by_currency']}"
            )
            print(f"{'='*60}\n")

        self._write_result(result)
        return result
