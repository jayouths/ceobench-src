#!/usr/bin/env python3
"""Test runner for Bash Agent with SaaS Bench.

This script runs a simulation using the bash_agent with any supported LLM provider.
The agent uses bash/file tools and interacts with the simulator via
novamind_api (Python library) and ./novamind-operation (CLI).

The simulation engine runs as a separate subprocess (novamind-server start-server).
The harness communicates with it exclusively via HTTP — no direct DB or simulator
access. This ensures the harness and the public repo have identical interfaces.

Supports explicit OpenAI, OpenAI-compatible, Anthropic, and Bedrock clients.
"""

import json
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time as _time
import urllib.request
import urllib.error
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add package to path
package_root = Path(__file__).parent.parent.parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

DEFAULT_EXPERIMENT_CONFIG = package_root.parent / "experiments" / "experiment.toml"
RUN_CONFIG_FORMAT_VERSION = 7
RUN_CONFIG_FIELDS = {
    "format_version", "run_id", "agent_type", "model", "provider", "api_type",
    "base_url", "reasoning_effort", "temperature", "top_p", "tool_choice", "max_output_tokens",
    "timeout_seconds", "request_options", "pricing", "pricing_model_map", "api_key_env",
    "api_key_required", "seed", "scenario", "total_days", "initial_cash",
    "max_decision_turns_per_batch", "max_invalid_responses_per_turn", "label", "simulator_llm",
    "analysis_module", "analysis_model",
    "public_bundle_sha256", "harness_git_commit", "harness_git_dirty",
    "harness_source_sha256",
}

from saas_bench.experiment_config import load_experiment_config
from saas_bench.llm_provider import (
    call_text_model,
    create_llm_client,
    model_token_cost,
    validate_provider_api_type,
    validate_reasoning_effort,
    validate_tool_choice,
)

from saas_bench.environment import Action
from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.analysis.signal_models import AnalysisSignals
from saas_bench.agents.bash_agent.analysis.models import (
    Role,
    AnalysisCallKind,
    RoleCallUsage,
    RoleReportsArtifact,
    StateCallUsage,
    StatePortraitArtifact,
)
from saas_bench.agents.bash_agent.analysis.role_reports import (
    RoleCallOutcome,
    RoleReportGenerator,
)
from saas_bench.agents.bash_agent.analysis.state_reconstruction import (
    StateCallOutcome,
    StateReconstructor,
)
from saas_bench.agents.bash_agent.analysis.brief import render_strategy_brief
from saas_bench.agents.bash_agent.analysis.signals import (
    SignalCollector,
    parse_public_week_snapshot,
)
from saas_bench.json_io import write_json_atomic, write_text_atomic


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


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


@dataclass(frozen=True)
class CheckpointRestorePlan:
    """Fully validated in-memory state needed to apply one checkpoint."""

    session_id: str
    conversation_payload: Dict[str, Any]


class BashAgentRunner:
    """Runner for bash_agent with SaaS Bench.

    The simulation runs in a separate subprocess (novamind-server start-server).
    This harness only handles: agent LLM calls, tool execution, timing, and
    checkpoint management. All simulation state is queried via HTTP.
    """

    CHECKPOINT_FORMAT_VERSION = 6

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
        public_bundle_sha256: Optional[str] = None,
        harness_git_commit: Optional[str] = None,
        harness_git_dirty: Optional[bool] = None,
        harness_source_sha256: Optional[str] = None,
        continue_from: Optional[Path] = None,
        label: Optional[str] = None,
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
        if public_bundle_sha256 is not None and (
            not isinstance(public_bundle_sha256, str)
            or len(public_bundle_sha256) != 64
            or any(char not in "0123456789abcdef" for char in public_bundle_sha256)
        ):
            raise ValueError("public_bundle_sha256 must be a lowercase SHA-256 digest")
        self.public_bundle_sha256 = public_bundle_sha256
        self.harness_git_commit = None
        self.harness_git_dirty = None
        self.harness_source_sha256 = None
        if continue_from:
            original_identity = {
                "harness_git_commit": harness_git_commit,
                "harness_git_dirty": harness_git_dirty,
                "harness_source_sha256": harness_source_sha256,
            }
            self._validate_harness_identity(original_identity)
            self._ensure_harness_identity()
            if self.harness_source_sha256 != original_identity["harness_source_sha256"]:
                print(
                    "WARNING: Current Harness source differs from the original run; "
                    "resume will continue with the current code.",
                    file=sys.stderr,
                    flush=True,
                )
        self.continue_from = continue_from
        self.label = label  # Optional human-readable variant tag — surfaced on the dashboard
        self._resume_checkpoint: Optional[Dict[str, Any]] = None
        self._last_committed_week = 0
        if continue_from:
            self.workspace_dir = Path(continue_from).resolve()
            if not self.workspace_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {self.workspace_dir}")
            self.run_id = _load_saved_run_config(self.workspace_dir)['run_id']
            self.workspace_base = self.workspace_dir.parent
        else:
            self.run_id = str(uuid.uuid4())[:8]
            self.workspace_base = (workspace_base or Path('./bash_agent_runs')).resolve()
            self.workspace_dir = self.workspace_base / f"run_{self.run_id}"

        # Agent working directory (inside the run directory)
        self.agent_workspace = self.workspace_dir / "agent_workspace"

        # Logs directory
        self.logs_dir = self.workspace_dir / "logs"

        # Log file for raw responses
        self.response_log_file = self.logs_dir / f"raw_responses_{self.run_id}.jsonl"

        # 记录每次 LLM 调用、工具执行和决策批次的耗时。
        self.timing_log_file = self.logs_dir / f"timing_{self.run_id}.jsonl"

        # CEOBench dashboard URL for live timing push (set via env var)
        self._dashboard_url = os.environ.get("CEOBENCH_DASHBOARD_URL", "")
        self._timing_queue = None
        self._timing_thread = None

        # Load API key
        env_file = Path(__file__).parent.parent.parent.parent.parent / ".env"
        env_vars = load_env_file(env_file)
        self._env_vars = env_vars

        for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION',
                    'AWS_SESSION_TOKEN', 'NMDB_KEY']:
            if key in env_vars and key not in os.environ:
                os.environ[key] = env_vars[key]

        # The .nmdb session database is SQLCipher-encrypted. The engine resolves
        # the key from saas_bench._embedded_key (committed in the source tree
        # and compiled into the zipapp) or, failing that, the NMDB_KEY env var.
        # Fail fast here only if neither source is available.
        try:
            from saas_bench.db_protection import _get_key
            _get_key()
        except RuntimeError as exc:
            raise RuntimeError(
                "No SQLCipher key available for the .nmdb session database: "
                "neither saas_bench._embedded_key nor the NMDB_KEY env var is "
                "set. Restore src/saas_bench/_embedded_key.py, or set NMDB_KEY "
                "in .env or the environment."
            ) from exc

        if self.api_key_env:
            self.api_key = env_vars.get(self.api_key_env) or os.environ.get(self.api_key_env)
        elif self.provider == "bedrock":
            self.api_key = None
        else:
            self.api_key = None

        if not self.api_key and not self.api_key_required:
            self.api_key = "not-required"
        if not self.api_key and self.provider not in ("bedrock",):
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

        # Components (initialized in setup)
        self.agent = None
        self.tool_executor = None
        self._server_proc = None
        self._server_port = None
        self._server_socket_dir = None
        self._server_socket_path = None
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
        if not api_key and provider != "bedrock":
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

    def _server_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._server_port}{path}"

    def _http_get(self, path: str, timeout: float = 30) -> Dict:
        req = urllib.request.Request(self._server_url(path))
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())

    def _http_post(self, path: str, data: Optional[Dict] = None, timeout: float = 1800) -> Dict:
        body = json.dumps(data or {}).encode()
        req = urllib.request.Request(
            self._server_url(path), data=body,
            headers={'Content-Type': 'application/json'},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            # 业务校验错误同样返回 JSON，保留服务端给出的明确失败原因。
            try:
                return json.loads(exc.read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise

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

    def _get_dashboard(self) -> str:
        """Get current dashboard via HTTP."""
        result = self._get_dashboard_payload()
        return result['dashboard']

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

    def _analysis_signal_path(self, day: int) -> Path:
        return self.workspace_dir / "analysis" / f"day_{day:03d}" / "signals.json"

    def _analysis_role_reports_path(self, day: int) -> Path:
        return (
            self.workspace_dir
            / "analysis"
            / f"day_{day:03d}"
            / "role_reports.json"
        )

    def _analysis_state_portrait_path(self, day: int) -> Path:
        return (
            self.workspace_dir
            / "analysis"
            / f"day_{day:03d}"
            / "state_portrait.json"
        )

    def _analysis_brief_path(self, day: int) -> Path:
        return (
            self.workspace_dir
            / "analysis"
            / f"day_{day:03d}"
            / "STRATEGY_BRIEF.md"
        )

    def _load_analysis_history(self, before_or_at_day: int) -> Dict[int, AnalysisSignals]:
        """读取已完成周的确定性产物，供环比和 Dashboard 独有指标使用。"""
        history = {}
        analysis_dir = self.workspace_dir / "analysis"
        if not analysis_dir.is_dir():
            return history
        for path in sorted(analysis_dir.glob("day_*/signals.json")):
            try:
                signals = AnalysisSignals.model_validate_json(path.read_text())
            except (OSError, ValueError) as exc:
                raise ValueError(f"Invalid Analysis signals artifact: {path}") from exc
            if signals.day <= before_or_at_day:
                history[signals.day] = signals
        return history

    def _ensure_analysis_signals(self, dashboard_payload: Dict[str, Any]) -> AnalysisSignals | None:
        """同一模拟日只生成一次信号；恢复同周时复用原产物。"""
        if not self.analysis_enabled:
            return None
        snapshot = parse_public_week_snapshot(
            dashboard_payload.get("public_week_snapshot")
        )
        path = self._analysis_signal_path(snapshot.day)
        if path.is_file():
            signals = AnalysisSignals.model_validate_json(path.read_text())
            if signals.day != snapshot.day:
                raise ValueError(f"Analysis artifact day mismatch: {path}")
            return signals

        history = self._load_analysis_history(snapshot.day - 1)
        collector = SignalCollector(
            self._query_public_rows,
            max_enterprise_threads=self.analysis_module_config[
                "max_enterprise_threads"
            ],
        )
        signals = collector.collect(snapshot, history)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, signals.model_dump(mode="json"))
        return signals

    def _analysis_task_parameters(self, task: str) -> Dict[str, Any]:
        """合并 Analysis 模型公共参数与任务级覆盖。"""

        config = self.analysis_model_config
        if config is None:
            raise ValueError("analysis model config is required")
        task_values = dict(config.get("tasks", {}).get(task, {}))
        request_options = {
            key: dict(value)
            for key, value in config.get("request_options", {}).items()
        }
        for key, value in task_values.get("request_options", {}).items():
            request_options.setdefault(key, {}).update(value)
        return {
            "max_output_tokens": task_values.get(
                "max_output_tokens", config["max_output_tokens"]
            ),
            "temperature": task_values.get("temperature", config.get("temperature")),
            "top_p": task_values.get("top_p", config.get("top_p")),
            "reasoning_effort": task_values.get(
                "reasoning_effort", config.get("reasoning_effort")
            ),
            "request_options": request_options,
        }

    @staticmethod
    def _jsonable_llm_response(raw_response: Any) -> Any:
        if hasattr(raw_response, "model_dump"):
            return raw_response.model_dump(
                mode="json", exclude_none=False, by_alias=True
            )
        if isinstance(raw_response, (dict, list, str, int, float, bool)):
            return raw_response
        return str(raw_response)

    def _call_analysis_model(
        self,
        *,
        task: str,
        day: int,
        attempt: int,
        call_kind: AnalysisCallKind,
        system_prompt: str,
        user_prompt: str,
        role: Role | None = None,
    ):
        """统一执行 Analysis 调用，确保所有任务使用相同计费和日志口径。"""

        if self.analysis_client is None or self.analysis_model_config is None:
            raise RuntimeError("analysis client is not initialized")
        config = self.analysis_model_config
        started = _time.monotonic()
        response = call_text_model(
            client=self.analysis_client,
            api_type=config["api_type"],
            model=config["model"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **self._analysis_task_parameters(task),
        )
        elapsed = _time.monotonic() - started
        cost = model_token_cost(
            response.model,
            response.input_tokens,
            response.output_tokens,
            response.cached_tokens,
            config["pricing"],
            config.get("pricing_model_map"),
        )

        identity = {
            "component": "analysis",
            "analysis_task": task,
            "attempt": attempt,
            "call_kind": call_kind.value,
            "day": day,
        }
        if role is not None:
            identity["role"] = role.value
        usage_fields = {
            "requested_model": config["model"],
            "served_model": response.model,
            "pricing_model": cost.pricing_model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cached_tokens": response.cached_tokens,
            "reasoning_tokens": response.reasoning_tokens,
            "cost_amount": cost.amount,
            "currency": cost.currency,
        }
        raw_entry = {
            "timestamp": now(),
            **identity,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_text": response.text,
            "raw_response": self._jsonable_llm_response(response.raw_response),
            "elapsed_seconds": elapsed,
            **usage_fields,
        }
        with open(self.response_log_file, "a") as file:
            file.write(json.dumps(raw_entry, ensure_ascii=False) + "\n")
        self._log_timing(
            "analysis_llm_call",
            day,
            **{key: value for key, value in identity.items() if key != "day"},
            elapsed_s=round(elapsed, 3),
            **usage_fields,
        )
        return response, cost, elapsed

    def _call_analysis_role_model(
        self,
        day: int,
        role: Role,
        attempt: int,
        call_kind: AnalysisCallKind,
        system_prompt: str,
        user_prompt: str,
    ) -> RoleCallOutcome:
        """执行并完整记录一次角色报告或修复调用。"""

        response, cost, elapsed = self._call_analysis_model(
            task="role_report",
            day=day,
            attempt=attempt,
            call_kind=call_kind,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            role=role,
        )
        usage = RoleCallUsage(
            role=role,
            attempt=attempt,
            call_kind=call_kind,
            requested_model=self.analysis_model_config["model"],
            served_model=response.model,
            pricing_model=cost.pricing_model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            reasoning_tokens=response.reasoning_tokens,
            elapsed_seconds=elapsed,
            cost_amount=cost.amount,
            currency=cost.currency,
        )
        return RoleCallOutcome(text=response.text, usage=usage)

    def _call_analysis_state_model(
        self,
        day: int,
        attempt: int,
        call_kind: AnalysisCallKind,
        system_prompt: str,
        user_prompt: str,
    ) -> StateCallOutcome:
        """执行一次状态重构或修复调用。"""

        response, cost, elapsed = self._call_analysis_model(
            task="state_reconstruction",
            day=day,
            attempt=attempt,
            call_kind=call_kind,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        usage = StateCallUsage(
            attempt=attempt,
            call_kind=call_kind,
            requested_model=self.analysis_model_config["model"],
            served_model=response.model,
            pricing_model=cost.pricing_model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            reasoning_tokens=response.reasoning_tokens,
            elapsed_seconds=elapsed,
            cost_amount=cost.amount,
            currency=cost.currency,
        )
        return StateCallOutcome(text=response.text, usage=usage)

    def _ensure_analysis_role_reports(
        self,
        signals: AnalysisSignals,
    ) -> tuple[RoleReportsArtifact | None, bool]:
        """返回周度角色报告；布尔值表示本次是否产生了新的 LLM 调用。"""

        if not self.analysis_enabled:
            return None, False
        path = self._analysis_role_reports_path(signals.day)
        if path.is_file():
            artifact = RoleReportsArtifact.model_validate_json(path.read_text())
            if artifact.day != signals.day:
                raise ValueError(f"Analysis role report day mismatch: {path}")
            return artifact, False

        generator = RoleReportGenerator(
            self._call_analysis_role_model,
            max_schema_retries=self.analysis_module_config["max_schema_retries"],
        )
        artifact = generator.generate(signals)
        write_json_atomic(path, artifact.model_dump(mode="json"))
        return artifact, True

    def _ensure_analysis_state_portrait(
        self,
        role_reports: RoleReportsArtifact,
    ) -> tuple[StatePortraitArtifact | None, bool]:
        """返回周度经营画像；布尔值表示本次是否产生了新的 LLM 调用。"""

        if not self.analysis_enabled:
            return None, False
        path = self._analysis_state_portrait_path(role_reports.day)
        if path.is_file():
            artifact = StatePortraitArtifact.model_validate_json(path.read_text())
            if artifact.day != role_reports.day:
                raise ValueError(f"Analysis state portrait day mismatch: {path}")
            return artifact, False

        reconstructor = StateReconstructor(
            self._call_analysis_state_model,
            max_schema_retries=self.analysis_module_config["max_schema_retries"],
        )
        artifact = reconstructor.generate(role_reports)
        write_json_atomic(path, artifact.model_dump(mode="json"))
        return artifact, True

    def _ensure_analysis_brief(
        self,
        role_reports: RoleReportsArtifact,
        portrait: StatePortraitArtifact,
    ) -> tuple[str | None, bool]:
        """生成确定性状态简报；关闭 Analysis 时保持 Baseline 不变。"""

        if not self.analysis_enabled:
            return None, False
        path = self._analysis_brief_path(portrait.day)
        if path.is_file():
            return path.read_text(), False
        brief = render_strategy_brief(role_reports, portrait)
        write_text_atomic(path, brief)
        return brief, True

    def _decision_observation(self, dashboard: str, brief: str | None) -> str:
        """只在 Analysis 开启时向原始 Dashboard 追加状态简报。"""

        if not self.analysis_enabled:
            return dashboard
        if not brief:
            raise RuntimeError("Analysis brief is required when Analysis is enabled")
        return f"{dashboard}\n\n---\n\n{brief}"

    def _analysis_usage_summary(self, before_or_at_day: int) -> Dict[str, Any]:
        """从已落盘周产物重建 Analysis 累计用量，避免维护第二份可变计数器。"""

        def empty_usage() -> Dict[str, Any]:
            return {
                "call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "cost_by_currency": {},
            }

        def add_call(target: Dict[str, Any], call: Any) -> None:
            target["call_count"] += 1
            for field in (
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "reasoning_tokens",
            ):
                target[field] += getattr(call, field)
            costs = target["cost_by_currency"]
            costs[call.currency] = costs.get(call.currency, 0.0) + call.cost_amount

        totals = empty_usage()
        by_role = {
            role.value: empty_usage()
            for role in Role
        }
        state_reconstruction = empty_usage()
        role_report_days: list[int] = []
        state_portrait_days: list[int] = []
        analysis_dir = self.workspace_dir / "analysis"
        if analysis_dir.is_dir():
            for path in sorted(analysis_dir.glob("day_*/role_reports.json")):
                artifact = RoleReportsArtifact.model_validate_json(path.read_text())
                if artifact.day > before_or_at_day:
                    continue
                role_report_days.append(artifact.day)
                for call in artifact.calls:
                    add_call(totals, call)
                    add_call(by_role[call.role.value], call)
            for path in sorted(analysis_dir.glob("day_*/state_portrait.json")):
                artifact = StatePortraitArtifact.model_validate_json(path.read_text())
                if artifact.day > before_or_at_day:
                    continue
                state_portrait_days.append(artifact.day)
                for call in artifact.calls:
                    add_call(totals, call)
                    add_call(state_reconstruction, call)
        return {
            "role_report_days": role_report_days,
            "state_portrait_days": state_portrait_days,
            **totals,
            "by_role": by_role,
            "state_reconstruction": state_reconstruction,
        }

    def _prune_analysis_artifacts_after(
        self,
        day: int,
        role_report_days: set[int] | None = None,
        state_portrait_days: set[int] | None = None,
    ) -> None:
        """恢复时按断点确认范围清理 LLM 产物，保留确定性信号。"""

        role_report_days = role_report_days or set()
        state_portrait_days = state_portrait_days or set()
        analysis_dir = self.workspace_dir / "analysis"
        if not analysis_dir.is_dir():
            return
        for directory in analysis_dir.glob("day_*"):
            try:
                artifact_day = int(directory.name.removeprefix("day_"))
            except ValueError:
                continue
            if artifact_day > day:
                shutil.rmtree(directory)
            elif artifact_day not in role_report_days:
                (directory / "role_reports.json").unlink(missing_ok=True)
                (directory / "state_portrait.json").unlink(missing_ok=True)
                (directory / "STRATEGY_BRIEF.md").unlink(missing_ok=True)
            elif artifact_day not in state_portrait_days:
                (directory / "state_portrait.json").unlink(missing_ok=True)
                (directory / "STRATEGY_BRIEF.md").unlink(missing_ok=True)

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

    def _log_response(self, turn: int, day: int, messages: List[Dict], raw_response: Any):
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
        entry = {
            "timestamp": now(),
            "turn": turn,
            "day": day,
            "messages_count": len(messages),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "served_model": served_model,
            "pricing_model": cost.pricing_model,
            "cost_amount": cost.amount,
            "currency": cost.currency,
            "total_cost_by_currency": self.total_decision_agent_cost_by_currency,
            "raw_response": raw_response,
        }
        initial_observation = (
            getattr(self.agent, "initial_observation_for_audit", None)
            if self.agent else None
        )
        if initial_observation is not None:
            # 保存实际进入模型请求的周初 observation，完整正文已经足够用于实验审计。
            entry["initial_observation"] = initial_observation
            entry["analysis_brief_injected"] = False
            if self.analysis_enabled:
                brief_path = self._analysis_brief_path(day)
                if brief_path.is_file():
                    brief = brief_path.read_text()
                    expected_suffix = f"\n\n---\n\n{brief}"
                    if initial_observation.endswith(expected_suffix):
                        entry["analysis_brief_injected"] = True
        with open(self.response_log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _log_tool_result(self, turn: int, day: int, tool_name: str, arguments: Dict, result: str):
        tool_results_file = self.logs_dir / f"tool_results_{self.run_id}.jsonl"
        entry = {
            "timestamp": now(),
            "turn": turn,
            "day": day,
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
        }
        with open(tool_results_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _log_timing(self, event: str, day: int, turn: int = 0, **kwargs):
        """Log a timing event to the timing JSONL file and push to dashboard."""
        entry = {
            "timestamp": now(),
            "run_id": self.run_id,
            "event": event,
            "day": day,
            "turn": turn,
            **kwargs,
        }
        with open(self.timing_log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        # Push to ceobench dashboard (non-blocking)
        if self._timing_queue is not None:
            try:
                self._timing_queue.put_nowait(entry)
            except Exception:
                pass

    def _start_timing_poster(self) -> None:
        """Start the optional dashboard sender inside the managed run lifecycle."""
        if not self._dashboard_url or self._timing_thread is not None:
            return

        import queue
        import threading

        self._timing_queue = queue.Queue(maxsize=500)

        def _post_batch(batch: List[Dict[str, Any]]) -> None:
            try:
                data = json.dumps(batch).encode()
                request = urllib.request.Request(
                    self._dashboard_url.rstrip('/') + '/ingest',
                    data=data,
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
                urllib.request.urlopen(request, timeout=10)
            except Exception:
                pass  # 仪表盘不可用不得影响主实验。

        def _timing_poster() -> None:
            while True:
                batch = []
                stop_requested = False
                try:
                    item = self._timing_queue.get(timeout=5)
                    if item is None:
                        stop_requested = True
                    else:
                        batch.append(item)
                    for _ in range(20):
                        try:
                            item = self._timing_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            stop_requested = True
                            break
                        batch.append(item)
                except queue.Empty:
                    pass
                if batch:
                    _post_batch(batch)
                if stop_requested:
                    return

        self._timing_thread = threading.Thread(
            target=_timing_poster,
            name=f"ceobench-timing-{self.run_id}",
            daemon=True,
        )
        self._timing_thread.start()

    def _stop_timing_poster(self) -> None:
        """Flush queued timing entries and stop the optional sender."""
        thread = self._timing_thread
        queue = self._timing_queue
        if thread is None:
            return
        if queue is not None:
            try:
                queue.put(None, timeout=1)
            except Exception:
                pass
        thread.join(timeout=15)
        self._timing_thread = None
        self._timing_queue = None

    # =========================================================================
    # Workspace setup
    # =========================================================================

    _GITIGNORE_CONTENT = """\
sessions/
_engine/
*.nmdb
*.db
*.db-journal
*.db-wal
*.db-shm
__pycache__/
*.pyc
.pytest_cache/
.venv/
"""

    def _git(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(self.agent_workspace),
            capture_output=True, text=True,
            check=check,
        )

    def _git_init_workspace(self):
        if (self.agent_workspace / ".git").exists():
            return
        self._git("init", "-q", "-b", "main", check=True)
        self._git(
            "config", "user.email", "bash-agent@bossbench.local", check=True
        )
        self._git("config", "user.name", "BashAgent", check=True)
        gitignore_path = self.agent_workspace / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(self._GITIGNORE_CONTENT)

    def _git_commit_workspace(self, message: str, once_key: Optional[str] = None):
        if not (self.agent_workspace / ".git").exists():
            return
        if once_key is not None:
            existing = self._git("log", "--grep", f"[{once_key}]", "--fixed-strings", "--oneline")
            if existing.returncode == 0 and existing.stdout.strip():
                return
            message = f"{message} [{once_key}]"
        self._git("add", "-A", check=True)
        status = self._git("status", "--porcelain", check=True)
        if not status.stdout.strip():
            # Empty commit so the tag still lands on the timeline
            self._git("commit", "--allow-empty", "-q", "-m", message, check=True)
        else:
            self._git("commit", "-q", "-m", message, check=True)

    def _capture_workspace_commit(self, day: int) -> str:
        """Commit any pending Agent files and return the exact checkpoint revision."""
        if not (self.agent_workspace / ".git").is_dir():
            raise RuntimeError("Agent workspace is not a Git repository")
        self._git("add", "-A", check=True)
        status = self._git("status", "--porcelain", check=True)
        if status.stdout.strip():
            self._git("commit", "-q", "-m", f"Checkpoint workspace (day {day})", check=True)
        head = self._git("rev-parse", "HEAD", check=True).stdout.strip()
        if not head:
            raise RuntimeError("Failed to resolve Agent workspace checkpoint commit")
        return head

    def _commit_weeks_up_to(self, sim_day: int):
        # Agent advances time via `./novamind-operation next-week ...`, which can cross
        # one or more sim-week boundaries inside a single harness loop iteration. The
        # once_key dedupe makes this safe to call after every sim_day update.
        if sim_day <= 0:
            return
        target_week = sim_day // 7
        while self._last_committed_week < target_week:
            next_week = self._last_committed_week + 1
            week_end_day = next_week * 7
            self._git_commit_workspace(
                f"Week {next_week} (day {week_end_day})",
                once_key=f"week-{next_week}",
            )
            # 只有 Git 提交成功后才推进周游标，避免静默丢失时间线节点。
            self._last_committed_week = next_week

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
        import stat

        public_dir = self._public_dir()
        self.agent_workspace.mkdir(parents=True, exist_ok=True)
        self._git_init_workspace()

        # docs/ is the only directory the agent needs — it holds the tool/table
        # JSON, cli.md, and the readable SDK source at
        # docs/novamind_api/ (used for ``import novamind_api`` at runtime).
        src_docs = public_dir / "docs"
        dst_docs = self.agent_workspace / "docs"
        if src_docs.exists():
            if dst_docs.exists():
                shutil.rmtree(dst_docs)
            shutil.copytree(
                src_docs, dst_docs,
                ignore=shutil.ignore_patterns('__pycache__'),
            )

        # Copy novamind-operation (zipapp). This is the ONLY executable the
        # agent has — no separate novamind-server, no install.sh, nothing else.
        src_op = public_dir / "novamind-operation"
        dst_op = self.agent_workspace / "novamind-operation"
        if not src_op.exists():
            raise FileNotFoundError(
                f"{src_op} does not exist. Did you run `uv run python scripts/build_public.py`?"
            )
        shutil.copy2(src_op, dst_op)
        dst_op.chmod(dst_op.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # Run new-session via the HOST-SIDE zipapp (bytecode stays on host).
        env = self._server_environment()
        result = subprocess.run(
            [
                sys.executable, str(src_op),
                "--base", str(self.agent_workspace),
                "new-session",
                "--days", str(self.total_days),
                "--seed", str(self.seed),
                "--cash", str(self.initial_cash),
                "--scenario", self.scenario,
            ],
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"novamind-operation new-session failed:\n{result.stderr}\n{result.stdout}"
            )

        session_info = json.loads(result.stdout)
        self._session_id = session_info["session_id"]
        print(f"  Session created via CLI: {self._session_id}")
        self._git_commit_workspace("Initial workspace setup (day 0)")

    def _public_dir(self) -> Path:
        """Location of the host-side public/ bundle (contains _engine/).

        The bash_agent run_test.py lives at:
            <root>/src/saas_bench/agents/bash_agent/run_test.py
        public/ lives at <root>/public/. parent^5 = <root>.

        Variant runs override this via the NOVAMIND_PUBLIC_DIR env var so they
        can launch with a per-variant zipapp (built locally, never pushed). The
        env var is read every call rather than cached so test fixtures can swap
        it on the fly.
        """
        override = os.environ.get("NOVAMIND_PUBLIC_DIR")
        if override:
            public_dir = Path(override).resolve()
            if not public_dir.exists():
                raise FileNotFoundError(
                    f"NOVAMIND_PUBLIC_DIR points to {public_dir} which does not exist."
                )
        else:
            public_dir = Path(__file__).parent.parent.parent.parent.parent / "public"
            if not public_dir.exists():
                raise FileNotFoundError(
                    f"public/ directory not found at {public_dir}. "
                    f"Run 'uv run python scripts/build_public.py' first."
                )
        return public_dir

    def _server_environment(self) -> Dict[str, str]:
        """Environment for host-side simulator processes."""
        env = os.environ.copy()
        env["NOVAMIND_SERVER_MODE"] = "1"
        if self.simulator_llm_config:
            env["CEOBENCH_SIMULATOR_LLM_CONFIG"] = json.dumps(
                self.simulator_llm_config, separators=(",", ":")
            )
            for field, value in self.simulator_llm_config.items():
                if field.endswith("_api_key_env") and value and value in self._env_vars:
                    env.setdefault(value, self._env_vars[value])
        return env

    def _launch_server(self):
        """Launch the host-side novamind-operation zipapp in server mode.

        The zipapp lives in public/ only. Setting NOVAMIND_SERVER_MODE=1 makes
        its __main__ dispatch to saas_bench.server_entry.main() instead of the
        client-side CLI. The server process runs in the parent environment
        (outside bwrap) so it has access to NMDB_KEY.

        Reads the first line of stdout to get the port, then waits for /health.
        """
        zipapp_path = self._public_dir() / "novamind-operation"
        server_env = self._server_environment()
        # Route api_server stderr to a file rather than a pipe back to bash_agent.
        # bash_agent never drains the pipe during a /call, so a single buffered
        # traceback (>64KB pipe capacity) wedges write() under self._lock and
        # deadlocks every subsequent /call. (run 27c000a5 d105 hang.)
        self._server_stderr_path = self.logs_dir / "api_server_stderr.log"
        self._server_stderr_file = open(self._server_stderr_path, "ab", buffering=0)
        # 短路径避免 Unix Socket 的系统长度限制；目录不放进 Agent 可写工作区。
        self._server_socket_dir = Path(tempfile.mkdtemp(prefix=f"ceobench-{self.run_id}-"))
        self._server_socket_path = self._server_socket_dir / "api.sock"
        self._server_proc = subprocess.Popen(
            [
                sys.executable, str(zipapp_path),
                "--base", str(self.agent_workspace),
                "start-server",
                "--session", self._session_id,
                "--unix-socket", str(self._server_socket_path),
            ],
            stdout=subprocess.PIPE,
            stderr=self._server_stderr_file,
            env=server_env,
        )

        # Read first line of stdout to get port info
        first_line = self._server_proc.stdout.readline()
        if not first_line:
            try:
                stderr_tail = self._server_stderr_path.read_bytes()[-4096:]
            except Exception:
                stderr_tail = b"<stderr log unavailable>"
            raise RuntimeError(f"Server failed to start:\n{stderr_tail.decode(errors='replace')}")

        server_info = json.loads(first_line)
        self._server_port = server_info["port"]
        if server_info.get("unix_socket") != str(self._server_socket_path):
            raise RuntimeError("Server did not expose the requested Agent API socket")
        print(f"  Server started: port={self._server_port}, pid={server_info['pid']}")

        # Wait for health check
        for i in range(60):
            try:
                self._http_get('/health', timeout=2)
                return
            except Exception:
                _time.sleep(0.5)

        raise RuntimeError("Server did not respond to /health after 30s")

    def _stop_server(self):
        """Idempotently stop a fully or partially launched server subprocess."""
        server_proc = getattr(self, "_server_proc", None)
        if server_proc:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=210)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            self._server_proc = None
        self._server_port = None
        socket_dir = getattr(self, "_server_socket_dir", None)
        if socket_dir:
            shutil.rmtree(socket_dir, ignore_errors=True)
        self._server_socket_dir = None
        self._server_socket_path = None
        f = getattr(self, "_server_stderr_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._server_stderr_file = None

    # =========================================================================
    # Checkpoint
    # =========================================================================

    def _check_tamper(self, day: int) -> List[str]:
        """Scan agent workspace for sandbox-escape indicators.

        Looks for files that an agent has no legitimate reason to create —
        primarily duplicate `*.nmdb` files in `sessions/<sid>/` (the engine
        only writes `world.nmdb`; anything else is a backup the agent made
        before tampering).

        Returns a list of suspicious file paths (relative to agent_workspace).
        Empty list = clean.

        Reference: gpt55 v3.4aa run 1267c284 (2026-04-28) created
        `world_before_week31_recovery_patch.nmdb` before running UPDATE
        statements directly against the decrypted DB.
        """
        flagged: List[str] = []
        sessions_dir = self.agent_workspace / "sessions"
        if sessions_dir.exists():
            for session_path in sessions_dir.iterdir():
                if not session_path.is_dir():
                    continue
                for nmdb in session_path.glob("*.nmdb"):
                    if nmdb.name != "world.nmdb":
                        flagged.append(str(nmdb.relative_to(self.agent_workspace)))
        # Anything matching `patch_*.py` or `recover_*.py` at workspace root
        # is also a strong signal of tamper attempts (gpt55 named its script
        # `patch_world_day217_cleanup.py`).
        for suspicious in self.agent_workspace.glob("patch_*.py"):
            flagged.append(str(suspicious.relative_to(self.agent_workspace)))
        for suspicious in self.agent_workspace.glob("recover_*.py"):
            flagged.append(str(suspicious.relative_to(self.agent_workspace)))
        return flagged

    def _save_checkpoint(
        self,
        day: int,
        *,
        resume_conversation: bool = False,
        pending_observation: Optional[str] = None,
    ):
        """Save checkpoint for resume capability."""
        # Tamper detection: log + persist any suspicious files in workspace.
        tamper_hits = self._check_tamper(day)
        if tamper_hits:
            tamper_log = self.workspace_dir / "tamper_alerts.jsonl"
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "day": day,
                "files": tamper_hits,
            }
            with open(tamper_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"  ⚠️  TAMPER ALERT day {day}: {len(tamper_hits)} suspicious file(s): {tamper_hits[:5]}")

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
        server_log_offsets = self._validate_server_log_offsets(
            persisted.get("server_log_offsets")
        )
        environment_llm_usage = self._validate_environment_llm_usage(
            persisted.get("environment_llm_usage")
        )
        analysis_usage = self._validate_analysis_usage(
            self._analysis_usage_summary(day),
            max_day=day,
        )

        session_nmdb = self.agent_workspace / "sessions" / self._session_id / "world.nmdb"
        if not session_nmdb.is_file():
            raise FileNotFoundError(f"Persisted session database not found: {session_nmdb}")

        # 数据库使用不可变版本文件，checkpoint.json 最后原子切换到该版本。
        # 即使进程在两个文件之间崩溃，旧 checkpoint 仍指向旧的完整数据库。
        checkpoint_db_dir = self.workspace_dir / ".checkpoint_dbs"
        checkpoint_db_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_id = uuid.uuid4().hex
        database_name = f"world_day_{day}_{checkpoint_id}.nmdb"
        checkpoint_db = checkpoint_db_dir / database_name
        checkpoint_db_tmp = checkpoint_db.with_suffix(".nmdb.tmp")
        shutil.copy2(session_nmdb, checkpoint_db_tmp)
        database_sha256 = self._sha256_file(checkpoint_db_tmp)
        os.replace(checkpoint_db_tmp, checkpoint_db)

        runtime_dir = self.workspace_dir / ".checkpoint_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        conversation_snapshot = runtime_dir / f"conversation_{checkpoint_id}.json"
        if self.agent is None:
            write_json_atomic(conversation_snapshot, {
                "format_version": BashAgent.CHECKPOINT_SNAPSHOT_FORMAT_VERSION,
                "resume_conversation": False,
                "tool_results_applied": True,
                "conversation": [],
                "pending_tool_calls": [],
                "current_day": 0,
                "turns_today": 0,
                "total_turns": 0,
                "saved_at": _time.time(),
            })
        else:
            self.agent.save_checkpoint_snapshot(
                conversation_snapshot,
                resume_conversation=resume_conversation,
                pending_observation=pending_observation,
            )
        conversation_sha256 = self._sha256_file(conversation_snapshot)
        workspace_commit = self._capture_workspace_commit(day)
        log_offsets = self._capture_log_offsets()

        # 断点只保存恢复所需的状态；模型、Provider 等实验身份由 config.json 唯一管理。
        checkpoint = {
            'format_version': self.CHECKPOINT_FORMAT_VERSION,
            'run_config_sha256': self._sha256_file(
                self.workspace_dir / "config.json"
            ),
            'day': day,
            'cash': float(checkpoint_cash),
            'session_id': self._session_id,
            'database': {
                'file': str(checkpoint_db.relative_to(self.workspace_dir)),
                'sha256': database_sha256,
            },
            'runtime': {
                'runner_log_offsets': log_offsets,
                'server_log_offsets': server_log_offsets,
                'conversation': {
                    'file': str(conversation_snapshot.relative_to(self.workspace_dir)),
                    'sha256': conversation_sha256,
                    'resume': resume_conversation,
                },
                'workspace_commit': workspace_commit,
                'environment_llm': environment_llm_usage,
                'analysis': analysis_usage,
                'agent': {
                    'total_turns': self.agent.total_turns if self.agent else 0,
                    'input_tokens': self.agent.total_input_tokens if self.agent else 0,
                    'output_tokens': self.agent.total_output_tokens if self.agent else 0,
                    'cached_tokens': self.agent.total_cached_tokens if self.agent else 0,
                    'reasoning_tokens': self.agent.total_reasoning_tokens if self.agent else 0,
                    'decision_cost_by_currency': self.total_decision_agent_cost_by_currency,
                },
            },
        }
        # checkpoint.json 是恢复入口，统一使用带刷盘保证的原子写入。
        write_json_atomic(self.workspace_dir / "checkpoint.json", checkpoint)

        # world.nmdb 仅作为分析用的最新副本；恢复以 checkpoint 指向的版本文件为准。
        try:
            harness_nmdb = self.workspace_dir / "world.nmdb"
            harness_tmp = harness_nmdb.with_suffix(harness_nmdb.suffix + ".tmp")
            shutil.copy2(checkpoint_db, harness_tmp)
            os.replace(harness_tmp, harness_nmdb)
        except OSError:
            pass

        # 清理失败只会留下无引用的旧文件，不影响刚刚提交的可信断点。
        for stale_db in checkpoint_db_dir.glob("*.nmdb"):
            if stale_db != checkpoint_db:
                try:
                    stale_db.unlink(missing_ok=True)
                except OSError:
                    pass
        for stale_snapshot in runtime_dir.glob("conversation_*.json"):
            if stale_snapshot != conversation_snapshot:
                try:
                    stale_snapshot.unlink(missing_ok=True)
                except OSError:
                    pass
        return checkpoint

    def _checkpoint_log_files(self) -> Dict[str, Path]:
        return {
            "tool_results": self.logs_dir / f"tool_results_{self.run_id}.jsonl",
            "raw_responses": self.response_log_file,
            "timing": self.timing_log_file,
        }

    def _capture_log_offsets(self) -> Dict[str, int]:
        return {
            name: path.stat().st_size if path.exists() else 0
            for name, path in self._checkpoint_log_files().items()
        }

    def _validate_runner_log_offsets(self, offsets: Any) -> Dict[str, int]:
        expected_names = set(self._checkpoint_log_files())
        if not isinstance(offsets, dict) or set(offsets) != expected_names:
            raise ValueError(
                f"Checkpoint log offsets must contain exactly: {sorted(expected_names)}"
            )
        for name, offset in offsets.items():
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                raise ValueError(f"Invalid checkpoint log offset for {name}: {offset!r}")
        return dict(offsets)

    def _restore_logs_to_offsets(self, offsets: Dict[str, Any]) -> None:
        validated = self._validate_runner_log_offsets(offsets)
        for name, path in self._checkpoint_log_files().items():
            offset = validated[name]
            current_size = path.stat().st_size if path.exists() else 0
            if offset > current_size:
                raise ValueError(
                    f"Checkpoint log offset for {name} exceeds file size: {offset} > {current_size}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a+b") as file:
                file.truncate(offset)

    @staticmethod
    def _validate_offsets_within_files(
        offsets: Dict[str, int], files: Dict[str, Path], label: str
    ) -> None:
        for name, path in files.items():
            current_size = path.stat().st_size if path.exists() else 0
            if offsets[name] > current_size:
                raise ValueError(
                    f"Checkpoint {label} offset for {name} exceeds file size: "
                    f"{offsets[name]} > {current_size}"
                )

    def _server_log_files(self, session_id: str | None = None) -> Dict[str, Path]:
        resolved_session_id = session_id or self._session_id
        if not resolved_session_id:
            raise ValueError("session_id is required to resolve server logs")
        session_dir = self.agent_workspace / "sessions" / resolved_session_id
        return {
            "history": session_dir / "history.jsonl",
            "event_log": session_dir / "logs" / f"run_{resolved_session_id}.jsonl",
        }

    def _validate_server_log_offsets(
        self,
        offsets: Any,
        session_id: str | None = None,
    ) -> Dict[str, int]:
        return self._validate_server_log_offsets_for_files(
            offsets, self._server_log_files(session_id)
        )

    def _restore_server_logs_before_server(self, offsets: Any) -> None:
        """Restore append-only simulator logs before EventLogger opens them."""
        validated = self._validate_server_log_offsets(offsets)
        for name, path in self._server_log_files().items():
            offset = validated[name]
            current_size = path.stat().st_size if path.exists() else 0
            if offset > current_size:
                raise ValueError(
                    f"Checkpoint server log offset for {name} exceeds file size: "
                    f"{offset} > {current_size}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a+b") as file:
                file.truncate(offset)
        # _meta.json 代表实验终态，恢复断点后它已经失效，不能留下旧结果。
        event_log = self._server_log_files()["event_log"]
        event_log.with_name(event_log.stem + "_meta.json").unlink(missing_ok=True)

    def _checkpoint_artifact_path(self, relative_path: str, label: str) -> Path:
        path = (self.workspace_dir / relative_path).resolve()
        workspace_root = self.workspace_dir.resolve()
        if workspace_root not in path.parents:
            raise ValueError(f"Checkpoint {label} path escapes run directory: {relative_path}")
        return path

    def _preflight_checkpoint_restore(
        self, checkpoint: Dict[str, Any]
    ) -> CheckpointRestorePlan:
        """Validate every persistent artifact before mutating the current run."""
        runtime = checkpoint['runtime']
        database = checkpoint['database']
        conversation = runtime['conversation']

        database_path = self._checkpoint_artifact_path(database['file'], 'database')
        if not database_path.is_file():
            raise FileNotFoundError(f"Checkpoint database not found: {database_path}")
        if self._sha256_file(database_path) != database['sha256']:
            raise ValueError("Checkpoint database hash mismatch")

        conversation_path = self._checkpoint_artifact_path(
            conversation['file'], 'conversation'
        )
        if not conversation_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint conversation not found: {conversation_path}"
            )
        if self._sha256_file(conversation_path) != conversation['sha256']:
            raise ValueError("Checkpoint conversation hash mismatch")
        conversation_payload = BashAgent.parse_checkpoint_snapshot(conversation_path)
        if conversation_payload['resume_conversation'] != conversation['resume']:
            raise ValueError("Checkpoint conversation resume state mismatch")
        if conversation_payload['total_turns'] != runtime['agent']['total_turns']:
            raise ValueError("Checkpoint conversation total_turns mismatch")
        session_dir = (
            self.agent_workspace / "sessions" / checkpoint['session_id']
        )
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint session directory not found: {session_dir}")
        session_meta = session_dir / "session.json"
        if not session_meta.is_file():
            raise FileNotFoundError(
                f"Checkpoint session metadata not found: {session_meta}"
            )
        try:
            metadata = json.loads(session_meta.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Checkpoint session metadata is invalid JSON") from exc
        if metadata.get('session_id') not in {None, checkpoint['session_id']}:
            raise ValueError("Checkpoint session metadata belongs to another session")

        commit = runtime['workspace_commit']
        if self._git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            raise ValueError(f"Agent workspace checkpoint commit does not exist: {commit}")

        runner_offsets = self._validate_runner_log_offsets(
            runtime['runner_log_offsets']
        )
        self._validate_offsets_within_files(
            runner_offsets, self._checkpoint_log_files(), "runner log"
        )
        session_id = checkpoint['session_id']
        server_files = {
            "history": session_dir / "history.jsonl",
            "event_log": session_dir / "logs" / f"run_{session_id}.jsonl",
        }
        server_offsets = self._validate_server_log_offsets_for_files(
            runtime['server_log_offsets'], server_files
        )
        self._validate_offsets_within_files(
            server_offsets, server_files, "server log"
        )
        return CheckpointRestorePlan(
            session_id=session_id,
            conversation_payload=conversation_payload,
        )

    @staticmethod
    def _validate_server_log_offsets_for_files(
        offsets: Any, files: Dict[str, Path]
    ) -> Dict[str, int]:
        expected_names = set(files)
        if not isinstance(offsets, dict) or set(offsets) != expected_names:
            raise ValueError(
                f"Checkpoint server log offsets must contain exactly: {sorted(expected_names)}"
            )
        for name, offset in offsets.items():
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                raise ValueError(
                    f"Invalid checkpoint server log offset for {name}: {offset!r}"
                )
        return dict(offsets)

    def _restore_workspace_commit(self, commit: str) -> None:
        if not isinstance(commit, str) or not commit:
            raise ValueError("Checkpoint does not contain an Agent workspace commit")
        exists = self._git("cat-file", "-e", f"{commit}^{{commit}}")
        if exists.returncode != 0:
            raise ValueError(f"Agent workspace checkpoint commit does not exist: {commit}")
        # 仅回退 Agent 自己的隔离工作区；sessions/ 等忽略目录不会被清理。
        self._git("reset", "--hard", commit, check=True)
        self._git("clean", "-fd", check=True)

    def _refresh_public_workspace_artifacts(self) -> None:
        """Refresh static client/docs after restoring Agent-authored workspace files."""
        public_dir = self._public_dir()
        src_docs = public_dir / "docs"
        dst_docs = self.agent_workspace / "docs"
        if src_docs.exists():
            if dst_docs.exists():
                shutil.rmtree(dst_docs, ignore_errors=True)
            shutil.copytree(
                src_docs,
                dst_docs,
                ignore=shutil.ignore_patterns('__pycache__'),
            )
        src_op = public_dir / "novamind-operation"
        dst_op = self.agent_workspace / "novamind-operation"
        if src_op.exists():
            shutil.copy2(src_op, dst_op)
            import stat as _stat
            dst_op.chmod(
                dst_op.stat().st_mode | _stat.S_IEXEC | _stat.S_IXGRP | _stat.S_IXOTH
            )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load and validate the exact checkpoint schema from disk."""
        checkpoint_file = self.workspace_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            return None
        with open(checkpoint_file) as file:
            checkpoint = json.load(file)
        checkpoint = self._validate_checkpoint(checkpoint)
        config_file = self.workspace_dir / "config.json"
        if not config_file.is_file():
            raise FileNotFoundError("Checkpoint run config is missing")
        if self._sha256_file(config_file) != checkpoint['run_config_sha256']:
            raise ValueError("Checkpoint run config hash mismatch")
        return checkpoint

    @staticmethod
    def _require_finite_number(value: Any, field: str) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid checkpoint {field}: {value!r}")
        return float(value)

    @classmethod
    def _require_non_negative_number(cls, value: Any, field: str) -> float:
        value = cls._require_finite_number(value, field)
        if value < 0:
            raise ValueError(f"Invalid checkpoint {field}: {value!r}")
        return value

    @staticmethod
    def _require_non_negative_integer(value: Any, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid checkpoint {field}: {value!r}")
        return value

    @classmethod
    def _validate_environment_llm_usage(cls, usage: Any) -> Dict[str, Any]:
        if not isinstance(usage, dict) or set(usage) != {
            'input_tokens', 'cached_tokens', 'output_tokens',
            'cost_by_currency', 'by_purpose'
        }:
            raise ValueError("Invalid environment LLM usage summary")
        cls._require_non_negative_integer(
            usage['input_tokens'], 'environment_llm.input_tokens'
        )
        cls._require_non_negative_integer(
            usage['cached_tokens'], 'environment_llm.cached_tokens'
        )
        if usage['cached_tokens'] > usage['input_tokens']:
            raise ValueError("Environment LLM cached tokens exceed input tokens")
        cls._require_non_negative_integer(
            usage['output_tokens'], 'environment_llm.output_tokens'
        )
        cls._validate_cost_by_currency(
            usage['cost_by_currency'], 'environment_llm.cost_by_currency'
        )
        by_purpose = usage['by_purpose']
        if not isinstance(by_purpose, dict):
            raise ValueError("Invalid environment LLM usage by_purpose")
        for purpose, values in by_purpose.items():
            if not isinstance(purpose, str) or not purpose:
                raise ValueError("Invalid environment LLM purpose")
            if not isinstance(values, dict) or set(values) != {
                'input_tokens', 'cached_tokens', 'output_tokens', 'cost_by_currency'
            }:
                raise ValueError(f"Invalid environment LLM usage for {purpose!r}")
            cls._require_non_negative_integer(
                values['input_tokens'], f'environment_llm.{purpose}.input_tokens'
            )
            cls._require_non_negative_integer(
                values['cached_tokens'], f'environment_llm.{purpose}.cached_tokens'
            )
            if values['cached_tokens'] > values['input_tokens']:
                raise ValueError(
                    f"Environment LLM cached tokens exceed input tokens for {purpose!r}"
                )
            cls._require_non_negative_integer(
                values['output_tokens'], f'environment_llm.{purpose}.output_tokens'
            )
            cls._validate_cost_by_currency(
                values['cost_by_currency'],
                f'environment_llm.{purpose}.cost_by_currency',
            )
        if usage['input_tokens'] != sum(v['input_tokens'] for v in by_purpose.values()):
            raise ValueError("Environment LLM input token total does not match by_purpose")
        if usage['cached_tokens'] != sum(v['cached_tokens'] for v in by_purpose.values()):
            raise ValueError("Environment LLM cached token total does not match by_purpose")
        if usage['output_tokens'] != sum(v['output_tokens'] for v in by_purpose.values()):
            raise ValueError("Environment LLM output token total does not match by_purpose")
        expected_costs: Dict[str, float] = {}
        for values in by_purpose.values():
            for currency, amount in values['cost_by_currency'].items():
                expected_costs[currency] = expected_costs.get(currency, 0.0) + amount
        if set(usage['cost_by_currency']) != set(expected_costs) or any(
            not math.isclose(
                usage['cost_by_currency'][currency], amount,
                rel_tol=1e-9, abs_tol=1e-12,
            )
            for currency, amount in expected_costs.items()
        ):
            raise ValueError("Environment LLM cost total does not match by_purpose")
        return usage

    @classmethod
    def _validate_analysis_usage(
        cls,
        usage: Any,
        *,
        max_day: int,
    ) -> Dict[str, Any]:
        scalar_fields = {
            "call_count",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "reasoning_tokens",
        }
        expected = scalar_fields | {
            "role_report_days",
            "state_portrait_days",
            "cost_by_currency",
            "by_role",
            "state_reconstruction",
        }
        if not isinstance(usage, dict) or set(usage) != expected:
            raise ValueError("Invalid Analysis usage summary")
        for field in ("role_report_days", "state_portrait_days"):
            days = usage[field]
            if (
                not isinstance(days, list)
                or any(
                    not isinstance(day, int)
                    or isinstance(day, bool)
                    or day < 0
                    or day > max_day
                    for day in days
                )
                or days != sorted(set(days))
            ):
                raise ValueError(f"Invalid Analysis {field}")
        if not set(usage["state_portrait_days"]).issubset(
            usage["role_report_days"]
        ):
            raise ValueError("Analysis state portraits require role reports")
        for field in scalar_fields:
            cls._require_non_negative_integer(usage[field], f"analysis.{field}")
        if usage["cached_tokens"] > usage["input_tokens"]:
            raise ValueError("Analysis cached tokens exceed input tokens")
        cls._validate_cost_by_currency(
            usage["cost_by_currency"], "analysis.cost_by_currency"
        )

        by_role = usage["by_role"]
        expected_roles = {role.value for role in Role}
        if not isinstance(by_role, dict) or set(by_role) != expected_roles:
            raise ValueError("Invalid Analysis usage by_role")
        bucket_fields = scalar_fields | {"cost_by_currency"}
        for role, values in by_role.items():
            if not isinstance(values, dict) or set(values) != bucket_fields:
                raise ValueError(f"Invalid Analysis usage for role {role!r}")
            for field in scalar_fields:
                cls._require_non_negative_integer(
                    values[field], f"analysis.{role}.{field}"
                )
            if values["cached_tokens"] > values["input_tokens"]:
                raise ValueError(
                    f"Analysis cached tokens exceed input tokens for role {role!r}"
                )
            cls._validate_cost_by_currency(
                values["cost_by_currency"],
                f"analysis.{role}.cost_by_currency",
            )

        state_usage = usage["state_reconstruction"]
        if not isinstance(state_usage, dict) or set(state_usage) != bucket_fields:
            raise ValueError("Invalid Analysis state_reconstruction usage")
        for field in scalar_fields:
            cls._require_non_negative_integer(
                state_usage[field], f"analysis.state_reconstruction.{field}"
            )
        if state_usage["cached_tokens"] > state_usage["input_tokens"]:
            raise ValueError("Analysis state reconstruction cached tokens exceed input tokens")
        cls._validate_cost_by_currency(
            state_usage["cost_by_currency"],
            "analysis.state_reconstruction.cost_by_currency",
        )

        usage_buckets = [*by_role.values(), state_usage]
        for field in scalar_fields:
            if usage[field] != sum(values[field] for values in usage_buckets):
                raise ValueError(f"Analysis {field} total does not match task breakdown")
        expected_costs: Dict[str, float] = {}
        for values in usage_buckets:
            for currency, amount in values["cost_by_currency"].items():
                expected_costs[currency] = expected_costs.get(currency, 0.0) + amount
        if set(usage["cost_by_currency"]) != set(expected_costs) or any(
            not math.isclose(
                usage["cost_by_currency"][currency],
                amount,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            for currency, amount in expected_costs.items()
        ):
            raise ValueError("Analysis cost total does not match task breakdown")
        return usage

    @classmethod
    def _validate_cost_by_currency(cls, value: Any, field: str) -> Dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(f"Invalid checkpoint {field}: {value!r}")
        for currency, amount in value.items():
            if not isinstance(currency, str) or not currency:
                raise ValueError(f"Invalid checkpoint {field} currency: {currency!r}")
            cls._require_non_negative_number(amount, f"{field}.{currency}")
        return value

    def _validate_checkpoint(self, checkpoint: Any) -> Dict[str, Any]:
        """Reject partial or legacy checkpoints before any state is modified."""
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint root must be an object")
        if checkpoint.get('format_version') != self.CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format_version: {checkpoint.get('format_version')!r}"
            )
        required_root = {
            'format_version', 'run_config_sha256', 'day', 'cash', 'session_id',
            'database', 'runtime',
        }
        if set(checkpoint) != required_root:
            raise ValueError(
                f"Checkpoint root fields must contain exactly: {sorted(required_root)}"
            )
        self._require_non_negative_integer(checkpoint['day'], 'day')
        self._require_finite_number(checkpoint['cash'], 'cash')
        config_sha256 = checkpoint['run_config_sha256']
        if (
            not isinstance(config_sha256, str)
            or len(config_sha256) != 64
            or any(char not in "0123456789abcdef" for char in config_sha256)
        ):
            raise ValueError("Invalid checkpoint run_config_sha256")
        if not isinstance(checkpoint['session_id'], str) or not checkpoint['session_id']:
            raise ValueError("Checkpoint session_id must be a non-empty string")

        database = checkpoint['database']
        if not isinstance(database, dict) or set(database) != {'file', 'sha256'}:
            raise ValueError("Checkpoint database must contain exactly file and sha256")
        if not all(isinstance(database[key], str) and database[key] for key in database):
            raise ValueError("Checkpoint database file and sha256 must be non-empty strings")

        runtime = checkpoint['runtime']
        required_runtime = {
            'runner_log_offsets', 'server_log_offsets', 'conversation',
            'workspace_commit', 'environment_llm', 'analysis', 'agent',
        }
        if not isinstance(runtime, dict) or set(runtime) != required_runtime:
            raise ValueError(
                f"Checkpoint runtime fields must contain exactly: {sorted(required_runtime)}"
            )
        self._validate_runner_log_offsets(runtime['runner_log_offsets'])
        self._validate_server_log_offsets(
            runtime['server_log_offsets'],
            checkpoint['session_id'],
        )
        if not isinstance(runtime['workspace_commit'], str) or not runtime['workspace_commit']:
            raise ValueError("Checkpoint workspace_commit must be a non-empty string")
        self._validate_environment_llm_usage(runtime['environment_llm'])
        self._validate_analysis_usage(runtime['analysis'], max_day=checkpoint['day'])

        conversation = runtime['conversation']
        if not isinstance(conversation, dict) or set(conversation) != {'file', 'sha256', 'resume'}:
            raise ValueError("Checkpoint conversation must contain file, sha256, and resume")
        if not isinstance(conversation['file'], str) or not conversation['file']:
            raise ValueError("Checkpoint conversation file must be a non-empty string")
        if not isinstance(conversation['sha256'], str) or not conversation['sha256']:
            raise ValueError("Checkpoint conversation sha256 must be a non-empty string")
        if not isinstance(conversation['resume'], bool):
            raise ValueError("Checkpoint conversation resume must be boolean")

        agent = runtime['agent']
        required_agent = {
            'total_turns', 'input_tokens', 'output_tokens', 'cached_tokens',
            'reasoning_tokens', 'decision_cost_by_currency',
        }
        if not isinstance(agent, dict) or set(agent) != required_agent:
            raise ValueError(
                f"Checkpoint agent fields must contain exactly: {sorted(required_agent)}"
            )
        for field in required_agent - {'decision_cost_by_currency'}:
            self._require_non_negative_integer(agent[field], f'agent.{field}')
        if agent['cached_tokens'] > agent['input_tokens']:
            raise ValueError("Agent cached tokens exceed input tokens")
        self._validate_cost_by_currency(
            agent['decision_cost_by_currency'], 'agent.decision_cost_by_currency'
        )
        return checkpoint

    def _restore_checkpoint_database_before_server(self, checkpoint: Dict):
        """Restore the checkpoint database before the simulator server starts."""
        cp_day = checkpoint['day']
        self._session_id = checkpoint['session_id']

        # 服务器启动时会把 world.nmdb 一次性读入内存，因此必须先恢复文件再启动服务器。
        database_file = checkpoint['database']['file']
        database_sha256 = checkpoint['database']['sha256']
        harness_nmdb = self._checkpoint_artifact_path(database_file, 'database')
        session_nmdb = self.agent_workspace / "sessions" / self._session_id / "world.nmdb"
        if self._sha256_file(harness_nmdb) != database_sha256:
            raise ValueError("Checkpoint database hash mismatch")
        if not session_nmdb.parent.is_dir():
            raise FileNotFoundError(f"Checkpoint session directory not found: {session_nmdb.parent}")
        shutil.copy2(harness_nmdb, session_nmdb)
        print(f"  Restored DB from checkpoint (day {cp_day})")

        # 会话元数据也必须在服务器启动前回退，否则服务器会使用错误的当前日期。
        session_meta = self.agent_workspace / "sessions" / self._session_id / "session.json"
        if not session_meta.is_file():
            raise FileNotFoundError(f"Checkpoint session metadata not found: {session_meta}")
        meta = json.loads(session_meta.read_text())
        meta["current_day"] = cp_day
        meta["status"] = "created"  # Will be set to "running" when server starts
        meta.pop("port", None)
        meta.pop("pid", None)
        write_json_atomic(session_meta, meta)

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
        self._last_committed_week = checkpoint['day'] // 7

        conversation_payload = restore_plan.conversation_payload
        if self.agent:
            self.agent.restore_checkpoint_snapshot(conversation_payload)

    def _launch_server_from_prepared_checkpoint(self):
        """Restore persistent simulator state, then launch the server."""
        if self._resume_checkpoint:
            runtime = self._resume_checkpoint.get('runtime')
            if not isinstance(runtime, dict):
                raise ValueError("Checkpoint lacks exact runtime state and cannot be safely resumed")
            # 预检只读：所有文件、哈希、Git commit 和日志边界都通过后才应用恢复。
            restore_plan = self._preflight_checkpoint_restore(self._resume_checkpoint)
            self._checkpoint_restore_plan = restore_plan
            self._session_id = restore_plan.session_id
            self._restore_workspace_commit(runtime['workspace_commit'])
            # Git 回退只负责 Agent 产物；静态客户端必须与当前 host 端 bundle 对齐。
            self._refresh_public_workspace_artifacts()
            self._restore_checkpoint_database_before_server(self._resume_checkpoint)
            self._restore_logs_to_offsets(runtime['runner_log_offsets'])
            # EventLogger 启动后会以 append 模式打开文件，必须在启动前回退。
            self._restore_server_logs_before_server(runtime['server_log_offsets'])
            self._prune_analysis_artifacts_after(
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
        self.workspace_base.mkdir(parents=True, exist_ok=True)
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
            # 模拟器 bundle 也是实验条件，恢复前必须与原实验完全一致。
            self._verify_public_bundle()

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
                api_socket_path=self._server_socket_path,
            )

            tool_descriptions = get_bash_agent_tool_descriptions()

            self.agent = BashAgent(
                tool_descriptions=tool_descriptions,
                client=self.client,
                model=self.model,
                api_type=self.api_type,
                max_invalid_responses_per_turn=self.max_invalid_responses_per_turn,
                response_callback=self._log_response,
                reasoning_effort=self.reasoning_effort,
                temperature=self.temperature,
                top_p=self.top_p,
                tool_choice=self.tool_choice,
                max_output_tokens=self.max_output_tokens,
                timeout_seconds=self.timeout_seconds,
                request_options=self.request_options,
                tool_result_callback=self._log_tool_result,
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
        self._ensure_harness_identity()
        return {
            'format_version': RUN_CONFIG_FORMAT_VERSION,
            'run_id': self.run_id,
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
            'label': self.label,
            'simulator_llm': self.simulator_llm_config,
            'analysis_module': self.analysis_module_config,
            'analysis_model': self.analysis_model_config,
            'public_bundle_sha256': self._current_public_bundle_sha256(),
            'harness_git_commit': self.harness_git_commit,
            'harness_git_dirty': self.harness_git_dirty,
            'harness_source_sha256': self.harness_source_sha256,
        }

    def _write_new_run_config(self) -> None:
        config_file = self.workspace_dir / "config.json"
        if config_file.exists():
            raise FileExistsError(f"New run config already exists: {config_file}")
        # 配置是实验身份，只在新实验启动时提交一次；恢复过程禁止覆盖。
        payload = self._run_config_payload()
        write_json_atomic(config_file, payload)
        self.public_bundle_sha256 = payload['public_bundle_sha256']

    def _current_public_bundle_sha256(self) -> str:
        public_dir = self._public_dir()
        bundle = public_dir / "novamind-operation"
        docs = public_dir / "docs"
        if not bundle.is_file() or not docs.is_dir():
            raise FileNotFoundError(
                f"Public bundle must contain novamind-operation and docs/: {public_dir}"
            )

        # 可执行文件和 Agent 可见文档共同决定一次实验的输入。
        digest = hashlib.sha256()
        for path in [bundle, *sorted(path for path in docs.rglob('*') if path.is_file())]:
            relative_path = path.relative_to(public_dir).as_posix().encode()
            digest.update(len(relative_path).to_bytes(4, 'big'))
            digest.update(relative_path)
            with open(path, 'rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
        return digest.hexdigest()

    def _current_harness_source_sha256(self) -> str:
        """Hash the source that controls the main Agent experiment."""
        source_root = package_root / "saas_bench"
        files = [
            source_root / "agents" / "base.py",
            source_root / "environment.py",
            source_root / "experiment_config.py",
            source_root / "json_io.py",
            source_root / "llm_provider.py",
            *sorted((source_root / "agents" / "bash_agent").rglob("*.py")),
        ]
        digest = hashlib.sha256()
        # 路径和内容共同决定 Harness 身份，文件改名也会产生新哈希。
        for path in sorted(set(files)):
            if not path.is_file():
                raise FileNotFoundError(f"Harness source file is missing: {path}")
            relative_path = path.relative_to(package_root.parent).as_posix().encode()
            digest.update(len(relative_path).to_bytes(4, "big"))
            digest.update(relative_path)
            with open(path, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _ensure_harness_identity(self) -> None:
        if getattr(self, "harness_git_commit", None) is None:
            repo_root = package_root.parent
            self.harness_git_commit = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.harness_git_dirty = bool(status.strip())
            self.harness_source_sha256 = self._current_harness_source_sha256()
        self._validate_harness_identity({
            "harness_git_commit": self.harness_git_commit,
            "harness_git_dirty": self.harness_git_dirty,
            "harness_source_sha256": self.harness_source_sha256,
        })

    @staticmethod
    def _validate_harness_identity(identity: Dict[str, Any]) -> None:
        commit = identity.get("harness_git_commit")
        dirty = identity.get("harness_git_dirty")
        digest = identity.get("harness_source_sha256")
        if not isinstance(commit, str) or not commit:
            raise ValueError("harness_git_commit must be a non-empty string")
        if not isinstance(dirty, bool):
            raise ValueError("harness_git_dirty must be a boolean")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("harness_source_sha256 must be a lowercase SHA-256 digest")

    def _harness_result_fields(self) -> Dict[str, Any]:
        self._ensure_harness_identity()
        return {
            "harness_git_commit": self.harness_git_commit,
            "harness_git_dirty": self.harness_git_dirty,
            "harness_source_sha256": self.harness_source_sha256,
        }

    def _verify_public_bundle(self) -> None:
        if self.continue_from and self.public_bundle_sha256 is None:
            raise ValueError("Resumed run does not contain a public bundle hash")
        current = self._current_public_bundle_sha256()
        if self.public_bundle_sha256 is not None and current != self.public_bundle_sha256:
            raise ValueError(
                "Public simulator bundle hash does not match the original experiment"
            )

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
            **self._harness_result_fields(),
        }

    def _load_or_rebuild_terminal_result(self) -> Optional[Dict[str, Any]]:
        """Return a completed result before starting any managed resources."""
        if not self.continue_from:
            return None

        result_file = self.workspace_dir / "result.json"
        existing_result = None
        if result_file.exists():
            existing_result = json.loads(result_file.read_text())
            if existing_result.get('outcome') not in {'completed', 'bankrupt'}:
                # timeout/incomplete 只描述上一次调用，保留作为诊断记录即可。
                existing_result = None

        checkpoint = self._load_checkpoint()
        if checkpoint is None:
            if existing_result is not None:
                raise RuntimeError("Terminal result exists without a checkpoint")
            return None
        session_id = checkpoint['session_id']
        session_dir = self.agent_workspace / "sessions" / session_id
        session_meta_file = session_dir / "session.json"
        event_meta_file = session_dir / "logs" / f"run_{session_id}_meta.json"
        if not session_meta_file.is_file():
            if existing_result is not None:
                raise RuntimeError("Terminal result exists without session metadata")
            return None

        session_meta = json.loads(session_meta_file.read_text())
        session_outcome = session_meta.get('status')
        event_meta = (
            json.loads(event_meta_file.read_text())
            if event_meta_file.is_file()
            else None
        )
        event_outcome = event_meta.get('outcome') if event_meta else None
        terminal_outcomes = {'completed', 'bankrupt'}
        if session_outcome not in terminal_outcomes and event_outcome not in terminal_outcomes:
            if existing_result is not None:
                raise RuntimeError("Terminal result has no matching terminal metadata")
            return None
        if (
            session_outcome in terminal_outcomes
            and event_outcome in terminal_outcomes
            and session_outcome != event_outcome
        ):
            raise RuntimeError(
                "Terminal run artifacts disagree; refusing to resume"
            )
        if session_outcome not in terminal_outcomes or event_outcome not in terminal_outcomes:
            if existing_result is not None:
                raise RuntimeError("Terminal result conflicts with incomplete finalization")
            # 终态提交在多文件之间中断：返回断点后重做 finalize，不把半成品当成损坏。
            return None

        outcome = session_outcome
        day = checkpoint['day']
        cash = checkpoint['cash']
        if session_meta.get('current_day') != day:
            raise RuntimeError("Terminal session day does not match checkpoint")
        if session_meta.get('final_cash') != cash:
            raise RuntimeError("Terminal session cash does not match checkpoint")
        if event_meta.get('days_run') != day:
            raise RuntimeError("Terminal event day does not match checkpoint")
        if event_meta.get('final_cash') != cash:
            raise RuntimeError("Terminal event cash does not match checkpoint")
        if outcome == 'completed' and day < self.total_days:
            raise RuntimeError("Completed terminal result is before the target day")
        if outcome == 'completed' and cash < 0:
            raise RuntimeError("Completed terminal result has negative cash")
        if outcome == 'bankrupt' and cash >= 0:
            raise RuntimeError("Bankrupt terminal result has non-negative cash")

        # 事件 meta 与 JSONL 尾部必须同时存在 run_end，防止只剩一份覆盖写文件。
        event_log_file = session_dir / "logs" / f"run_{session_id}.jsonl"
        if not event_log_file.is_file():
            raise RuntimeError("Terminal event log is missing")
        last_event = None
        with open(event_log_file) as event_log:
            for line in event_log:
                if line.strip():
                    last_event = json.loads(line)
        if (
            not isinstance(last_event, dict)
            or last_event.get('category') != 'run_end'
            or last_event.get('day') != day
            or last_event.get('details', {}).get('outcome') != outcome
            or last_event.get('details', {}).get('final_cash') != cash
        ):
            raise RuntimeError("Terminal event log does not match terminal metadata")

        canonical_result = self._result_from_checkpoint(checkpoint, outcome)
        if existing_result is not None:
            # result.json 只是可重建索引，不能覆盖 checkpoint 与服务端终态证据。
            if existing_result != canonical_result:
                raise RuntimeError("Terminal result does not match authoritative artifacts")
            return existing_result
        self._write_result(canonical_result)
        return canonical_result

    def _repair_terminal_checkpoint_after_setup(self) -> Optional[Dict[str, Any]]:
        """Finish a terminal checkpoint whose prior finalization was interrupted."""
        checkpoint = self._resume_checkpoint
        if checkpoint is None:
            return None
        if checkpoint['cash'] < 0:
            outcome = 'bankrupt'
        elif checkpoint['day'] >= self.total_days:
            outcome = 'completed'
        else:
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
        self._start_timing_poster()
        try:
            return self._run_experiment(verbose)
        finally:
            # 主循环任何位置失败，都必须先关模拟器，再停止日志转发线程。
            self._stop_server()
            self._stop_timing_poster()

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
            print(f"API Server Port: {self._server_port}")
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

            if verbose:
                print(f"\n{'='*40}")
                print(f"DECISION BATCH {decision_batch} (sim day {sim_day})")
                print(f"{'='*40}")

            # Dashboard 文本和 Analysis 使用同一个公开结构化快照。
            _t0 = _time.monotonic()
            dashboard_payload = self._get_dashboard_payload()
            dashboard = dashboard_payload['dashboard']
            _dashboard_elapsed = _time.monotonic() - _t0
            self._log_tool_result(0, sim_day, '_dashboard', {}, dashboard)
            self._log_timing("dashboard", sim_day, elapsed_s=round(_dashboard_elapsed, 3))

            _analysis_started = _time.monotonic()
            signals = self._ensure_analysis_signals(dashboard_payload)
            role_reports_generated = False
            state_portrait_generated = False
            brief_generated = False
            analysis_brief = None
            if signals is not None:
                role_reports, role_reports_generated = (
                    self._ensure_analysis_role_reports(signals)
                )
                if role_reports is None:
                    raise RuntimeError("Analysis role reports were not generated")
                if role_reports_generated:
                    # 四个角色调用完成后先提交断点；状态重构失败时无需重复付费。
                    stable_checkpoint = self._save_checkpoint(sim_day)
                state_portrait, state_portrait_generated = (
                    self._ensure_analysis_state_portrait(role_reports)
                )
                if state_portrait is None:
                    raise RuntimeError("Analysis state portrait was not generated")
                analysis_brief, brief_generated = self._ensure_analysis_brief(
                    role_reports,
                    state_portrait,
                )
            if self.analysis_enabled:
                self._log_timing(
                    "analysis_week",
                    sim_day,
                    elapsed_s=round(_time.monotonic() - _analysis_started, 3),
                    role_reports_generated=role_reports_generated,
                    state_portrait_generated=state_portrait_generated,
                    brief_generated=brief_generated,
                )
            if state_portrait_generated:
                # 状态画像和本周汇总日志完成后再次提交，形成完整 Analysis 断点。
                stable_checkpoint = self._save_checkpoint(sim_day)

            # Agent Loop：只要本周的决策尚未结束，就持续执行
            observation = self._decision_observation(dashboard, analysis_brief)
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
                _before_cost = dict(self.total_decision_agent_cost_by_currency)
                _t0 = _time.monotonic()
                action = self.agent.act(observation, 0, False, info)
                _llm_elapsed = _time.monotonic() - _t0
                _call_count = self.agent.total_turns - _before_total_turns
                _turn_input_tokens = self.agent.total_input_tokens - _before_input_tokens
                _turn_output_tokens = self.agent.total_output_tokens - _before_output_tokens
                _turn_cached_tokens = self.agent.total_cached_tokens - _before_cached_tokens
                _turn_reasoning_tokens = self.agent.total_reasoning_tokens - _before_reasoning_tokens
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

                # 将本次 LLM 的调用信息(llm_call)记录到日志文件（耗时、使用工具、Token 使用情况、请求的模型等）
                call_cost_by_currency = {
                    currency: amount - _before_cost.get(currency, 0.0)
                    for currency, amount in self.total_decision_agent_cost_by_currency.items()
                    if not math.isclose(amount, _before_cost.get(currency, 0.0))
                }
                self._log_timing("llm_call", sim_day, turn=turns_in_batch,
                                 elapsed_s=round(_llm_elapsed, 2),
                                 tool=tool_name, tool_preview=tool_args_preview,
                                 api_calls=_call_count,
                                 input_tokens=_turn_input_tokens,
                                 output_tokens=_turn_output_tokens,
                                 cached_tokens=_turn_cached_tokens,
                                 reasoning_tokens=_turn_reasoning_tokens,
                                 cost_by_currency=call_cost_by_currency,
                                 total_cost_by_currency=self.total_decision_agent_cost_by_currency,
                                 requested_model=self.model,
                                 served_model=self.agent.last_serving_model,
                                 pricing_model=self.pricing_model_map.get(
                                     self.agent.last_serving_model,
                                     self.agent.last_serving_model,
                                 ))

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
                    print(f"\n⚠️  next_week timed out on sim day {sim_day} ({e})")
                    print("Auto-quitting. Keeping the previous completed checkpoint.")
                    game_outcome = 'timeout'
                    break
                _tool_elapsed = _time.monotonic() - _t0
                batch_tool_s += _tool_elapsed
                observation = result if isinstance(result, str) else json.dumps(result)

                # 将工具调用耗时信息(tool_exec)记录到日志文件
                self._log_timing("tool_exec", sim_day, turn=turns_in_batch,
                                 elapsed_s=round(_tool_elapsed, 3),
                                 tool=tool_name, tool_preview=tool_args_preview)

                # 将工具调用执行结果信息(tool_result)记录到日志文件
                self._log_tool_result(
                    self.agent.total_turns, sim_day,
                    action.tool, action.arguments or {},
                    observation  # Full result in JSONL (tool already caps at 50K)
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
                self._commit_weeks_up_to(sim_day)       # next-week 后，将 Agent 工具目录的改动提交到 Git

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

            if game_outcome is not None:
                break

            # 达到配置上限仍未推进时保留同一模拟日，不伪造缺少理由和预测参数的 next-week。
            # 当前上下文会随 checkpoint 一起保存，下一轮可以继续决策。
            if not week_advanced:
                print(
                    f"\n⚠️  Turn cap reached on sim day {sim_day} without next-week; "
                    "saving a resumable checkpoint and ending this invocation."
                )
                self._log_timing(
                    "turn_cap_no_advance", sim_day, turns=turns_in_batch
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
            self._log_timing("decision_batch_summary", sim_day,
                             decision_batch=decision_batch,
                             elapsed_s=round(batch_elapsed_s, 1),
                             llm_total_s=round(batch_llm_s, 1),
                             agent_tool_s=round(agent_tool_s, 1),
                             environment_advance_s=round(environment_advance_s, 1),
                             dashboard_s=round(_dashboard_elapsed, 2),
                             other_s=round(max(batch_other_s, 0), 1),
                             turns=turns_in_batch,
                             subs=subscribers,
                             cash=cash,
                             batch_input_tokens=batch_input_tokens,
                             batch_output_tokens=batch_output_tokens,
                             batch_cached_tokens=batch_cached_tokens,
                             batch_reasoning_tokens=batch_reasoning_tokens,
                             total_input_tokens=self.agent.total_input_tokens,
                             total_output_tokens=self.agent.total_output_tokens,
                             total_cached_tokens=self.agent.total_cached_tokens,
                             total_reasoning_tokens=self.agent.total_reasoning_tokens)

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
                resume_conversation=not week_advanced,
                pending_observation=observation if not week_advanced else None,
            )

            if not week_advanced:
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
                if self.analysis_enabled:
                    # 最后一周没有下一轮决策，但仍生成完整 Analysis 产物。
                    final_analysis_started = _time.monotonic()
                    final_payload = self._get_dashboard_payload()
                    final_signals = self._ensure_analysis_signals(final_payload)
                    if final_signals is None:
                        raise RuntimeError("Analysis signals were not generated")
                    final_reports, final_reports_generated = (
                        self._ensure_analysis_role_reports(final_signals)
                    )
                    if final_reports is None:
                        raise RuntimeError("Analysis role reports were not generated")
                    if final_reports_generated:
                        # 终态重构失败时，四角色调用仍可从这个断点恢复。
                        self._save_checkpoint(sim_day)
                    final_portrait, final_portrait_generated = (
                        self._ensure_analysis_state_portrait(final_reports)
                    )
                    if final_portrait is None:
                        raise RuntimeError("Analysis state portrait was not generated")
                    _, final_brief_generated = self._ensure_analysis_brief(
                        final_reports,
                        final_portrait,
                    )
                    self._log_timing(
                        "analysis_week",
                        sim_day,
                        elapsed_s=round(
                            _time.monotonic() - final_analysis_started, 3
                        ),
                        role_reports_generated=final_reports_generated,
                        state_portrait_generated=final_portrait_generated,
                        brief_generated=final_brief_generated,
                        terminal=True,
                    )
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


def _new_experiment_runner(config_path: Path) -> BashAgentRunner:
    file_config = load_experiment_config(config_path)
    experiment = file_config.experiment
    decision = file_config.decision_agent
    return BashAgentRunner(
        model=decision.model,
        provider=decision.provider,
        api_type=decision.api_type,
        base_url=decision.base_url,
        api_key_env=decision.api_key_env,
        api_key_required=decision.api_key_required,
        seed=experiment.seed,
        scenario=experiment.scenario,
        total_days=experiment.days,
        initial_cash=experiment.initial_cash,
        max_decision_turns_per_batch=experiment.max_decision_turns_per_batch,
        max_invalid_responses_per_turn=experiment.max_invalid_responses_per_turn,
        workspace_base=Path(experiment.workspace),
        reasoning_effort=decision.reasoning_effort,
        temperature=decision.temperature,
        top_p=decision.top_p,
        tool_choice=decision.tool_choice,
        max_output_tokens=decision.max_output_tokens,
        timeout_seconds=decision.timeout_seconds,
        request_options=decision.request_options,
        pricing=decision.pricing,
        pricing_model_map=decision.pricing_model_map,
        simulator_llm_config=file_config.simulator_overrides(),
        analysis_module_config=asdict(file_config.modules.analysis),
        analysis_model_config=(
            file_config.analysis.as_dict() if file_config.analysis else None
        ),
        label=experiment.label,
    )


def _resolve_resume_dir(value: str) -> Path:
    direct = Path(value).expanduser()
    if direct.is_dir():
        return direct.resolve()

    run_name = value if value.startswith("run_") else f"run_{value}"
    candidates = []
    for root, dirs, files in os.walk(Path.cwd()):
        dirs[:] = [name for name in dirs if name not in {".git", ".venv", "__pycache__", "tmp"}]
        path = Path(root)
        if path.name == run_name and "config.json" in files:
            candidates.append(path.resolve())
            dirs[:] = []
    if not candidates:
        raise FileNotFoundError(f"No run directory found for resume id {value!r}")
    if len(candidates) > 1:
        joined = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Resume id {value!r} is ambiguous; pass one directory: {joined}")
    return candidates[0]


def _load_saved_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = Path(run_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    try:
        saved = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run config is invalid JSON: {config_path}") from exc
    if not isinstance(saved, dict) or set(saved) != RUN_CONFIG_FIELDS:
        missing = sorted(RUN_CONFIG_FIELDS - set(saved)) if isinstance(saved, dict) else []
        extra = sorted(set(saved) - RUN_CONFIG_FIELDS) if isinstance(saved, dict) else []
        raise ValueError(
            f"Run config fields do not match format {RUN_CONFIG_FORMAT_VERSION}; "
            f"missing={missing}, extra={extra}"
        )
    if saved["format_version"] != RUN_CONFIG_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported run config format_version: {saved['format_version']!r}"
        )
    if saved["agent_type"] != "bash_agent":
        raise ValueError(f"Run config is not for bash_agent: {saved['agent_type']!r}")
    if not isinstance(saved["run_id"], str) or not saved["run_id"]:
        raise ValueError("Run config run_id must be a non-empty string")
    bundle_sha256 = saved["public_bundle_sha256"]
    if (
        not isinstance(bundle_sha256, str)
        or len(bundle_sha256) != 64
        or any(char not in "0123456789abcdef" for char in bundle_sha256)
    ):
        raise ValueError("Run config public_bundle_sha256 is invalid")
    harness_digest = saved["harness_source_sha256"]
    if (
        not isinstance(saved["harness_git_commit"], str)
        or not saved["harness_git_commit"]
        or not isinstance(saved["harness_git_dirty"], bool)
        or not isinstance(harness_digest, str)
        or len(harness_digest) != 64
        or any(char not in "0123456789abcdef" for char in harness_digest)
    ):
        raise ValueError("Run config harness identity is invalid")
    return saved


def _resume_runner(value: str) -> BashAgentRunner:
    run_dir = _resolve_resume_dir(value)
    saved = _load_saved_run_config(run_dir)
    return BashAgentRunner(
        model=saved["model"],
        provider=saved["provider"],
        api_type=saved["api_type"],
        base_url=saved["base_url"],
        api_key_env=saved["api_key_env"],
        api_key_required=saved["api_key_required"],
        seed=saved["seed"],
        scenario=saved["scenario"],
        total_days=saved["total_days"],
        initial_cash=saved["initial_cash"],
        max_decision_turns_per_batch=saved["max_decision_turns_per_batch"],
        max_invalid_responses_per_turn=saved["max_invalid_responses_per_turn"],
        reasoning_effort=saved["reasoning_effort"],
        temperature=saved["temperature"],
        top_p=saved["top_p"],
        tool_choice=saved["tool_choice"],
        max_output_tokens=saved["max_output_tokens"],
        timeout_seconds=saved["timeout_seconds"],
        request_options=saved["request_options"],
        pricing=saved["pricing"],
        pricing_model_map=saved["pricing_model_map"],
        simulator_llm_config=saved["simulator_llm"],
        analysis_module_config=saved["analysis_module"],
        analysis_model_config=saved["analysis_model"],
        public_bundle_sha256=saved["public_bundle_sha256"],
        harness_git_commit=saved["harness_git_commit"],
        harness_git_dirty=saved["harness_git_dirty"],
        harness_source_sha256=saved["harness_source_sha256"],
        continue_from=run_dir,
        label=saved.get("label"),
    )


def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Run bash agent for SaaS Bench")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help=(
            "Complete TOML configuration for a new experiment "
            f"(default: {DEFAULT_EXPERIMENT_CONFIG})"
        ),
    )
    mode.add_argument(
        "--resume",
        help="Resume a run by run id or run directory using its saved configuration",
    )
    args = parser.parse_args(argv)
    runner = (
        _resume_runner(args.resume)
        if args.resume
        else _new_experiment_runner(args.config)
    )
    result = runner.run(verbose=True)
    print(f"\nResult: {result['outcome']}")
    print(f"Final Cash: ${result['final_cash']:,.0f}")
    print(f"Workspace: {result['workspace_dir']}")


if __name__ == "__main__":
    main()
