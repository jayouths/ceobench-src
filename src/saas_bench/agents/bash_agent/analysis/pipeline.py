"""Analysis 模块的周度编排、模型调用和产物管理。"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

from saas_bench.experiment.json_io import write_json_atomic, write_text_atomic
from saas_bench.experiment.llm_provider import call_text_model, model_token_cost

from .brief import render_strategy_brief
from .models import (
    AnalysisCallKind,
    Role,
    RoleCallUsage,
    RoleReportsArtifact,
    StateCallUsage,
    StatePortraitArtifact,
)
from .signal_models import AnalysisSignals
from .role_reports import RoleCallOutcome, RoleReportGenerator
from .signals import SignalCollector, parse_public_week_snapshot
from .state_reconstruction import StateCallOutcome, StateReconstructor


class AnalysisPipeline:
    """将 Analysis 的五次周度调用与通用实验 Runner 解耦。"""

    def __init__(
        self,
        *,
        enabled: bool,
        module_config: dict[str, Any],
        model_config: dict[str, Any] | None,
        client: Any,
        workspace_dir: Path,
        query_public_rows: Callable[[str], list[dict[str, Any]]],
        log_trajectory: Callable[..., None],
    ) -> None:
        self.enabled = enabled
        self.module_config = module_config
        self.model_config = model_config
        self.client = client
        self.workspace_dir = workspace_dir
        self.query_public_rows = query_public_rows
        self.log_trajectory = log_trajectory

    def signal_path(self, day: int) -> Path:
        return self.workspace_dir / "analysis" / f"day_{day:03d}" / "signals.json"

    def role_reports_path(self, day: int) -> Path:
        return (
            self.workspace_dir
            / "analysis"
            / f"day_{day:03d}"
            / "role_reports.json"
        )

    def state_portrait_path(self, day: int) -> Path:
        return (
            self.workspace_dir
            / "analysis"
            / f"day_{day:03d}"
            / "state_portrait.json"
        )

    def brief_path(self, day: int) -> Path:
        return (
            self.workspace_dir
            / "analysis"
            / f"day_{day:03d}"
            / "STRATEGY_BRIEF.md"
        )

    def load_history(self, before_or_at_day: int) -> dict[int, AnalysisSignals]:
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

    def ensure_signals(
        self, dashboard_payload: dict[str, Any]
    ) -> AnalysisSignals | None:
        """同一模拟日只生成一次信号；恢复同周时复用原产物。"""
        if not self.enabled:
            return None
        snapshot = parse_public_week_snapshot(
            dashboard_payload.get("public_week_snapshot")
        )
        path = self.signal_path(snapshot.day)
        if path.is_file():
            signals = AnalysisSignals.model_validate_json(path.read_text())
            if signals.day != snapshot.day:
                raise ValueError(f"Analysis artifact day mismatch: {path}")
            return signals

        history = self.load_history(snapshot.day - 1)
        collector = SignalCollector(
            self.query_public_rows,
            max_enterprise_threads=self.module_config["max_enterprise_threads"],
        )
        signals = collector.collect(snapshot, history)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, signals.model_dump(mode="json"))
        return signals

    def task_parameters(self, task: str) -> dict[str, Any]:
        """合并 Analysis 模型公共参数与任务级覆盖。"""
        config = self.model_config
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
    def _jsonable_response(raw_response: Any) -> Any:
        if hasattr(raw_response, "model_dump"):
            return raw_response.model_dump(
                mode="json", exclude_none=False, by_alias=True
            )
        if isinstance(raw_response, (dict, list, str, int, float, bool)):
            return raw_response
        return str(raw_response)

    def call_model(
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
        """统一执行 Analysis 调用，确保任务共用计费和日志口径。"""
        if self.client is None or self.model_config is None:
            raise RuntimeError("analysis client is not initialized")
        config = self.model_config
        started = time.monotonic()
        response = call_text_model(
            client=self.client,
            api_type=config["api_type"],
            model=config["model"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **self.task_parameters(task),
        )
        elapsed = time.monotonic() - started
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
        }
        if role is not None:
            identity["role"] = role.value
        self.log_trajectory(
            "llm_call",
            day,
            **identity,
            status="completed",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response.text,
            raw_response=self._jsonable_response(response.raw_response),
            elapsed_seconds=elapsed,
            provider=config["provider"],
            api_type=config["api_type"],
            requested_model=config["model"],
            served_model=response.model,
            pricing_model=cost.pricing_model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            reasoning_tokens=response.reasoning_tokens,
            cost_amount=cost.amount,
            currency=cost.currency,
        )
        return response, cost, elapsed

    def call_role_model(
        self,
        day: int,
        role: Role,
        attempt: int,
        call_kind: AnalysisCallKind,
        system_prompt: str,
        user_prompt: str,
    ) -> RoleCallOutcome:
        response, cost, elapsed = self.call_model(
            task="role_report",
            day=day,
            attempt=attempt,
            call_kind=call_kind,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            role=role,
        )
        return RoleCallOutcome(
            text=response.text,
            usage=RoleCallUsage(
                role=role,
                attempt=attempt,
                call_kind=call_kind,
                requested_model=self.model_config["model"],
                served_model=response.model,
                pricing_model=cost.pricing_model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_tokens=response.cached_tokens,
                reasoning_tokens=response.reasoning_tokens,
                elapsed_seconds=elapsed,
                cost_amount=cost.amount,
                currency=cost.currency,
            ),
        )

    def call_state_model(
        self,
        day: int,
        attempt: int,
        call_kind: AnalysisCallKind,
        system_prompt: str,
        user_prompt: str,
    ) -> StateCallOutcome:
        response, cost, elapsed = self.call_model(
            task="state_reconstruction",
            day=day,
            attempt=attempt,
            call_kind=call_kind,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return StateCallOutcome(
            text=response.text,
            usage=StateCallUsage(
                attempt=attempt,
                call_kind=call_kind,
                requested_model=self.model_config["model"],
                served_model=response.model,
                pricing_model=cost.pricing_model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_tokens=response.cached_tokens,
                reasoning_tokens=response.reasoning_tokens,
                elapsed_seconds=elapsed,
                cost_amount=cost.amount,
                currency=cost.currency,
            ),
        )

    def ensure_role_reports(
        self, signals: AnalysisSignals
    ) -> tuple[RoleReportsArtifact | None, bool]:
        if not self.enabled:
            return None, False
        path = self.role_reports_path(signals.day)
        if path.is_file():
            artifact = RoleReportsArtifact.model_validate_json(path.read_text())
            if artifact.day != signals.day:
                raise ValueError(f"Analysis role report day mismatch: {path}")
            return artifact, False

        generator = RoleReportGenerator(
            self.call_role_model,
            max_schema_retries=self.module_config["max_schema_retries"],
        )
        artifact = generator.generate(signals)
        write_json_atomic(path, artifact.model_dump(mode="json"))
        return artifact, True

    def ensure_state_portrait(
        self, role_reports: RoleReportsArtifact
    ) -> tuple[StatePortraitArtifact | None, bool]:
        if not self.enabled:
            return None, False
        path = self.state_portrait_path(role_reports.day)
        if path.is_file():
            artifact = StatePortraitArtifact.model_validate_json(path.read_text())
            if artifact.day != role_reports.day:
                raise ValueError(f"Analysis state portrait day mismatch: {path}")
            return artifact, False

        reconstructor = StateReconstructor(
            self.call_state_model,
            max_schema_retries=self.module_config["max_schema_retries"],
        )
        artifact = reconstructor.generate(role_reports)
        write_json_atomic(path, artifact.model_dump(mode="json"))
        return artifact, True

    def ensure_brief(
        self,
        role_reports: RoleReportsArtifact,
        portrait: StatePortraitArtifact,
    ) -> tuple[str | None, bool]:
        """生成确定性状态简报；关闭 Analysis 时不产生文件。"""
        if not self.enabled:
            return None, False
        path = self.brief_path(portrait.day)
        if path.is_file():
            return path.read_text(), False
        brief = render_strategy_brief(role_reports, portrait)
        write_text_atomic(path, brief)
        return brief, True

    def decision_observation(self, dashboard: str, brief: str | None) -> str:
        """只在 Analysis 开启时向原始 Dashboard 追加状态简报。"""
        if not self.enabled:
            return dashboard
        if not brief:
            raise RuntimeError("Analysis brief is required when Analysis is enabled")
        return f"{dashboard}\n\n---\n\n{brief}"

    def usage_summary(self, before_or_at_day: int) -> dict[str, Any]:
        """从已落盘周产物重建累计用量，避免维护第二份可变计数器。"""

        def empty_usage() -> dict[str, Any]:
            return {
                "call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "cost_by_currency": {},
            }

        def add_call(target: dict[str, Any], call: Any) -> None:
            target["call_count"] += 1
            for field in ("input_tokens", "output_tokens", "cached_tokens"):
                target[field] += getattr(call, field)
            reasoning_tokens = call.reasoning_tokens
            if target["reasoning_tokens"] is not None:
                target["reasoning_tokens"] = (
                    target["reasoning_tokens"] + reasoning_tokens
                    if reasoning_tokens is not None
                    else None
                )
            costs = target["cost_by_currency"]
            costs[call.currency] = costs.get(call.currency, 0.0) + call.cost_amount

        totals = empty_usage()
        by_role = {role.value: empty_usage() for role in Role}
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

    def prune_artifacts_after(
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
