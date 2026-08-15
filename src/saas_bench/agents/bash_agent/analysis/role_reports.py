"""调用四个职能角色并生成严格校验的周度报告。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from .models import (
    Direction,
    Role,
    RoleAnalysis,
    AnalysisCallKind,
    RoleCallUsage,
    RoleReport,
    RoleReportsArtifact,
)
from .role_prompts import build_repair_prompt, build_role_prompts
from .signal_models import AnalysisSignals


@dataclass(frozen=True)
class RoleCallOutcome:
    text: str
    usage: RoleCallUsage


RoleModelCall = Callable[
    [int, Role, int, AnalysisCallKind, str, str],
    RoleCallOutcome,
]


class RoleReportGenerationError(RuntimeError):
    pass


class RoleReportGenerator:
    """按固定角色顺序调用模型，失败时执行有限次数的自包含修复。"""

    def __init__(self, call_model: RoleModelCall, *, max_schema_retries: int):
        if max_schema_retries < 0:
            raise ValueError("max_schema_retries must be non-negative")
        self.call_model = call_model
        self.max_schema_retries = max_schema_retries

    def generate(self, signals: AnalysisSignals) -> RoleReportsArtifact:
        reports: list[RoleReport] = []
        calls: list[RoleCallUsage] = []
        for role in Role:
            report, role_calls = self._generate_role(signals, role)
            reports.append(report)
            calls.extend(role_calls)
        return RoleReportsArtifact(day=signals.day, reports=reports, calls=calls)

    def _generate_role(
        self,
        signals: AnalysisSignals,
        role: Role,
    ) -> tuple[RoleReport, list[RoleCallUsage]]:
        system_prompt, user_prompt = build_role_prompts(signals, role)
        role_calls: list[RoleCallUsage] = []
        last_error = ""
        last_text = ""

        for attempt in range(1, self.max_schema_retries + 2):
            call_kind = (
                AnalysisCallKind.INITIAL if attempt == 1 else AnalysisCallKind.REPAIR
            )
            if call_kind is AnalysisCallKind.REPAIR:
                system_prompt, user_prompt = build_repair_prompt(
                    signals,
                    role,
                    last_text,
                    last_error,
                )
            outcome = self.call_model(
                signals.day,
                role,
                attempt,
                call_kind,
                system_prompt,
                user_prompt,
            )
            if (
                outcome.usage.role is not role
                or outcome.usage.attempt != attempt
                or outcome.usage.call_kind is not call_kind
            ):
                raise ValueError("role call usage identity does not match invocation")
            role_calls.append(outcome.usage)
            last_text = outcome.text

            try:
                payload = json.loads(last_text)
                if not isinstance(payload, dict):
                    raise ValueError("top-level response must be a JSON object")
                analysis = RoleAnalysis.model_validate(payload)
                report = RoleReport.from_analysis(role, signals.day, analysis)
                self._validate_evidence_metrics(signals, report)
                return (
                    report,
                    role_calls,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)

        raise RoleReportGenerationError(
            f"{role.value} role report remained invalid after "
            f"{len(role_calls)} call(s): {last_error}"
        )

    @staticmethod
    def _validate_evidence_metrics(
        signals: AnalysisSignals,
        report: RoleReport,
    ) -> None:
        """证据必须引用本角色真实字段，环比方向必须与程序计算一致。"""

        role_payload = getattr(signals, report.role.value).model_dump(mode="json")
        prefix = report.role.value + "."
        errors: list[str] = []
        for evidence in report.evidence:
            if not evidence.metric.startswith(prefix):
                errors.append(
                    f"metric path must start with {prefix!r}: {evidence.metric!r}"
                )
                continue
            target = role_payload
            for part in evidence.metric.removeprefix(prefix).split("."):
                if not isinstance(target, dict) or part not in target:
                    errors.append(f"unknown metric path: {evidence.metric!r}")
                    target = None
                    break
                target = target[part]

            if target is None:
                continue

            expected_direction = None
            if isinstance(target, dict) and "direction" in target:
                expected_direction = target["direction"]
            elif (
                isinstance(target, dict)
                and target.get("status") in {"insufficient_data", "not_applicable"}
            ):
                expected_direction = Direction.INSUFFICIENT_DATA.value
            if (
                expected_direction is not None
                and evidence.direction.value != expected_direction
            ):
                errors.append(
                    f"metric direction mismatch for {evidence.metric!r}: "
                    f"expected {expected_direction!r}, got {evidence.direction.value!r}"
                )
        if errors:
            raise ValueError("; ".join(errors))
