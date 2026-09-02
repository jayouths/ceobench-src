"""调用四个职能角色并生成严格校验的周度报告。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from json import JSONDecodeError

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
from .json_response import parse_json_object
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
    """并行调用独立角色，失败时执行有限次数的自包含修复。"""

    def __init__(
        self,
        call_model: RoleModelCall,
        *,
        max_schema_retries: int,
        role_report_concurrency: int = 1,
    ):
        if max_schema_retries < 0:
            raise ValueError("max_schema_retries must be non-negative")
        if not 1 <= role_report_concurrency <= len(Role):
            raise ValueError("role_report_concurrency must be between 1 and 4")
        self.call_model = call_model
        self.max_schema_retries = max_schema_retries
        self.role_report_concurrency = role_report_concurrency

    def generate(self, signals: AnalysisSignals) -> RoleReportsArtifact:
        reports: list[RoleReport] = []
        calls: list[RoleCallUsage] = []
        roles = list(Role)
        if self.role_report_concurrency == 1:
            outcomes = [self._generate_role(signals, role) for role in roles]
        else:
            # 四个角色只读取同一份不可变信号，彼此没有数据依赖；map 在并行
            # 执行的同时保持角色枚举顺序，使落盘产物和串行版本一致。
            with ThreadPoolExecutor(
                max_workers=self.role_report_concurrency,
                thread_name_prefix="analysis-role",
            ) as executor:
                outcomes = list(executor.map(
                    lambda role: self._generate_role(signals, role), roles
                ))
        for report, role_calls in outcomes:
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
                payload = parse_json_object(last_text)
                analysis = RoleAnalysis.model_validate(payload)
                report = RoleReport.from_analysis(role, signals.day, analysis)
                self._validate_evidence_metrics(signals, report)
                return (
                    report,
                    role_calls,
                )
            except (JSONDecodeError, ValidationError, ValueError) as exc:
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

            # 只有确定性信号层已经计算出 direction 的环比对象，
            # 才允许角色报告声称 up/down/flat。单点值、文本和配置值
            # 都不能自行推断趋势，否则会把当前状态误写成变化方向。
            expected_direction = Direction.INSUFFICIENT_DATA.value
            if isinstance(target, dict) and "direction" in target:
                expected_direction = target["direction"]
            if evidence.direction.value != expected_direction:
                errors.append(
                    f"metric direction mismatch for {evidence.metric!r}: "
                    f"expected {expected_direction!r}, got {evidence.direction.value!r}"
                )
        if errors:
            raise ValueError("; ".join(errors))
