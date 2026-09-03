"""合并四角色报告并生成统一经营状态画像。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from json import JSONDecodeError

from pydantic import ValidationError

from .models import (
    AnalysisCallKind,
    RoleReportsArtifact,
    StateAssessment,
    StateCallUsage,
    StatePortraitArtifact,
)
from .state_prompts import build_state_prompts, build_state_repair_prompt
from .json_response import parse_json_object


@dataclass(frozen=True)
class StateCallOutcome:
    text: str
    usage: StateCallUsage


StateModelCall = Callable[
    [int, int, AnalysisCallKind, str, str],
    StateCallOutcome,
]


class StateReconstructionError(RuntimeError):
    pass


class StateReconstructor:
    """调用状态重构模型，并对结构和证据引用执行有限次修复。"""

    def __init__(self, call_model: StateModelCall, *, max_schema_retries: int):
        if max_schema_retries < 0:
            raise ValueError("max_schema_retries must be non-negative")
        self.call_model = call_model
        self.max_schema_retries = max_schema_retries

    def generate(
        self,
        role_reports: RoleReportsArtifact,
    ) -> StatePortraitArtifact:
        system_prompt, user_prompt = build_state_prompts(role_reports)
        calls: list[StateCallUsage] = []
        last_error = ""
        last_text = ""

        for attempt in range(1, self.max_schema_retries + 2):
            call_kind = (
                AnalysisCallKind.INITIAL if attempt == 1 else AnalysisCallKind.REPAIR
            )
            if call_kind is AnalysisCallKind.REPAIR:
                system_prompt, user_prompt = build_state_repair_prompt(
                    role_reports,
                    last_text,
                    last_error,
                )
            outcome = self.call_model(
                role_reports.day,
                attempt,
                call_kind,
                system_prompt,
                user_prompt,
            )
            if (
                outcome.usage.attempt != attempt
                or outcome.usage.call_kind is not call_kind
            ):
                raise ValueError("state call usage identity does not match invocation")
            calls.append(outcome.usage)
            last_text = outcome.text

            try:
                payload = parse_json_object(last_text)
                assessment = StateAssessment.model_validate(payload)
                self._validate_evidence_references(role_reports, assessment)
                return StatePortraitArtifact.from_assessment(
                    role_reports.day,
                    assessment,
                    calls,
                )
            except (JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)

        raise StateReconstructionError(
            "state reconstruction remained invalid after "
            f"{len(calls)} call(s): {last_error}"
        )

    @staticmethod
    def _validate_evidence_references(
        role_reports: RoleReportsArtifact,
        assessment: StateAssessment,
    ) -> None:
        """画像只能引用四份角色报告中真实存在的证据编号。"""

        known_ids = {
            evidence.id
            for report in role_reports.reports
            for evidence in report.evidence
        }
        references: list[str] = []
        for dimension in assessment.dimensions:
            references.extend(dimension.evidence_ids)
        references.extend(assessment.key_evidence_ids)
        for hypothesis in assessment.hypotheses:
            references.extend(hypothesis.evidence_for)
            references.extend(hypothesis.evidence_against)
        for risk in assessment.latent_risks:
            references.extend(risk.evidence_ids)

        unknown = sorted(set(references) - known_ids)
        if unknown:
            raise ValueError(f"state portrait references unknown evidence ids: {unknown}")
