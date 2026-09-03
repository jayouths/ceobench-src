"""Analysis LLM 输出的严格 Schema。"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisModel(BaseModel):
    """Analysis 产物禁止未定义字段，避免模型静默改变 Schema。"""

    model_config = ConfigDict(extra="forbid")


Confidence = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
NonNegativeDay = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0)]
HorizonWeeks = Annotated[int, Field(strict=True, ge=1, le=26)]
Severity = Annotated[int, Field(strict=True, ge=1, le=5)]


class Role(StrEnum):
    MARKET = "market"
    FINANCE = "finance"
    PRODUCT = "product"
    CUSTOMER = "customer"


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class EvidenceCard(AnalysisModel):
    """程序从确定性经营信号生成的原子事实，LLM 只能选择、不能改写。"""

    id: str = Field(pattern=r"^(MAR|FIN|PRO|CUS)-\d{3}$")
    metric: str = Field(min_length=1, max_length=120)
    meaning: str = Field(min_length=1, max_length=300)
    fact: str = Field(min_length=1, max_length=4000)
    window: str = Field(min_length=1, max_length=100)
    direction: Direction | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class RoleHypothesis(AnalysisModel):
    cause: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)
    confidence: Confidence
    validation: str = Field(min_length=1, max_length=400)


class RoleRisk(AnalysisModel):
    risk: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)
    early_indicator: str = Field(min_length=1, max_length=300)
    horizon_weeks: HorizonWeeks
    severity: Severity


class RoleSelection(AnalysisModel):
    """角色 LLM 只选择事实并生成需要判断力的假设与风险。"""

    selected_evidence_ids: list[str] = Field(min_length=1, max_length=5)
    hypotheses: list[RoleHypothesis] = Field(default_factory=list, max_length=3)
    risks: list[RoleRisk] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_evidence_references(self) -> Self:
        if len(self.selected_evidence_ids) != len(set(self.selected_evidence_ids)):
            raise ValueError("selected evidence ids must be unique")
        return self


class RoleReport(AnalysisModel):
    role: Role
    day: NonNegativeDay
    key_evidence_ids: list[str] = Field(min_length=1, max_length=5)
    # 这里保存所有被选择或引用的事实；输入规模由信号采集上限控制，
    # 不再用任意数量上限拒绝一份引用关系完整的合法报告。
    evidence: list[EvidenceCard] = Field(min_length=1)
    hypotheses: list[RoleHypothesis] = Field(default_factory=list, max_length=3)
    risks: list[RoleRisk] = Field(default_factory=list, max_length=3)

    _PREFIX_BY_ROLE: ClassVar[dict[Role, str]] = {
        Role.MARKET: "MAR",
        Role.FINANCE: "FIN",
        Role.PRODUCT: "PRO",
        Role.CUSTOMER: "CUS",
    }

    @model_validator(mode="after")
    def validate_role_prefix(self) -> Self:
        expected_prefix = self._PREFIX_BY_ROLE[self.role] + "-"
        evidence_ids = [item.id for item in self.evidence]
        invalid = [item_id for item_id in evidence_ids if not item_id.startswith(expected_prefix)]
        if invalid:
            raise ValueError(
                f"{self.role.value} evidence ids must start with {expected_prefix}: {invalid}"
            )
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("role report evidence ids must be unique")
        known = set(evidence_ids)
        unknown_keys = sorted(set(self.key_evidence_ids) - known)
        if unknown_keys:
            raise ValueError(
                f"role report key evidence ids are unknown: {unknown_keys}"
            )
        for hypothesis in self.hypotheses:
            unknown = sorted(set(hypothesis.evidence_ids) - known)
            if unknown:
                raise ValueError(
                    f"hypothesis references unknown evidence ids: {unknown}"
                )
        for risk in self.risks:
            unknown = sorted(set(risk.evidence_ids) - known)
            if unknown:
                raise ValueError(f"risk references unknown evidence ids: {unknown}")
        return self

    @classmethod
    def from_selection(
        cls,
        role: Role,
        day: int,
        selection: RoleSelection,
        cards: list[EvidenceCard],
        state_core_ids: list[str] | None = None,
    ) -> Self:
        cards_by_id = {card.id: card for card in cards}
        referenced_ids = list(selection.selected_evidence_ids)
        referenced_ids.extend(state_core_ids or [])
        for hypothesis in selection.hypotheses:
            referenced_ids.extend(hypothesis.evidence_ids)
        for risk in selection.risks:
            referenced_ids.extend(risk.evidence_ids)
        referenced_ids = list(dict.fromkeys(referenced_ids))
        unknown = sorted(set(referenced_ids) - cards_by_id.keys())
        if unknown:
            raise ValueError(f"role selection references unknown evidence ids: {unknown}")
        return cls.model_validate({
            "role": role,
            "day": day,
            "key_evidence_ids": selection.selected_evidence_ids,
            "evidence": [
                cards_by_id[evidence_id].model_dump(mode="json")
                for evidence_id in referenced_ids
            ],
            "hypotheses": selection.hypotheses,
            "risks": selection.risks,
        })


class AnalysisCallKind(StrEnum):
    INITIAL = "initial"
    REPAIR = "repair"


class RoleCallUsage(AnalysisModel):
    """一次角色 LLM 调用的可复现用量和计价结果。"""

    role: Role
    attempt: PositiveInt
    call_kind: AnalysisCallKind
    requested_model: str = Field(min_length=1)
    served_model: str = Field(min_length=1)
    pricing_model: str = Field(min_length=1)
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cached_tokens: NonNegativeInt
    reasoning_tokens: NonNegativeInt | None
    elapsed_seconds: NonNegativeFloat
    cost_amount: NonNegativeFloat
    currency: str = Field(pattern=r"^[A-Z]{3}$")

    @model_validator(mode="after")
    def validate_cached_tokens(self) -> Self:
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached tokens cannot exceed input tokens")
        return self


class StateCallUsage(AnalysisModel):
    """一次状态重构 LLM 调用的可复现用量和计价结果。"""

    attempt: PositiveInt
    call_kind: AnalysisCallKind
    requested_model: str = Field(min_length=1)
    served_model: str = Field(min_length=1)
    pricing_model: str = Field(min_length=1)
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cached_tokens: NonNegativeInt
    reasoning_tokens: NonNegativeInt | None
    elapsed_seconds: NonNegativeFloat
    cost_amount: NonNegativeFloat
    currency: str = Field(pattern=r"^[A-Z]{3}$")

    @model_validator(mode="after")
    def validate_cached_tokens(self) -> Self:
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached tokens cannot exceed input tokens")
        return self


class RoleReportsArtifact(AnalysisModel):
    """一个模拟周的四角色报告及其全部调用成本。"""

    schema_version: Literal["3.0"] = "3.0"
    day: NonNegativeDay
    reports: list[RoleReport] = Field(min_length=4, max_length=4)
    calls: list[RoleCallUsage] = Field(min_length=4)

    @model_validator(mode="after")
    def validate_complete_week(self) -> Self:
        expected_roles = set(Role)
        report_roles = [report.role for report in self.reports]
        if set(report_roles) != expected_roles or len(set(report_roles)) != 4:
            raise ValueError("reports must contain each role exactly once")
        if any(report.day != self.day for report in self.reports):
            raise ValueError("all role reports must match artifact day")

        for role in Role:
            role_calls = [call for call in self.calls if call.role is role]
            if not role_calls:
                raise ValueError(f"missing {role.value} role call usage")
            attempts = [call.attempt for call in role_calls]
            if attempts != list(range(1, len(role_calls) + 1)):
                raise ValueError(f"{role.value} call attempts must be consecutive")
            if role_calls[0].call_kind is not AnalysisCallKind.INITIAL:
                raise ValueError(f"{role.value} first call must be initial")
            if any(
                call.call_kind is not AnalysisCallKind.REPAIR
                for call in role_calls[1:]
            ):
                raise ValueError(f"{role.value} later calls must be repairs")
        return self


class DimensionName(StrEnum):
    CASH_HEALTH = "cash_health"
    DEMAND_MOMENTUM = "demand_momentum"
    UNIT_ECONOMICS = "unit_economics"
    SERVICE_PRESSURE = "service_pressure"
    CUSTOMER_HEALTH = "customer_health"


DIMENSION_ORDER = tuple(DimensionName)
DIMENSION_LABELS: dict[DimensionName, frozenset[str]] = {
    DimensionName.CASH_HEALTH: frozenset({
        "healthy", "watch", "stressed", "critical", "insufficient_data",
    }),
    DimensionName.DEMAND_MOMENTUM: frozenset({
        "contracting", "stable", "growing", "surging", "insufficient_data",
    }),
    DimensionName.UNIT_ECONOMICS: frozenset({
        "healthy", "marginal", "loss_making", "insufficient_data",
    }),
    DimensionName.SERVICE_PRESSURE: frozenset({
        "underutilized", "balanced", "pressured", "overloaded", "insufficient_data",
    }),
    DimensionName.CUSTOMER_HEALTH: frozenset({
        "healthy", "watch", "deteriorating", "critical", "insufficient_data",
    }),
}


class OperatingDimension(AnalysisModel):
    dimension: DimensionName
    label: str = Field(min_length=1)
    confidence: Confidence
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_dimension_label(self) -> Self:
        allowed = DIMENSION_LABELS[self.dimension]
        if self.label not in allowed:
            raise ValueError(
                f"invalid label for {self.dimension.value}: {self.label!r}; "
                f"allowed={sorted(allowed)}"
            )
        if self.label != "insufficient_data" and not self.evidence_ids:
            raise ValueError("a classified operating dimension must cite evidence")
        return self


class StateHypothesis(AnalysisModel):
    cause: str = Field(min_length=1, max_length=500)
    evidence_for: list[str] = Field(min_length=1, max_length=8)
    evidence_against: list[str] = Field(default_factory=list, max_length=8)
    competing_causes: list[str] = Field(default_factory=list, max_length=4)
    confidence: Confidence
    validation_test: str = Field(min_length=1, max_length=500)


class LatentRisk(AnalysisModel):
    risk: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    early_indicator: str = Field(min_length=1, max_length=300)
    horizon_weeks: HorizonWeeks
    severity: Severity


class StateAssessment(AnalysisModel):
    """LLM 输出的经营状态，不包含程序生成的标识字段。"""

    diagnosis: str = Field(min_length=1, max_length=500)
    dimensions: list[OperatingDimension] = Field(min_length=5, max_length=5)
    # 关键事实直接引用程序生成的证据卡片，避免状态模型再次改写数字。
    key_evidence_ids: list[str] = Field(min_length=1, max_length=3)
    hypotheses: list[StateHypothesis] = Field(default_factory=list, max_length=2)
    latent_risks: list[LatentRisk] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        names = [item.dimension for item in self.dimensions]
        if len(names) != len(set(names)) or set(names) != set(DIMENSION_ORDER):
            raise ValueError("dimensions must contain each fixed dimension exactly once")
        return self


class StatePortrait(StateAssessment):
    """持久化的经营画像仅由程序补充实验日期。"""

    day: NonNegativeDay

    @classmethod
    def from_assessment(cls, day: int, assessment: StateAssessment) -> Self:
        return cls.model_validate({
            "day": day,
            **assessment.model_dump(),
        })


class StatePortraitArtifact(StatePortrait):
    """一个模拟周的经营画像及其全部状态重构调用成本。"""

    schema_version: Literal["2.0"] = "2.0"
    calls: list[StateCallUsage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_calls(self) -> Self:
        attempts = [call.attempt for call in self.calls]
        if attempts != list(range(1, len(self.calls) + 1)):
            raise ValueError("state reconstruction attempts must be consecutive")
        if self.calls[0].call_kind is not AnalysisCallKind.INITIAL:
            raise ValueError("state reconstruction first call must be initial")
        if any(
            call.call_kind is not AnalysisCallKind.REPAIR
            for call in self.calls[1:]
        ):
            raise ValueError("state reconstruction later calls must be repairs")
        return self

    @classmethod
    def from_assessment(
        cls,
        day: int,
        assessment: StateAssessment,
        calls: list[StateCallUsage],
    ) -> Self:
        return cls.model_validate({
            "day": day,
            **assessment.model_dump(),
            "calls": [call.model_dump() for call in calls],
        })
