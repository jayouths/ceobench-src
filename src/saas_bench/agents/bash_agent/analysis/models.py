"""Analysis LLM 输出的严格 Schema。"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisModel(BaseModel):
    """Analysis 产物禁止未定义字段，避免模型静默改变 Schema。"""

    model_config = ConfigDict(extra="forbid")


Confidence = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
NonNegativeDay = Annotated[int, Field(strict=True, ge=0)]
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
    INSUFFICIENT_DATA = "insufficient_data"


class Evidence(AnalysisModel):
    id: str = Field(pattern=r"^[A-Z]{3}-[1-5]$")
    observation: str = Field(min_length=1, max_length=500)
    metric: str = Field(min_length=1, max_length=120)
    direction: Direction
    strength: Confidence
    lag_note: str = Field(min_length=1, max_length=300)


class RoleHypothesis(AnalysisModel):
    cause: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)
    confidence: Confidence
    validation: str = Field(min_length=1, max_length=400)


class RoleRisk(AnalysisModel):
    risk: str = Field(min_length=1, max_length=400)
    early_indicator: str = Field(min_length=1, max_length=300)
    horizon_weeks: HorizonWeeks
    severity: Severity


class RoleAnalysis(AnalysisModel):
    """LLM 只填写分析内容，角色和日期由程序附加。"""

    evidence: list[Evidence] = Field(default_factory=list, max_length=5)
    hypotheses: list[RoleHypothesis] = Field(default_factory=list, max_length=3)
    risks: list[RoleRisk] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_evidence_references(self) -> Self:
        evidence_ids = [item.id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence ids must be unique")
        known = set(evidence_ids)
        for hypothesis in self.hypotheses:
            unknown = sorted(set(hypothesis.evidence_ids) - known)
            if unknown:
                raise ValueError(
                    f"hypothesis references unknown evidence ids: {unknown}"
                )
        return self


class RoleReport(RoleAnalysis):
    role: Role
    day: NonNegativeDay

    _PREFIX_BY_ROLE: ClassVar[dict[Role, str]] = {
        Role.MARKET: "MAR",
        Role.FINANCE: "FIN",
        Role.PRODUCT: "PRO",
        Role.CUSTOMER: "CUS",
    }

    @model_validator(mode="after")
    def validate_role_prefix(self) -> Self:
        expected_prefix = self._PREFIX_BY_ROLE[self.role] + "-"
        invalid = [item.id for item in self.evidence if not item.id.startswith(expected_prefix)]
        if invalid:
            raise ValueError(
                f"{self.role.value} evidence ids must start with {expected_prefix}: {invalid}"
            )
        return self

    @classmethod
    def from_analysis(cls, role: Role, day: int, analysis: RoleAnalysis) -> Self:
        return cls.model_validate({
            "role": role,
            "day": day,
            **analysis.model_dump(),
        })


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


class StateFact(AnalysisModel):
    statement: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    confidence: Confidence


class StateHypothesis(AnalysisModel):
    cause: str = Field(min_length=1, max_length=500)
    evidence_for: list[str] = Field(min_length=1, max_length=8)
    evidence_against: list[str] = Field(default_factory=list, max_length=8)
    competing_causes: list[str] = Field(min_length=1, max_length=4)
    confidence: Confidence
    validation_test: str = Field(min_length=1, max_length=500)


class LatentRisk(AnalysisModel):
    risk: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    early_indicator: str = Field(min_length=1, max_length=300)
    horizon_weeks: HorizonWeeks
    severity: Severity


class CausalStep(AnalysisModel):
    cause: str = Field(min_length=1, max_length=300)
    effect: str = Field(min_length=1, max_length=300)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    confidence: Confidence


class StateAssessment(AnalysisModel):
    """LLM 输出的经营状态，不包含程序生成的标识字段。"""

    diagnosis: str = Field(min_length=1, max_length=500)
    dimensions: list[OperatingDimension] = Field(min_length=5, max_length=5)
    facts: list[StateFact] = Field(default_factory=list, max_length=6)
    hypotheses: list[StateHypothesis] = Field(default_factory=list, max_length=4)
    latent_risks: list[LatentRisk] = Field(default_factory=list, max_length=4)
    causal_chain: list[CausalStep] = Field(default_factory=list, max_length=6)

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
