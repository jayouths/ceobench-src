"""Analysis 确定性经营信号的数据契约。"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saas_bench.public_week_snapshot import DeliveredQuality, PublicWeekSnapshot


class SignalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DataStatus(StrEnum):
    AVAILABLE = "available"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_APPLICABLE = "not_applicable"


class ChangeDirection(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    INSUFFICIENT_DATA = "insufficient_data"


Number = Annotated[int | float, Field(union_mode="left_to_right")]


class NumericObservation(SignalModel):
    value: Number | None
    status: DataStatus

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        if self.status is DataStatus.AVAILABLE:
            if self.value is None or isinstance(self.value, bool):
                raise ValueError("available observations require a numeric value")
            if not math.isfinite(float(self.value)):
                raise ValueError("observation values must be finite")
        elif self.value is not None:
            raise ValueError("unavailable observations must use value=null")
        return self


class MetricComparison(SignalModel):
    current: NumericObservation
    previous: NumericObservation
    absolute_change: Number | None
    relative_change: float | None
    direction: ChangeDirection
    comparison_status: DataStatus

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.comparison_status is DataStatus.AVAILABLE:
            if (
                self.current.status is not DataStatus.AVAILABLE
                or self.previous.status is not DataStatus.AVAILABLE
                or self.absolute_change is None
                or self.direction is ChangeDirection.INSUFFICIENT_DATA
            ):
                raise ValueError("available comparisons require two available values")
        elif self.absolute_change is not None or self.relative_change is not None:
            raise ValueError("unavailable comparisons cannot contain changes")
        return self


class ObservationWindow(SignalModel):
    start_day: int | None = Field(ge=1)
    end_day: int | None = Field(ge=1)
    covered_days: int = Field(ge=0)
    required_days: int = Field(gt=0)
    status: DataStatus


class AnalysisWindows(SignalModel):
    current_7d: ObservationWindow
    previous_7d: ObservationWindow
    recent_28d: ObservationWindow


class GroupLeadSignal(SignalModel):
    group_id: str
    customer_type: Literal["individual", "enterprise"]
    leads: MetricComparison


class EffectiveLeads(SignalModel):
    individual: MetricComparison
    enterprise_accounts: MetricComparison
    total_accounts: MetricComparison
    by_group: list[GroupLeadSignal]


class AcquisitionSourceSignal(SignalModel):
    source: str
    source_type: Literal["organic", "network", "paid"]
    individual: MetricComparison
    enterprise_accounts: MetricComparison
    total_accounts: MetricComparison
    share: MetricComparison


class AcquisitionMix(SignalModel):
    by_source: list[AcquisitionSourceSignal]


class PaidEfficiency(SignalModel):
    spend: MetricComparison
    raw_leads: MetricComparison
    effective_leads: MetricComparison
    raw_cpl: MetricComparison
    effective_cpl: MetricComparison


class ChannelGroupEfficiency(PaidEfficiency):
    channel_id: str
    group_id: str


class PaidAcquisition(SignalModel):
    overall: PaidEfficiency
    by_channel_group: list[ChannelGroupEfficiency]


class SocialPost(SignalModel):
    post_id: int
    day: int
    content: str


class SocialFeedback(SignalModel):
    post_count: MetricComparison
    current_posts: list[SocialPost]
    previous_posts: list[SocialPost]


class MacroCondition(SignalModel):
    status: DataStatus
    observation_day: int
    measurement_day: int | None
    measurement_age_days: int | None
    publication_delay_days: int = 30
    pmi_value: float | None
    pmi_change: float | None
    pmi_trend: str | None
    cycle_phase: str | None
    description: str | None


class MarketSignals(SignalModel):
    effective_leads: EffectiveLeads
    acquisition_mix: AcquisitionMix
    paid_acquisition: PaidAcquisition
    social_feedback: SocialFeedback
    macro_condition: MacroCondition


class RevenueSignals(SignalModel):
    subscription: MetricComparison
    advertising: MetricComparison
    total: MetricComparison


class CostSignals(SignalModel):
    service_delivery: MetricComparison
    acquisition: MetricComparison
    operations: MetricComparison
    development: MetricComparison
    recurring_total: MetricComparison
    one_time_investment: MetricComparison


class RunwaySignals(SignalModel):
    coverage_days: int = Field(ge=0, le=28)
    average_daily_recurring_net_cash_flow: NumericObservation
    average_daily_recurring_burn: NumericObservation
    cash_runway_days: NumericObservation


class FinanceSignals(SignalModel):
    ledger_max_id: int = Field(ge=0)
    current_cash: NumericObservation
    operating_revenue: RevenueSignals
    net_cash_flow: MetricComparison
    costs: CostSignals
    service_delivery_margin: MetricComparison
    runway: RunwaySignals


class UsageSignals(SignalModel):
    total_units: MetricComparison
    daily_average_units: MetricComparison


class CapacitySignals(SignalModel):
    average_utilization: MetricComparison
    peak_utilization: MetricComparison
    peak_overload_excess: MetricComparison
    overload_days: MetricComparison


class ReliabilitySignals(SignalModel):
    average_p95_ms: MetricComparison
    peak_p95_ms: MetricComparison
    average_error_rate: MetricComparison
    peak_error_rate: MetricComparison
    downtime_minutes: MetricComparison
    outage_days: MetricComparison


class ProductConfiguration(SignalModel):
    tier_a: int
    tier_b: int
    tier_c: int
    quota_a: int
    quota_b: int
    quota_c: int
    capacity_tier: int
    daily_operations_spend: float
    daily_development_spend: float


class ConfigurationChange(SignalModel):
    day: int
    field: str
    previous: Number
    current: Number


class ProductConfigurationSignals(SignalModel):
    current: ProductConfiguration
    previous_week: ProductConfiguration | None
    changes: list[ConfigurationChange]
    comparison_status: DataStatus


class ResearchProjectSignal(SignalModel):
    project_id: str
    tier: int
    status: Literal["in_progress", "completed"]
    started_day: int
    expected_completion_day: int
    expected_quality_boost: float
    quality_boost_applied: float


class ResearchPipeline(SignalModel):
    in_progress: list[ResearchProjectSignal]
    completed: list[ResearchProjectSignal]
    completions: MetricComparison


class ProductSignals(SignalModel):
    usage: UsageSignals
    capacity: CapacitySignals
    reliability: ReliabilitySignals
    configuration: ProductConfigurationSignals
    research_pipeline: ResearchPipeline
    delivered_quality: DeliveredQuality


class PlanCustomerBase(SignalModel):
    plan: Literal["A", "B", "C"]
    individual_accounts: int = Field(ge=0)
    enterprise_accounts: int = Field(ge=0)
    enterprise_seats: int = Field(ge=0)


class CustomerBase(SignalModel):
    active_individual_accounts: NumericObservation
    active_enterprise_accounts: NumericObservation
    active_enterprise_seats: NumericObservation
    individual_net_change: MetricComparison
    enterprise_seat_net_change: MetricComparison
    by_plan: list[PlanCustomerBase]


class NewPaidSubscriptions(SignalModel):
    individual_accounts: MetricComparison
    enterprise_accounts: MetricComparison
    enterprise_seats: MetricComparison


class ChurnSignals(SignalModel):
    cancellations: MetricComparison
    upgrades: MetricComparison
    downgrades: MetricComparison
    weekly_account_churn_rate: NumericObservation
    trailing_28d_account_churn_rate: NumericObservation


class IssueSignals(SignalModel):
    open_issues: NumericObservation
    average_open_age_days: NumericObservation
    maximum_open_age_days: NumericObservation
    open_over_7_days: NumericObservation
    open_over_14_days: NumericObservation
    opened: MetricComparison
    resolved: MetricComparison
    average_resolution_days: MetricComparison


class EnterpriseThreadSummary(SignalModel):
    thread_id: int
    customer_id: int
    thread_type: str
    group_id: str
    seat_count: int
    waiting_days: int
    latest_day: int
    latest_sender: str
    latest_message: str


class EnterpriseNegotiations(SignalModel):
    open_threads: int = Field(ge=0)
    open_seats: int = Field(ge=0)
    awaiting_agent_response: int = Field(ge=0)
    average_waiting_days: NumericObservation
    maximum_waiting_days: NumericObservation
    accepted: MetricComparison
    agent_rejected: MetricComparison
    oldest_open_threads: list[EnterpriseThreadSummary]
    details_truncated: bool


class CustomerSignals(SignalModel):
    customer_base: CustomerBase
    new_paid_subscriptions: NewPaidSubscriptions
    churn: ChurnSignals
    issues: IssueSignals
    enterprise_negotiations: EnterpriseNegotiations


class AnalysisSignals(SignalModel):
    schema_version: Literal["1.0"] = "1.0"
    signal_catalog_version: Literal["1.0"] = "1.0"
    day: int = Field(ge=0)
    week: int = Field(ge=0)
    windows: AnalysisWindows
    public_week_snapshot: PublicWeekSnapshot
    market: MarketSignals
    finance: FinanceSignals
    product: ProductSignals
    customer: CustomerSignals
