"""通过公开查询生成 Analysis 的确定性经营信号。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import TypeAdapter

from saas_bench.public_week_snapshot import PublicWeekSnapshot

from . import signal_queries
from .signal_models import (
    AcquisitionMix,
    AcquisitionSourceSignal,
    AnalysisSignals,
    AnalysisWindows,
    CapacitySignals,
    ChangeDirection,
    ChannelGroupEfficiency,
    ChurnSignals,
    ConfigurationChange,
    CostSignals,
    CustomerBase,
    CustomerSignals,
    DataStatus,
    EffectiveLeads,
    EnterpriseNegotiations,
    EnterpriseThreadSummary,
    FinanceSignals,
    GroupLeadSignal,
    IssueSignals,
    MacroCondition,
    MarketSignals,
    MetricComparison,
    NewPaidSubscriptions,
    NumericObservation,
    ObservationWindow,
    PaidAcquisition,
    PaidEfficiency,
    PlanCustomerBase,
    ProductConfiguration,
    ProductConfigurationSignals,
    ProductSignals,
    ReliabilitySignals,
    ResearchPipeline,
    ResearchProjectSignal,
    RevenueSignals,
    RunwaySignals,
    SocialFeedback,
    SocialPost,
    UsageSignals,
)


PublicQuery = Callable[[str], list[dict[str, Any]]]
_SNAPSHOT_ADAPTER = TypeAdapter(PublicWeekSnapshot)


def parse_public_week_snapshot(payload: Any) -> PublicWeekSnapshot:
    """校验 API 返回的快照，禁止结构变化静默进入实验。"""

    return _SNAPSHOT_ADAPTER.validate_python(payload)


def _window(day: int, required_days: int, offset_days: int = 0) -> ObservationWindow:
    end = day - offset_days
    if end < 1:
        return ObservationWindow(
            start_day=None,
            end_day=None,
            covered_days=0,
            required_days=required_days,
            status=DataStatus.INSUFFICIENT_DATA,
        )
    start = max(1, end - required_days + 1)
    covered = end - start + 1
    return ObservationWindow(
        start_day=start,
        end_day=end,
        covered_days=covered,
        required_days=required_days,
        status=(
            DataStatus.AVAILABLE
            if covered == required_days
            else DataStatus.INSUFFICIENT_DATA
        ),
    )


def build_analysis_windows(day: int) -> AnalysisWindows:
    return AnalysisWindows(
        current_7d=_window(day, 7),
        previous_7d=_window(day, 7, offset_days=7),
        recent_28d=_window(day, 28),
    )


def _observation(
    value: int | float | None,
    status: DataStatus = DataStatus.AVAILABLE,
) -> NumericObservation:
    return NumericObservation(value=value if status is DataStatus.AVAILABLE else None, status=status)


def _comparison_from_observations(
    current: NumericObservation,
    previous: NumericObservation,
) -> MetricComparison:
    if (
        current.status is not DataStatus.AVAILABLE
        or previous.status is not DataStatus.AVAILABLE
    ):
        status = (
            DataStatus.NOT_APPLICABLE
            if DataStatus.NOT_APPLICABLE in {current.status, previous.status}
            else DataStatus.INSUFFICIENT_DATA
        )
        return MetricComparison(
            current=current,
            previous=previous,
            absolute_change=None,
            relative_change=None,
            direction=ChangeDirection.INSUFFICIENT_DATA,
            comparison_status=status,
        )

    current_value = current.value
    previous_value = previous.value
    assert current_value is not None and previous_value is not None
    change = current_value - previous_value
    if change > 0:
        direction = ChangeDirection.UP
    elif change < 0:
        direction = ChangeDirection.DOWN
    else:
        direction = ChangeDirection.FLAT
    relative_change = None if previous_value == 0 else float(change / previous_value)
    return MetricComparison(
        current=current,
        previous=previous,
        absolute_change=change,
        relative_change=relative_change,
        direction=direction,
        comparison_status=DataStatus.AVAILABLE,
    )


def _comparison(
    current: int | float | None,
    previous: int | float | None,
    current_status: DataStatus,
    previous_status: DataStatus,
) -> MetricComparison:
    return _comparison_from_observations(
        _observation(current, current_status),
        _observation(previous, previous_status),
    )


def _window_status(window: ObservationWindow, row_count: int | None = None) -> DataStatus:
    if window.status is not DataStatus.AVAILABLE:
        return DataStatus.INSUFFICIENT_DATA
    if row_count is not None and row_count != window.required_days:
        return DataStatus.INSUFFICIENT_DATA
    return DataStatus.AVAILABLE


def _in_window(day: int, window: ObservationWindow) -> bool:
    return (
        window.start_day is not None
        and window.end_day is not None
        and window.start_day <= day <= window.end_day
    )


def _sum(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(row.get(key) or 0) for row in rows)


class SignalCollector:
    """只使用 Baseline 已公开的快照和 `/query` 数据生成信号。"""

    def __init__(self, query: PublicQuery, *, max_enterprise_threads: int = 50):
        if max_enterprise_threads <= 0:
            raise ValueError("max_enterprise_threads must be positive")
        self.query = query
        self.max_enterprise_threads = max_enterprise_threads

    def collect(
        self,
        snapshot: PublicWeekSnapshot,
        history: Mapping[int, AnalysisSignals] | None = None,
    ) -> AnalysisSignals:
        history = history or {}
        day = snapshot.day
        windows = build_analysis_windows(day)
        start_14d = max(1, day - 13)

        # SQL 集中定义在 signal_queries；这里一次读取四个角色需要的公开事实，
        # 后续构建阶段只做确定性计算，不再访问数据库。
        lead_rows = (
            self.query(signal_queries.effective_leads(start_14d, day))
            if day else []
        )
        ad_rows = self.query(signal_queries.ad_channels(start_14d, day)) if day else []
        social_rows = (
            self.query(signal_queries.social_posts(start_14d, day))
            if day else []
        )
        macro_rows = self.query(signal_queries.LATEST_MACRO_CONDITION)
        ledger_max_rows = self.query(signal_queries.LEDGER_MAX_ID)
        ledger_max_id = int(ledger_max_rows[0]["ledger_max_id"])
        previous_artifact = history.get(day - 7)
        previous_ledger_max_id = (
            previous_artifact.finance.ledger_max_id if previous_artifact else None
        )
        ledger_rows = (
            self.query(signal_queries.ledger_after(previous_ledger_max_id))
            if previous_ledger_max_id is not None else []
        )
        service_rows = (
            self.query(signal_queries.service_days(start_14d, day))
            if day else []
        )
        config_rows = self.query(signal_queries.configuration_history(day))
        research_rows = self.query(signal_queries.RESEARCH_PROJECTS)
        customer_base_rows = self.query(signal_queries.CUSTOMER_BASE)
        paid_subscription_rows = (
            self.query(signal_queries.paid_subscriptions(start_14d, day)) if day else []
        )
        issue_rows = self._query_issues(windows)
        negotiation_summary = self._query_negotiation_summary(day)
        negotiation_details = self._query_negotiation_details(day)
        negotiation_outcomes = (
            self.query(signal_queries.negotiation_outcomes(start_14d, day))
            if day else []
        )

        market = self._build_market(
            snapshot, windows, history, lead_rows, ad_rows, social_rows, macro_rows
        )
        finance = self._build_finance(
            snapshot, windows, history, ledger_max_id, ledger_rows
        )
        product = self._build_product(
            snapshot, windows, history, service_rows, config_rows, research_rows
        )
        customer = self._build_customer(
            snapshot,
            windows,
            history,
            customer_base_rows,
            paid_subscription_rows,
            issue_rows,
            negotiation_summary,
            negotiation_details,
            negotiation_outcomes,
        )
        return AnalysisSignals(
            day=day,
            week=snapshot.week,
            windows=windows,
            public_week_snapshot=snapshot,
            market=market,
            finance=finance,
            product=product,
            customer=customer,
        )

    def _query_issues(self, windows: AnalysisWindows) -> dict[str, Any]:
        current = windows.current_7d
        previous = windows.previous_7d
        cs, ce = current.start_day or 1, current.end_day or 0
        ps, pe = previous.start_day or 1, previous.end_day or 0
        rows = self.query(signal_queries.issue_summary(cs, ce, ps, pe))
        return rows[0] if rows else {}

    def _query_negotiation_summary(self, day: int) -> dict[str, Any]:
        rows = self.query(signal_queries.negotiation_summary(day))
        return rows[0] if rows else {}

    def _query_negotiation_details(self, day: int) -> list[dict[str, Any]]:
        limit = self.max_enterprise_threads + 1
        return self.query(signal_queries.negotiation_details(day, limit))

    def _period_rows(
        self, rows: Sequence[Mapping[str, Any]], window: ObservationWindow
    ) -> list[Mapping[str, Any]]:
        return [row for row in rows if _in_window(int(row["day"]), window)]

    def _build_market(
        self,
        snapshot: PublicWeekSnapshot,
        windows: AnalysisWindows,
        history: Mapping[int, AnalysisSignals],
        lead_rows: list[dict[str, Any]],
        ad_rows: list[dict[str, Any]],
        social_rows: list[dict[str, Any]],
        macro_rows: list[dict[str, Any]],
    ) -> MarketSignals:
        current_status = windows.current_7d.status
        previous_status = windows.previous_7d.status
        activity = snapshot.weekly_activity
        previous_artifact = history.get(snapshot.day - 7)
        previous_activity = (
            previous_artifact.public_week_snapshot.weekly_activity
            if previous_artifact else None
        )

        def activity_count(name: str, previous: bool = False) -> int | None:
            source = previous_activity if previous else activity
            return getattr(source, name) if source is not None else None

        individual = _comparison(
            activity_count("new_individual_leads"),
            activity_count("new_individual_leads", True),
            current_status if activity else DataStatus.INSUFFICIENT_DATA,
            previous_status if previous_activity else DataStatus.INSUFFICIENT_DATA,
        )
        enterprise = _comparison(
            activity_count("new_enterprise_leads"),
            activity_count("new_enterprise_leads", True),
            current_status if activity else DataStatus.INSUFFICIENT_DATA,
            previous_status if previous_activity else DataStatus.INSUFFICIENT_DATA,
        )
        current_total = (
            activity.new_individual_leads + activity.new_enterprise_leads
            if activity else None
        )
        previous_total = (
            previous_activity.new_individual_leads + previous_activity.new_enterprise_leads
            if previous_activity else None
        )
        total = _comparison(
            current_total,
            previous_total,
            current_status if activity else DataStatus.INSUFFICIENT_DATA,
            previous_status if previous_activity else DataStatus.INSUFFICIENT_DATA,
        )

        grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"current": 0, "previous": 0}
        )
        sourced: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"current_individual": 0, "current_enterprise": 0,
                     "previous_individual": 0, "previous_enterprise": 0}
        )
        for row in lead_rows:
            period = "current" if _in_window(int(row["day"]), windows.current_7d) else "previous"
            customer_type = str(row["customer_type"])
            count = int(row["lead_count"])
            grouped[(str(row["group_id"]), customer_type)][period] += count
            sourced[(str(row["acquisition_source"]), customer_type)][
                f"{period}_{customer_type}"
            ] += count

        by_group = []
        for (group_id, customer_type), values in sorted(grouped.items()):
            by_group.append(GroupLeadSignal(
                group_id=group_id,
                customer_type=customer_type,
                leads=_comparison(
                    values["current"], values["previous"], current_status, previous_status
                ),
            ))

        source_names = sorted({source for source, _ in sourced})
        current_denominator = current_total or 0
        previous_denominator = previous_total or 0
        by_source = []
        for source in source_names:
            values = {key: 0 for key in (
                "current_individual", "current_enterprise",
                "previous_individual", "previous_enterprise",
            )}
            for customer_type in ("individual", "enterprise"):
                for key, value in sourced.get((source, customer_type), {}).items():
                    values[key] += value
            ci, ce = values["current_individual"], values["current_enterprise"]
            pi, pe = values["previous_individual"], values["previous_enterprise"]
            current_share_status = (
                DataStatus.INSUFFICIENT_DATA
                if current_status is not DataStatus.AVAILABLE
                else DataStatus.AVAILABLE
                if current_denominator > 0
                else DataStatus.NOT_APPLICABLE
            )
            previous_share_status = (
                DataStatus.INSUFFICIENT_DATA
                if previous_status is not DataStatus.AVAILABLE
                else DataStatus.AVAILABLE
                if previous_denominator > 0
                else DataStatus.NOT_APPLICABLE
            )
            by_source.append(AcquisitionSourceSignal(
                source=source,
                source_type=(
                    "organic" if source in {"organic", "word_of_mouth"}
                    else "network" if source == "network"
                    else "paid"
                ),
                individual=_comparison(ci, pi, current_status, previous_status),
                enterprise_accounts=_comparison(ce, pe, current_status, previous_status),
                total_accounts=_comparison(ci + ce, pi + pe, current_status, previous_status),
                share=_comparison(
                    (ci + ce) / current_denominator if current_denominator else None,
                    (pi + pe) / previous_denominator if previous_denominator else None,
                    current_share_status,
                    previous_share_status,
                ),
            ))

        effective_by_channel_group: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"current": 0, "previous": 0}
        )
        for row in lead_rows:
            source = str(row["acquisition_source"])
            if source in {"organic", "network"}:
                continue
            period = "current" if _in_window(int(row["day"]), windows.current_7d) else "previous"
            effective_by_channel_group[(source, str(row["group_id"]))][period] += int(row["lead_count"])

        raw_by_channel_group: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"current_spend": 0.0, "current_raw": 0.0,
                     "previous_spend": 0.0, "previous_raw": 0.0}
        )
        for row in ad_rows:
            period = "current" if _in_window(int(row["day"]), windows.current_7d) else "previous"
            values = raw_by_channel_group[(str(row["channel_id"]), str(row["group_id"]))]
            values[f"{period}_spend"] += float(row["spend"] or 0)
            values[f"{period}_raw"] += float(row["raw_leads"] or 0)

        channel_keys = sorted(set(raw_by_channel_group) | set(effective_by_channel_group))

        def efficiency(
            values: Mapping[str, float], effective: Mapping[str, int]
        ) -> PaidEfficiency:
            cs, ps = values.get("current_spend", 0.0), values.get("previous_spend", 0.0)
            cr, pr = values.get("current_raw", 0.0), values.get("previous_raw", 0.0)
            ce, pe = effective.get("current", 0), effective.get("previous", 0)
            def cpl_status(window_status: DataStatus, leads: float) -> DataStatus:
                if window_status is not DataStatus.AVAILABLE:
                    return DataStatus.INSUFFICIENT_DATA
                return DataStatus.AVAILABLE if leads > 0 else DataStatus.NOT_APPLICABLE

            current_raw_cpl_status = cpl_status(current_status, cr)
            previous_raw_cpl_status = cpl_status(previous_status, pr)
            current_effective_cpl_status = cpl_status(current_status, ce)
            previous_effective_cpl_status = cpl_status(previous_status, pe)
            return PaidEfficiency(
                spend=_comparison(cs, ps, current_status, previous_status),
                raw_leads=_comparison(cr, pr, current_status, previous_status),
                effective_leads=_comparison(ce, pe, current_status, previous_status),
                raw_cpl=_comparison(
                    cs / cr if cr else None,
                    ps / pr if pr else None,
                    current_raw_cpl_status,
                    previous_raw_cpl_status,
                ),
                effective_cpl=_comparison(
                    cs / ce if ce else None,
                    ps / pe if pe else None,
                    current_effective_cpl_status,
                    previous_effective_cpl_status,
                ),
            )

        by_channel_group = []
        overall_values = defaultdict(float)
        overall_effective = defaultdict(int)
        for channel_id, group_id in channel_keys:
            values = raw_by_channel_group[(channel_id, group_id)]
            effective = effective_by_channel_group[(channel_id, group_id)]
            for key, value in values.items():
                overall_values[key] += value
            for key, value in effective.items():
                overall_effective[key] += value
            item = efficiency(values, effective)
            by_channel_group.append(ChannelGroupEfficiency(
                channel_id=channel_id,
                group_id=group_id,
                **item.model_dump(),
            ))

        current_posts = [SocialPost(**row) for row in social_rows if _in_window(int(row["day"]), windows.current_7d)]
        previous_posts = [SocialPost(**row) for row in social_rows if _in_window(int(row["day"]), windows.previous_7d)]
        macro = macro_rows[0] if macro_rows else None
        macro_condition = MacroCondition(
            status=DataStatus.AVAILABLE if macro else DataStatus.INSUFFICIENT_DATA,
            observation_day=snapshot.day,
            measurement_day=int(macro["day"]) if macro else None,
            measurement_age_days=snapshot.day - int(macro["day"]) if macro else None,
            pmi_value=float(macro["pmi_value"]) if macro else None,
            pmi_change=float(macro["pmi_change"]) if macro else None,
            pmi_trend=str(macro["pmi_trend"]) if macro else None,
            cycle_phase=str(macro["cycle_phase"]) if macro else None,
            description=str(macro["description"]) if macro else None,
        )
        return MarketSignals(
            effective_leads=EffectiveLeads(
                individual=individual,
                enterprise_accounts=enterprise,
                total_accounts=total,
                by_group=by_group,
            ),
            acquisition_mix=AcquisitionMix(by_source=by_source),
            paid_acquisition=PaidAcquisition(
                overall=efficiency(overall_values, overall_effective),
                by_channel_group=by_channel_group,
            ),
            social_feedback=SocialFeedback(
                post_count=_comparison(
                    len(current_posts), len(previous_posts), current_status, previous_status
                ),
                current_posts=current_posts,
                previous_posts=previous_posts,
            ),
            macro_condition=macro_condition,
        )

    def _build_finance(
        self,
        snapshot: PublicWeekSnapshot,
        windows: AnalysisWindows,
        history: Mapping[int, AnalysisSignals],
        ledger_max_id: int,
        ledger_rows: list[dict[str, Any]],
    ) -> FinanceSignals:
        previous_artifact = history.get(snapshot.day - 7)
        current_status = (
            windows.current_7d.status
            if previous_artifact is not None
            else DataStatus.INSUFFICIENT_DATA
        )
        current = defaultdict(float)
        for row in ledger_rows:
            current[str(row["category"])] += float(row["amount"] or 0)

        def previous_observation(path: str) -> NumericObservation:
            if previous_artifact is None:
                return _observation(None, DataStatus.INSUFFICIENT_DATA)
            target: Any = previous_artifact.finance
            for part in path.split("."):
                target = getattr(target, part)
            return target.current

        def current_comparison(key: str, previous_path: str) -> MetricComparison:
            return _comparison_from_observations(
                _observation(current[key], current_status),
                previous_observation(previous_path),
            )

        current_revenue = current["subscription_payment"] + current["ad_revenue"]

        def cost(values: Mapping[str, float], categories: set[str]) -> float:
            result = -sum(values[category] for category in categories)
            return 0.0 if result == 0 else result

        service_categories = {"compute", "capacity"}
        acquisition_categories = {"advertising", "lead_acquisition_cost"}
        recurring_categories = service_categories | acquisition_categories | {"operations", "development"}
        one_time_categories = {"market_research", "group_research", "research_project"}

        def cost_pair(categories: set[str], previous_path: str) -> MetricComparison:
            return _comparison_from_observations(
                _observation(cost(current, categories), current_status),
                previous_observation(previous_path),
            )

        current_service_cost = cost(current, service_categories)
        current_margin_status = (
            DataStatus.INSUFFICIENT_DATA
            if current_status is not DataStatus.AVAILABLE
            else DataStatus.AVAILABLE
            if current_revenue > 0
            else DataStatus.NOT_APPLICABLE
        )

        weekly_recurring_net = current_revenue - cost(current, recurring_categories)
        trailing_weekly_net = [weekly_recurring_net] if current_status is DataStatus.AVAILABLE else []
        for offset in (7, 14, 21):
            artifact = history.get(snapshot.day - offset)
            if artifact is None:
                continue
            revenue = artifact.finance.operating_revenue.total.current
            recurring_cost = artifact.finance.costs.recurring_total.current
            if (
                revenue.status is DataStatus.AVAILABLE
                and recurring_cost.status is DataStatus.AVAILABLE
            ):
                trailing_weekly_net.append(
                    float(revenue.value) - float(recurring_cost.value)
                )
        coverage_days = len(trailing_weekly_net) * 7
        runway_status = (
            DataStatus.AVAILABLE
            if coverage_days == 28
            else DataStatus.INSUFFICIENT_DATA
        )
        if runway_status is DataStatus.AVAILABLE:
            recurring_net = sum(trailing_weekly_net) / 28
            burn = max(0.0, -recurring_net)
            runway = snapshot.current_state.cash / burn if burn > 0 else None
            burn_status = DataStatus.AVAILABLE
            runway_value_status = (
                DataStatus.AVAILABLE if burn > 0 else DataStatus.NOT_APPLICABLE
            )
        else:
            recurring_net = burn = runway = None
            burn_status = runway_value_status = DataStatus.INSUFFICIENT_DATA

        return FinanceSignals(
            ledger_max_id=ledger_max_id,
            current_cash=_observation(snapshot.current_state.cash),
            operating_revenue=RevenueSignals(
                subscription=current_comparison(
                    "subscription_payment", "operating_revenue.subscription"
                ),
                advertising=current_comparison(
                    "ad_revenue", "operating_revenue.advertising"
                ),
                total=_comparison_from_observations(
                    _observation(current_revenue, current_status),
                    previous_observation("operating_revenue.total"),
                ),
            ),
            net_cash_flow=_comparison_from_observations(
                _observation(
                    snapshot.current_state.cash
                    - float(previous_artifact.finance.current_cash.value)
                    if previous_artifact else None,
                    current_status,
                ),
                previous_observation("net_cash_flow"),
            ),
            costs=CostSignals(
                service_delivery=cost_pair(service_categories, "costs.service_delivery"),
                acquisition=cost_pair(acquisition_categories, "costs.acquisition"),
                operations=cost_pair({"operations"}, "costs.operations"),
                development=cost_pair({"development"}, "costs.development"),
                recurring_total=cost_pair(recurring_categories, "costs.recurring_total"),
                one_time_investment=cost_pair(one_time_categories, "costs.one_time_investment"),
            ),
            service_delivery_margin=_comparison_from_observations(
                _observation(
                    (current_revenue - current_service_cost) / current_revenue
                    if current_revenue else None,
                    current_margin_status,
                ),
                previous_observation("service_delivery_margin"),
            ),
            runway=RunwaySignals(
                coverage_days=coverage_days,
                average_daily_recurring_net_cash_flow=_observation(recurring_net, runway_status),
                average_daily_recurring_burn=_observation(burn, burn_status),
                cash_runway_days=_observation(runway, runway_value_status),
            ),
        )

    def _build_product(
        self,
        snapshot: PublicWeekSnapshot,
        windows: AnalysisWindows,
        history: Mapping[int, AnalysisSignals],
        service_rows: list[dict[str, Any]],
        config_rows: list[dict[str, Any]],
        research_rows: list[dict[str, Any]],
    ) -> ProductSignals:
        current_rows = self._period_rows(service_rows, windows.current_7d)
        previous_rows = self._period_rows(service_rows, windows.previous_7d)
        cs = _window_status(windows.current_7d, len(current_rows))
        ps = _window_status(windows.previous_7d, len(previous_rows))

        def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
            if not rows:
                return defaultdict(float)
            utilizations = [
                float(row["total_usage_units"]) / float(row["capacity_units"])
                for row in rows if float(row["capacity_units"]) > 0
            ]
            return {
                "usage_total": _sum(rows, "total_usage_units"),
                "usage_average": _sum(rows, "total_usage_units") / len(rows),
                "utilization_average": sum(utilizations) / len(utilizations) if utilizations else 0,
                "utilization_peak": max(utilizations, default=0),
                "overload_peak_excess": max(
                    0, max((value - 1 for value in utilizations), default=0)
                ),
                "overload_days": sum(value > 1 for value in utilizations),
                "p95_average": _sum(rows, "p95_ms") / len(rows),
                "p95_peak": max(float(row["p95_ms"]) for row in rows),
                "error_average": _sum(rows, "error_rate") / len(rows),
                "error_peak": max(float(row["error_rate"]) for row in rows),
                "downtime": _sum(rows, "downtime_minutes"),
                "outage_days": sum(float(row["downtime_minutes"]) > 0 for row in rows),
            }

        current = aggregate(current_rows)
        previous = aggregate(previous_rows)

        def pair(key: str) -> MetricComparison:
            return _comparison(current.get(key), previous.get(key), cs, ps)

        cfg = snapshot.configuration
        current_config = ProductConfiguration(
            tier_a=cfg.tier_a, tier_b=cfg.tier_b, tier_c=cfg.tier_c,
            quota_a=cfg.quota_a, quota_b=cfg.quota_b, quota_c=cfg.quota_c,
            capacity_tier=cfg.capacity_tier,
            daily_operations_spend=float(cfg.daily_operations_spend),
            daily_development_spend=float(cfg.daily_development_spend),
        )
        config_fields = {
            "tier_A": "tier_a", "tier_B": "tier_b", "tier_C": "tier_c",
            "quota_A": "quota_a", "quota_B": "quota_b", "quota_C": "quota_c",
            "capacity_tier": "capacity_tier",
            "spend_operations": "daily_operations_spend",
            "spend_development": "daily_development_spend",
        }
        previous_artifact = history.get(snapshot.day - 7)
        previous_config = (
            previous_artifact.product.configuration.current
            if previous_artifact else None
        )
        changes = []
        # config_history 同一天只保留最终配置。以上周公开快照为起点并包含
        # 周初边界日，才能捕获 Analysis 完成后、同一天发生的 Agent 配置修改。
        running_config = previous_config.model_dump() if previous_config else None
        boundary_day = snapshot.day - 7
        ordered = [
            row for row in config_rows
            if boundary_day <= int(row["day"]) <= snapshot.day
        ]
        for row in ordered:
            row_config = {
                target: row[source] for source, target in config_fields.items()
            }
            if running_config is None:
                running_config = row_config
                continue
            for source, target in config_fields.items():
                if running_config[target] != row_config[target]:
                    changes.append(ConfigurationChange(
                        day=int(row["day"]), field=target,
                        previous=running_config[target], current=row_config[target],
                    ))
            running_config = row_config

        # 正常情况下当前快照与最后一条 config_history 一致；显式比较可避免
        # 配置持久化异常被静默忽略。
        current_config_values = current_config.model_dump()
        if running_config is not None:
            for field, current_value in current_config_values.items():
                if running_config[field] != current_value:
                    changes.append(ConfigurationChange(
                        day=snapshot.day,
                        field=field,
                        previous=running_config[field],
                        current=current_value,
                    ))

        projects = [ResearchProjectSignal(**row) for row in research_rows]
        in_progress = [project for project in projects if project.status == "in_progress"]
        completed = [project for project in projects if project.status == "completed"]
        current_completed_ids = {project.project_id for project in completed}
        previous_completed_ids = (
            {project.project_id for project in previous_artifact.product.research_pipeline.completed}
            if previous_artifact else set()
        )
        current_completions = len(current_completed_ids - previous_completed_ids)
        previous_completions_observation = (
            previous_artifact.product.research_pipeline.completions.current
            if previous_artifact else _observation(None, DataStatus.INSUFFICIENT_DATA)
        )
        completions = _comparison_from_observations(
            _observation(
                current_completions,
                windows.current_7d.status if previous_artifact else DataStatus.INSUFFICIENT_DATA,
            ),
            previous_completions_observation,
        )

        return ProductSignals(
            usage=UsageSignals(
                total_units=pair("usage_total"),
                daily_average_units=pair("usage_average"),
            ),
            capacity=CapacitySignals(
                average_utilization=pair("utilization_average"),
                peak_utilization=pair("utilization_peak"),
                peak_overload_excess=pair("overload_peak_excess"),
                overload_days=pair("overload_days"),
            ),
            reliability=ReliabilitySignals(
                average_p95_ms=pair("p95_average"),
                peak_p95_ms=pair("p95_peak"),
                average_error_rate=pair("error_average"),
                peak_error_rate=pair("error_peak"),
                downtime_minutes=pair("downtime"),
                outage_days=pair("outage_days"),
            ),
            configuration=ProductConfigurationSignals(
                current=current_config,
                previous_week=previous_config,
                changes=changes,
                comparison_status=(
                    DataStatus.AVAILABLE if previous_config else DataStatus.INSUFFICIENT_DATA
                ),
            ),
            research_pipeline=ResearchPipeline(
                in_progress=in_progress,
                completed=completed,
                completions=completions,
            ),
            delivered_quality=snapshot.delivered_quality,
        )

    def _build_customer(
        self,
        snapshot: PublicWeekSnapshot,
        windows: AnalysisWindows,
        history: Mapping[int, AnalysisSignals],
        customer_base_rows: list[dict[str, Any]],
        paid_subscription_rows: list[dict[str, Any]],
        issue: dict[str, Any],
        negotiation_summary: dict[str, Any],
        negotiation_details: list[dict[str, Any]],
        negotiation_outcomes: list[dict[str, Any]],
    ) -> CustomerSignals:
        plan_values = {plan: {"individual": 0, "enterprise": 0, "seats": 0} for plan in "ABC"}
        enterprise_accounts = 0
        for row in customer_base_rows:
            plan = str(row["plan"])
            if plan not in plan_values:
                continue
            if row["customer_type"] == "small":
                plan_values[plan]["individual"] += int(row["accounts"] or 0)
            else:
                accounts = int(row["accounts"] or 0)
                seats = int(row["seats"] or 0)
                plan_values[plan]["enterprise"] += accounts
                plan_values[plan]["seats"] += seats
                enterprise_accounts += accounts

        prior = history.get(snapshot.day - 7)
        current_individual = snapshot.current_state.individual_subscribers
        current_seats = snapshot.current_state.enterprise_subscribed_seats
        previous_individual = (
            prior.customer.customer_base.active_individual_accounts.value if prior else None
        )
        previous_seats = (
            prior.customer.customer_base.active_enterprise_seats.value if prior else None
        )
        point_previous_status = DataStatus.AVAILABLE if prior else DataStatus.INSUFFICIENT_DATA

        current_paid = self._period_rows(paid_subscription_rows, windows.current_7d)
        previous_paid = self._period_rows(paid_subscription_rows, windows.previous_7d)

        def paid(rows: Sequence[Mapping[str, Any]], customer_type: str, field: str) -> int:
            return sum(
                int(row[field] or 0) for row in rows
                if row["customer_type"] == customer_type
            )

        cs, ps = windows.current_7d.status, windows.previous_7d.status
        activity = snapshot.weekly_activity
        previous_activity = prior.public_week_snapshot.weekly_activity if prior else None

        def activity_pair(field: str) -> MetricComparison:
            return _comparison(
                getattr(activity, field) if activity else None,
                getattr(previous_activity, field) if previous_activity else None,
                cs if activity else DataStatus.INSUFFICIENT_DATA,
                ps if previous_activity else DataStatus.INSUFFICIENT_DATA,
            )

        cancellations = activity_pair("cancellations")
        week_start = prior.customer.customer_base if prior else None
        starting_accounts = None
        if week_start:
            starting_accounts = (
                (week_start.active_individual_accounts.value or 0)
                + (week_start.active_enterprise_accounts.value or 0)
            )
        weekly_churn_status = (
            DataStatus.AVAILABLE
            if activity and starting_accounts and starting_accounts > 0
            else DataStatus.NOT_APPLICABLE if activity and starting_accounts == 0
            else DataStatus.INSUFFICIENT_DATA
        )
        weekly_churn = (
            activity.cancellations / starting_accounts
            if weekly_churn_status is DataStatus.AVAILABLE else None
        )

        day_28 = history.get(snapshot.day - 28)
        weekly_artifacts = [history.get(snapshot.day - offset) for offset in (0, 7, 14, 21)]
        # 当前周尚未写入 history，因此用当前快照补到四周窗口。
        cancellation_values = [activity.cancellations] if activity else []
        for artifact in weekly_artifacts[1:]:
            if artifact and artifact.public_week_snapshot.weekly_activity:
                cancellation_values.append(
                    artifact.public_week_snapshot.weekly_activity.cancellations
                )
        if day_28 and len(cancellation_values) == 4:
            base = day_28.customer.customer_base
            start_28_accounts = (
                (base.active_individual_accounts.value or 0)
                + (base.active_enterprise_accounts.value or 0)
            )
            trailing_status = (
                DataStatus.AVAILABLE if start_28_accounts > 0 else DataStatus.NOT_APPLICABLE
            )
            trailing_churn = (
                sum(cancellation_values) / start_28_accounts
                if start_28_accounts > 0 else None
            )
        else:
            trailing_status = DataStatus.INSUFFICIENT_DATA
            trailing_churn = None

        def issue_observation(key: str) -> NumericObservation:
            value = issue.get(key)
            if value is None:
                if key in {"avg_open_age", "max_open_age"} and int(issue.get("open_count") or 0) == 0:
                    return _observation(None, DataStatus.NOT_APPLICABLE)
                return _observation(None, DataStatus.INSUFFICIENT_DATA)
            return _observation(float(value) if "avg" in key else int(value))

        def issue_pair(current_key: str, previous_key: str, ratio: bool = False) -> MetricComparison:
            current_value = issue.get(current_key)
            previous_value = issue.get(previous_key)
            current_metric_status = cs
            previous_metric_status = ps
            if ratio and current_value is None and cs is DataStatus.AVAILABLE:
                current_metric_status = DataStatus.NOT_APPLICABLE
            if ratio and previous_value is None and ps is DataStatus.AVAILABLE:
                previous_metric_status = DataStatus.NOT_APPLICABLE
            return _comparison(
                float(current_value) if current_value is not None else None,
                float(previous_value) if previous_value is not None else None,
                current_metric_status,
                previous_metric_status,
            )

        outcome_counts = defaultdict(lambda: {"current": 0, "previous": 0})
        for row in negotiation_outcomes:
            period = "current" if _in_window(int(row["day"]), windows.current_7d) else "previous"
            outcome_counts[str(row["close_reason"])][period] += int(row["outcome_count"])

        details_truncated = len(negotiation_details) > self.max_enterprise_threads
        thread_details = [
            EnterpriseThreadSummary(**row)
            for row in negotiation_details[:self.max_enterprise_threads]
        ]
        open_threads = int(negotiation_summary.get("open_threads") or 0)

        return CustomerSignals(
            customer_base=CustomerBase(
                active_individual_accounts=_observation(current_individual),
                active_enterprise_accounts=_observation(enterprise_accounts),
                active_enterprise_seats=_observation(current_seats),
                individual_net_change=_comparison(
                    current_individual, previous_individual,
                    DataStatus.AVAILABLE, point_previous_status,
                ),
                enterprise_seat_net_change=_comparison(
                    current_seats, previous_seats,
                    DataStatus.AVAILABLE, point_previous_status,
                ),
                by_plan=[
                    PlanCustomerBase(
                        plan=plan,
                        individual_accounts=values["individual"],
                        enterprise_accounts=values["enterprise"],
                        enterprise_seats=values["seats"],
                    )
                    for plan, values in plan_values.items()
                ],
            ),
            new_paid_subscriptions=NewPaidSubscriptions(
                individual_accounts=_comparison(
                    paid(current_paid, "small", "accounts"),
                    paid(previous_paid, "small", "accounts"), cs, ps,
                ),
                enterprise_accounts=_comparison(
                    paid(current_paid, "large", "accounts"),
                    paid(previous_paid, "large", "accounts"), cs, ps,
                ),
                enterprise_seats=_comparison(
                    paid(current_paid, "large", "seats"),
                    paid(previous_paid, "large", "seats"), cs, ps,
                ),
            ),
            churn=ChurnSignals(
                cancellations=cancellations,
                upgrades=activity_pair("upgrades"),
                downgrades=activity_pair("downgrades"),
                weekly_account_churn_rate=_observation(weekly_churn, weekly_churn_status),
                trailing_28d_account_churn_rate=_observation(trailing_churn, trailing_status),
            ),
            issues=IssueSignals(
                open_issues=_observation(snapshot.current_state.open_issues),
                average_open_age_days=issue_observation("avg_open_age"),
                maximum_open_age_days=issue_observation("max_open_age"),
                open_over_7_days=issue_observation("over_7"),
                open_over_14_days=issue_observation("over_14"),
                opened=issue_pair("current_opened", "previous_opened"),
                resolved=issue_pair("current_resolved", "previous_resolved"),
                average_resolution_days=issue_pair(
                    "current_resolution_days", "previous_resolution_days", ratio=True
                ),
            ),
            enterprise_negotiations=EnterpriseNegotiations(
                open_threads=open_threads,
                open_seats=int(negotiation_summary.get("open_seats") or 0),
                awaiting_agent_response=int(negotiation_summary.get("awaiting_agent") or 0),
                average_waiting_days=_observation(
                    float(negotiation_summary["avg_waiting_days"])
                    if negotiation_summary.get("avg_waiting_days") is not None else None,
                    DataStatus.AVAILABLE if open_threads else DataStatus.NOT_APPLICABLE,
                ),
                maximum_waiting_days=_observation(
                    int(negotiation_summary["max_waiting_days"])
                    if negotiation_summary.get("max_waiting_days") is not None else None,
                    DataStatus.AVAILABLE if open_threads else DataStatus.NOT_APPLICABLE,
                ),
                accepted=_comparison(
                    outcome_counts["accepted"]["current"],
                    outcome_counts["accepted"]["previous"], cs, ps,
                ),
                agent_rejected=_comparison(
                    outcome_counts["agent_rejected"]["current"],
                    outcome_counts["agent_rejected"]["previous"], cs, ps,
                ),
                oldest_open_threads=thread_details,
                details_truncated=details_truncated,
            ),
        )
