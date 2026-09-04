"""Analysis 确定性经营信号测试。"""

import pytest

from saas_bench.agents.bash_agent.analysis.signal_catalog import SIGNAL_CATALOG
from saas_bench.agents.bash_agent.analysis.signal_models import (
    AnalysisSignals,
    DataStatus,
)
from saas_bench.agents.bash_agent.analysis.signals import (
    SignalCollector,
    build_analysis_windows,
)
from saas_bench.agents.bash_agent.analysis.signal_queries import issue_summary
from saas_bench.simulator.public_week_snapshot import build_public_week_snapshot
from tests.support.harness import make_analysis_pipeline


def _direct_query(conn):
    def query(sql):
        return [dict(row) for row in conn.execute(sql).fetchall()]

    return query


def test_analysis_windows_distinguish_incomplete_periods():
    day_0 = build_analysis_windows(0)
    day_7 = build_analysis_windows(7)
    day_28 = build_analysis_windows(28)

    assert day_0.current_7d.status is DataStatus.INSUFFICIENT_DATA
    assert day_7.current_7d.model_dump() == {
        "start_day": 1,
        "end_day": 7,
        "covered_days": 7,
        "required_days": 7,
        "status": DataStatus.AVAILABLE,
    }
    assert day_7.previous_7d.status is DataStatus.INSUFFICIENT_DATA
    assert day_28.recent_28d.status is DataStatus.AVAILABLE


def test_day_zero_signals_use_null_instead_of_false_zero(make_initialized_sim):
    conn, _, _ = make_initialized_sim(seed=42)
    signals = SignalCollector(_direct_query(conn)).collect(
        build_public_week_snapshot(conn, 0)
    )

    leads = signals.market.effective_leads.individual
    assert leads.current.value is None
    assert leads.current.status is DataStatus.INSUFFICIENT_DATA
    assert signals.finance.current_cash.value == pytest.approx(1_000_000)
    assert signals.finance.runway.cash_runway_days.status is DataStatus.INSUFFICIENT_DATA
    assert signals.customer.issues.open_issues.value == 0
    assert signals.customer.issues.average_open_age_days.status is DataStatus.NOT_APPLICABLE


def test_issue_resolution_signals_exclude_churn_closures(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    customer_ids = []
    for _ in range(2):
        cursor = conn.execute(
            """
            INSERT INTO customers (
                customer_type, group_id, created_day,
                steepness_left, steepness_right, c_max, usage_demand,
                quality_sensitivity, price_sensitivity, willingness_to_pay,
                usage_scale, patience, seat_count
            )
            VALUES ('small', 'S1', 1, 0.01, 0.01, 100.0, 10.0,
                    0.5, 0.5, 100.0, 1.0, 0.5, 1)
            """
        )
        customer_ids.append(cursor.lastrowid)
    conn.executemany(
        """
        INSERT INTO issues (
            customer_id, group_id, open_day, days_open,
            status, resolved_day, resolution_type
        ) VALUES (?, 'S1', ?, ?, 'resolved', ?, ?)
        """,
        [
            (customer_ids[0], 3, 2, 5, "ops_resolved"),
            (customer_ids[1], 1, 4, 5, "customer_churned"),
        ],
    )

    row = conn.execute(issue_summary(1, 7, -6, 0)).fetchone()

    assert row["current_resolved"] == 1
    assert row["current_resolution_days"] == pytest.approx(2.0)


def test_four_week_signal_pipeline_computes_all_roles(
    make_initialized_sim,
    make_agent_tools,
):
    conn, simulator, config = make_initialized_sim(seed=42)
    tools = make_agent_tools(conn, config, seed=42)
    actions = [
        tools.set_prices({"A": 9, "B": 29, "C": 79}),
        tools.set_model_tiers({"A": 1, "B": 2, "C": 4}),
        tools.set_usage_quotas({"A": 150, "B": 1000, "C": 3000}),
        tools.set_capacity_tier(1),
        tools.set_daily_spend({"operations": 300, "development": 300}),
        tools.set_targeted_ad_spend({"social_media": {"S1": 750}}),
    ]
    assert all(action.success for action in actions)

    collector = SignalCollector(_direct_query(conn))
    history = {
        0: collector.collect(build_public_week_snapshot(conn, 0)),
    }
    snapshots = {0: history[0].public_week_snapshot}
    for _ in range(4):
        result = simulator.step_week()
        snapshot = build_public_week_snapshot(conn, result.day, result)
        signals = collector.collect(snapshot, history)
        history[result.day] = signals
        snapshots[result.day] = snapshot

    day_7 = history[7]
    day_14 = history[14]
    day_28 = history[28]

    assert day_7.market.effective_leads.individual.current.value == (
        snapshots[7].weekly_activity.new_individual_leads
    )
    assert day_14.market.effective_leads.individual.previous.value == (
        day_7.market.effective_leads.individual.current.value
    )
    assert sum(
        group.leads.current.value or 0
        for group in day_28.market.effective_leads.by_group
    ) == day_28.market.effective_leads.total_accounts.current.value
    assert sum(
        group.leads.previous.value or 0
        for group in day_28.market.effective_leads.by_group
    ) == history[21].market.effective_leads.total_accounts.current.value
    paid = day_14.market.paid_acquisition.overall
    assert paid.raw_cpl.current.value == pytest.approx(
        paid.spend.current.value / paid.raw_leads.current.value
    )

    assert day_7.finance.net_cash_flow.current.value == pytest.approx(
        snapshots[7].current_state.cash - snapshots[0].current_state.cash
    )
    assert day_28.finance.runway.coverage_days == 28
    assert day_28.finance.runway.cash_runway_days.status is DataStatus.AVAILABLE

    assert day_14.product.usage.total_units.comparison_status is DataStatus.AVAILABLE
    assert day_14.product.capacity.peak_utilization.current.value >= 0
    assert day_14.product.capacity.peak_overload_excess.current.value >= 0
    assert day_14.product.configuration.model_tier.C.current.value == 4

    assert day_14.customer.customer_base.active_individual_accounts.value == (
        snapshots[14].current_state.individual_subscribers
    )
    assert day_14.customer.new_paid_subscriptions.individual_accounts.current.value >= 0
    assert day_14.customer.issues.open_issues.value == snapshots[14].current_state.open_issues

    payload = day_28.model_dump_json()
    assert AnalysisSignals.model_validate_json(payload) == day_28


def test_week_boundary_actions_are_included_in_finance_and_product_signals(
    make_initialized_sim,
    make_agent_tools,
):
    conn, simulator, config = make_initialized_sim(seed=42)
    collector = SignalCollector(_direct_query(conn))
    day_0 = collector.collect(build_public_week_snapshot(conn, 0))

    tools = make_agent_tools(conn, config, day=0, seed=42)
    research = tools.start_research_project(1)
    configuration = tools.set_model_tiers({"A": 2})
    assert research.success
    assert configuration.success

    result = simulator.step_week()
    day_7 = collector.collect(
        build_public_week_snapshot(conn, result.day, result),
        {0: day_0},
    )

    assert day_7.finance.costs.one_time_investment.current.value == pytest.approx(
        research.data["cost"]
    )
    assert day_7.finance.net_cash_flow.current.value == pytest.approx(
        day_7.public_week_snapshot.current_state.cash
        - day_0.public_week_snapshot.current_state.cash
    )
    tier_a = day_7.product.configuration.model_tier.A
    assert tier_a.previous.value == (
        day_0.public_week_snapshot.configuration.tier_a
    )
    assert tier_a.current.value == 2
    assert tier_a.direction.value == "up"


def test_consecutive_boundaries_do_not_duplicate_costs_and_keep_final_config(
    make_initialized_sim,
    make_agent_tools,
):
    conn, simulator, config = make_initialized_sim(seed=42)
    collector = SignalCollector(_direct_query(conn))
    day_0 = collector.collect(build_public_week_snapshot(conn, 0))

    day_0_tools = make_agent_tools(conn, config, day=0, seed=42)
    first_research = day_0_tools.start_research_project(1)
    assert first_research.success
    assert day_0_tools.set_model_tiers({"A": 2}).success

    first_week = simulator.step_week()
    day_7 = collector.collect(
        build_public_week_snapshot(conn, first_week.day, first_week),
        {0: day_0},
    )

    day_7_tools = make_agent_tools(conn, config, day=7, seed=42)
    second_research = day_7_tools.start_research_project(2)
    assert second_research.success
    assert day_7_tools.set_model_tiers({"A": 3}).success
    assert day_7_tools.set_model_tiers({"A": 4}).success
    assert day_7_tools.set_capacity_tier(1).success

    second_week = simulator.step_week()
    # 从 JSON 恢复上周信号，覆盖真实断点恢复使用的数据路径。
    restored_day_7 = AnalysisSignals.model_validate_json(day_7.model_dump_json())
    day_14 = collector.collect(
        build_public_week_snapshot(conn, second_week.day, second_week),
        {0: day_0, 7: restored_day_7},
    )

    assert day_7.finance.costs.one_time_investment.current.value == pytest.approx(
        first_research.data["cost"]
    )
    assert day_14.finance.costs.one_time_investment.current.value == pytest.approx(
        second_research.data["cost"]
    )
    assert day_14.finance.costs.one_time_investment.previous.value == pytest.approx(
        first_research.data["cost"]
    )
    tier_a = day_14.product.configuration.model_tier.A
    assert tier_a.previous.value == 2
    assert tier_a.current.value == 4
    assert tier_a.direction.value == "up"
    capacity = day_14.product.configuration.capacity_tier
    assert capacity.previous.value == 0
    assert capacity.current.value == 1
    assert capacity.direction.value == "up"


def test_day_zero_configuration_has_current_values_without_direction(
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim(seed=42)
    signals = SignalCollector(_direct_query(conn)).collect(
        build_public_week_snapshot(conn, 0)
    )

    tier_a = signals.product.configuration.model_tier.A
    assert tier_a.current.status is DataStatus.AVAILABLE
    assert tier_a.previous.status is DataStatus.INSUFFICIENT_DATA
    assert tier_a.direction is None
    assert "direction" not in tier_a.model_dump(mode="json")


def test_signal_catalog_covers_each_role():
    assert {key.split(".", 1)[0] for key in SIGNAL_CATALOG} == {
        "market", "finance", "product", "customer",
    }


def test_pipeline_writes_and_reuses_weekly_signal_artifact(
    tmp_path,
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim(seed=42)
    snapshot = build_public_week_snapshot(conn, 0)
    pipeline = make_analysis_pipeline(
        tmp_path,
        query_public_rows=_direct_query(conn),
    )
    payload = {
        "day": 0,
        "dashboard": "dashboard",
        "public_week_snapshot": snapshot.to_dict(),
    }

    first = pipeline.ensure_signals(payload)
    path = tmp_path / "analysis" / "day_000" / "signals.json"
    assert path.is_file()

    pipeline.query_public_rows = lambda sql: (_ for _ in ()).throw(
        AssertionError("existing same-day artifact must be reused")
    )
    second = pipeline.ensure_signals(payload)

    assert second == first


def test_disabled_analysis_adds_no_queries_or_artifacts(tmp_path):
    pipeline = make_analysis_pipeline(
        tmp_path,
        enabled=False,
        query_public_rows=lambda sql: (_ for _ in ()).throw(
            AssertionError("disabled Analysis must not query public data")
        ),
    )

    assert pipeline.ensure_signals({"dashboard": "baseline"}) is None
    assert not (tmp_path / "analysis").exists()


def test_resume_prunes_only_analysis_artifacts_after_checkpoint(tmp_path):
    pipeline = make_analysis_pipeline(tmp_path)
    for day in (0, 7, 14):
        directory = tmp_path / "analysis" / f"day_{day:03d}"
        directory.mkdir(parents=True)
        (directory / "signals.json").write_text("{}")

    pipeline.prune_artifacts_after(7)

    assert (tmp_path / "analysis" / "day_000").is_dir()
    assert (tmp_path / "analysis" / "day_007").is_dir()
    assert not (tmp_path / "analysis" / "day_014").exists()
