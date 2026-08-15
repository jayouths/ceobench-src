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
from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.public_week_snapshot import build_public_week_snapshot


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
    assert day_14.product.configuration.current.tier_c == 4

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
    assert day_7.product.configuration.previous_week == day_0.product.configuration.current
    assert any(
        change.day == 0
        and change.field == "tier_a"
        and change.previous == day_0.product.configuration.current.tier_a
        and change.current == 2
        for change in day_7.product.configuration.changes
    )


def test_signal_catalog_covers_each_role():
    assert {key.split(".", 1)[0] for key in SIGNAL_CATALOG} == {
        "market", "finance", "product", "customer",
    }


def test_runner_writes_and_reuses_weekly_signal_artifact(
    tmp_path,
    make_initialized_sim,
):
    conn, _, _ = make_initialized_sim(seed=42)
    snapshot = build_public_week_snapshot(conn, 0)
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path
    runner.analysis_enabled = True
    runner.analysis_module_config = {"max_enterprise_threads": 50}
    runner._query_public_rows = _direct_query(conn)
    payload = {
        "day": 0,
        "dashboard": "dashboard",
        "public_week_snapshot": snapshot.to_dict(),
    }

    first = runner._ensure_analysis_signals(payload)
    path = tmp_path / "analysis" / "day_000" / "signals.json"
    assert path.is_file()

    runner._query_public_rows = lambda sql: (_ for _ in ()).throw(
        AssertionError("existing same-day artifact must be reused")
    )
    second = runner._ensure_analysis_signals(payload)

    assert second == first


def test_disabled_analysis_adds_no_queries_or_artifacts(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path
    runner.analysis_enabled = False
    runner._query_public_rows = lambda sql: (_ for _ in ()).throw(
        AssertionError("disabled Analysis must not query public data")
    )

    assert runner._ensure_analysis_signals({"dashboard": "baseline"}) is None
    assert not (tmp_path / "analysis").exists()


def test_resume_prunes_only_analysis_artifacts_after_checkpoint(tmp_path):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path
    for day in (0, 7, 14):
        directory = tmp_path / "analysis" / f"day_{day:03d}"
        directory.mkdir(parents=True)
        (directory / "signals.json").write_text("{}")

    runner._prune_analysis_artifacts_after(7)

    assert (tmp_path / "analysis" / "day_000").is_dir()
    assert (tmp_path / "analysis" / "day_007").is_dir()
    assert not (tmp_path / "analysis" / "day_014").exists()
