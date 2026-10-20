"""保留作者的首日 PMI 读数、后续周期均值和延迟发布日程。"""

import pytest
from numpy.random import default_rng

from saas_bench.simulator.config import BenchmarkConfig
from saas_bench.simulator.database import init_database
from saas_bench.simulator.simulation import Simulator


def _advance_macro(simulator, through_day):
    results = []
    for day in range(simulator.current_day + 1, through_day + 1):
        simulator.current_day = day
        work = simulator._process_macroeconomic_cycle(simulator.get_current_config())
        results.append((simulator._macro_pmi_current, work))
    return results


@pytest.mark.parametrize(("interval", "delay"), [(30, 30), (3, 2)])
def test_pmi_publication_preserves_startup_snapshot_and_following_periods(
    make_initialized_sim, make_agent_tools, interval, delay,
):
    config = BenchmarkConfig(
        seed=42,
        macro_pmi_update_interval_days=interval,
        macro_pmi_publication_delay_days=delay,
    )
    conn, simulator, _ = make_initialized_sim(config=config, seed=42)
    tools = make_agent_tools(conn, config, seed=42)
    values = []
    try:
        for day in range(1, 3 * interval + delay + 1):
            [(pmi, work)] = _advance_macro(simulator, day)
            values.append(pmi)
            expected = []
            previous = config.macro_pmi_initial
            for period_end in range(1, day - delay + 1, interval):
                period_values = values[max(0, period_end - interval):period_end]
                average = round(sum(period_values) / len(period_values), 1)
                expected.append((period_end, average, round(average - previous, 1)))
                previous = average

            actual = conn.execute(
                "SELECT day, pmi_value, pmi_change FROM macroeconomic_conditions ORDER BY day",
            ).fetchall()
            assert [tuple(row) for row in actual] == expected
            notifications = conn.execute(
                "SELECT day FROM notifications WHERE type = 'macro_economic_update' ORDER BY day",
            ).fetchall()
            assert [row['day'] for row in notifications] == [end + delay for end, _, _ in expected]
            published_today = any(end + delay == day for end, _, _ in expected)
            assert bool(work) == published_today
            tools.set_current_day(day)
            overview = tools.get_market_overview().message
            assert ("ISM PMI:" in overview) == bool(expected)
            if expected:
                assert f"{delay} day delay" in overview
                assert "first reading is the Day 1 startup snapshot" in overview
                assert "later readings are measurement-period averages" in overview
    finally:
        conn.close()


@pytest.mark.parametrize("checkpoint_day", [17, 45])
def test_pmi_resume_preserves_partial_period_and_pending_publications(
    make_initialized_sim, tmp_path, checkpoint_day,
):
    conn, simulator, config = make_initialized_sim(seed=42)
    restored_conn = init_database(tmp_path / "restored-macro.db")
    try:
        _advance_macro(simulator, checkpoint_day)
        simulator.save_rng_states()
        conn.backup(restored_conn)
        expected_results = _advance_macro(simulator, 90)
        simulator.save_rng_states()

        restored = Simulator(restored_conn, config, default_rng(42))
        restored.initialize(resume=True)
        restored.current_day = checkpoint_day
        assert restored.restore_rng_states()
        assert _advance_macro(restored, 90) == expected_results
        restored.save_rng_states()
        assert list(restored_conn.iterdump()) == list(conn.iterdump())
    finally:
        restored_conn.close()
        conn.close()


def test_weekly_simulation_exposes_first_pmi_only_after_publication(make_initialized_sim):
    conn, simulator, _ = make_initialized_sim(seed=42)
    try:
        for _ in range(9):
            result = simulator.step_week()
            rows = conn.execute("SELECT day FROM macroeconomic_conditions ORDER BY day").fetchall()
            expected = [day for day in (1, 31) if day + 30 <= result.day]
            assert [row['day'] for row in rows] == expected
        notification_days = conn.execute(
            "SELECT day FROM notifications WHERE type = 'macro_economic_update'",
        ).fetchall()
        assert [row['day'] for row in notification_days] == [31, 61]
    finally:
        conn.close()
