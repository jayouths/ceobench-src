"""模拟世界随机状态的断点恢复回归测试。"""

from copy import deepcopy
import json

import numpy as np
import pytest
from numpy.random import default_rng

from saas_bench.simulator.config import AD_CHANNELS, BenchmarkConfig, ScenarioPack
from saas_bench.simulator.database import init_database
from saas_bench.simulator.shocks import ShockManager
from saas_bench.simulator.simulation import Simulator


def test_resume_restores_competitor_post_and_demand_shock_rngs(tmp_path):
    """恢复后的三条随机流必须从连续运行的同一位置继续。"""
    seed = 42
    config = BenchmarkConfig(seed=seed)
    scenario = ScenarioPack(
        name="Always surge",
        description="Exercise every demand-shock draw",
        demand_surge_prob=1.0,
    )
    conn = init_database(tmp_path / "world.db")

    continuous_rng = default_rng(seed)
    continuous_simulator = Simulator(conn, config, continuous_rng)
    continuous_simulator.initialize()
    continuous_shocks = ShockManager(conn, continuous_rng, scenario)

    continuous_simulator._competitor_post_noise_rng.random(3)
    for day in range(1, 8):
        continuous_shocks.check_and_generate_shocks(day)
    continuous_simulator.save_rng_states()

    expected_main = continuous_rng.random(4)
    expected_competitor_noise = (
        continuous_simulator._competitor_post_noise_rng.random(4)
    )
    expected_shocks = continuous_shocks.rng.random(4)

    resumed_rng = default_rng(seed)
    resumed_simulator = Simulator(conn, config, resumed_rng)
    resumed_simulator.initialize(resume=True)
    # 与正式恢复顺序一致：先重建 ShockManager，再恢复各随机流位置。
    resumed_shocks = ShockManager(conn, resumed_rng, scenario)

    assert resumed_simulator.restore_rng_states()
    assert resumed_shocks.restore_rng_state()
    np.testing.assert_array_equal(resumed_rng.random(4), expected_main)
    np.testing.assert_array_equal(
        resumed_simulator._competitor_post_noise_rng.random(4),
        expected_competitor_noise,
    )
    np.testing.assert_array_equal(resumed_shocks.rng.random(4), expected_shocks)


@pytest.mark.parametrize("checkpoint_day", [
    0, 7,
    pytest.param(28, marks=pytest.mark.slow),
    pytest.param(35, marks=pytest.mark.slow),
])
def test_resume_matches_continuous_business_state(tmp_path, monkeypatch, checkpoint_day):
    """比较真实日推进结果，覆盖初始状态、跨月漂移和已漂移状态的恢复。"""
    initial_channels = {
        name: deepcopy(channel.leads_per_1000_dollars)
        for name, channel in AD_CHANNELS.items()
    }
    for name, channel in AD_CHANNELS.items():
        monkeypatch.setattr(channel, "leads_per_1000_dollars", deepcopy(initial_channels[name]))

    seed = 42
    config = BenchmarkConfig(
        seed=seed,
        default_price_A=5.0,
        targeted_ad_spend={"search_ads": {"S1": 500.0}},
        default_spend_development=100.0,
        default_spend_operations=100.0,
    )
    scenario = ScenarioPack(name="test", description="test", demand_surge_prob=0.3)
    conn = init_database(tmp_path / "continuous.db")
    rng = default_rng(seed)
    simulator = Simulator(conn, config, rng)
    simulator.initialize()
    shocks = ShockManager(conn, rng, scenario)

    def advance(sim, manager, days):
        results = []
        for _ in range(days):
            manager.check_and_generate_shocks(sim.current_day + 1)
            results.append(sim.step_day())
        return results

    resumed_conn = init_database(tmp_path / "resumed.db")
    try:
        advance(simulator, shocks, checkpoint_day)
        conn.backup(resumed_conn)
        expected_mu = {
            gid: simulator._get_involuntary_churn_mu(gid)
            for gid in simulator.all_groups
        }
        expected_results = advance(simulator, shocks, 14)
        assert conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] > 0
        expected_database = list(conn.iterdump())

        # 新进程从初始渠道配置开始，不能继承连续分支已漂移的值，必须由断点恢复。
        for name, channel in AD_CHANNELS.items():
            monkeypatch.setattr(channel, "leads_per_1000_dollars", deepcopy(initial_channels[name]))
        resumed_rng = default_rng(seed)
        resumed = Simulator(resumed_conn, deepcopy(config), resumed_rng)
        resumed.initialize(resume=True)
        resumed_shocks = ShockManager(resumed_conn, resumed_rng, scenario)
        resumed.current_day = checkpoint_day
        if checkpoint_day > 0:
            assert resumed.restore_rng_states()
            assert resumed_shocks.restore_rng_state()
        assert {
            gid: resumed._get_involuntary_churn_mu(gid)
            for gid in resumed.all_groups
        } == expected_mu
        assert advance(resumed, resumed_shocks, 14) == expected_results
        assert list(resumed_conn.iterdump()) == expected_database
    finally:
        resumed_conn.close()
        conn.close()


@pytest.mark.parametrize("missing", ["_customer_pick_rng", "_sim_state", "group:S1", "macro"])
def test_resume_rejects_partial_saved_state(make_initialized_sim, missing):
    conn, sim, _ = make_initialized_sim(seed=42)
    try:
        sim.step_day()
        state = json.loads(conn.execute("SELECT state_json FROM _rng_states WHERE name='all'").fetchone()[0])
        if missing == "group:S1":
            del state['_group_rngs']['S1']
        elif missing == "macro":
            del state['_sim_state']['_macro_pmi_current']
        else:
            del state[missing]
        conn.execute("UPDATE _rng_states SET state_json=? WHERE name='all'", (json.dumps(state),))
        with pytest.raises((KeyError, RuntimeError)):
            sim.restore_rng_states()
    finally:
        conn.close()


def test_resume_keeps_bankruptcy_shutdown_state(tmp_path):
    config = BenchmarkConfig(seed=42, initial_cash=1, default_spend_development=100)
    conn = init_database(tmp_path / "bankrupt.db")
    try:
        sim = Simulator(conn, config, default_rng(42))
        sim.initialize()
        sim.step_day()
        assert sim.shutdown_mode
        restored = Simulator(conn, config, default_rng(42))
        restored.initialize(resume=True)
        assert restored.restore_rng_states()
        assert restored.shutdown_mode
    finally:
        conn.close()
