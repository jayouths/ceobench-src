"""模拟世界随机状态的断点恢复回归测试。"""

import numpy as np
from numpy.random import default_rng

from saas_bench.simulator.config import BenchmarkConfig, ScenarioPack
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
