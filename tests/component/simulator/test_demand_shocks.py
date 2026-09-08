"""需求冲击生效窗口回归测试。"""

import json

from numpy.random import default_rng

from saas_bench.simulator.config import ScenarioPack
from saas_bench.simulator.shocks import ShockManager


def test_future_demand_surge_only_applies_during_its_active_window(
    make_initialized_sim,
):
    """预生成的未来冲击不得在开始日前生效，结束日当天也应失效。"""
    conn, simulator, _ = make_initialized_sim(seed=42)
    details = {
        "duration_days": 3,
        "lead_multiplier": 4.0,
        "end_day": 10,
        "_active": True,
    }
    conn.execute(
        "INSERT INTO events (day, type, details_json) VALUES (7, 'demand_surge', ?)",
        (json.dumps(details),),
    )
    manager = ShockManager(
        conn,
        default_rng(42),
        ScenarioPack(name="test", description="test", demand_surge_prob=0.0),
    )

    expected_by_day = {1: 1.0, 6: 1.0, 7: 4.0, 9: 4.0, 10: 1.0}
    for day, expected in expected_by_day.items():
        simulator.current_day = day
        assert simulator._get_surge_lead_multiplier() == expected
        assert manager.apply_shock_effects(day, 1.0, 0.0)[0] == expected
