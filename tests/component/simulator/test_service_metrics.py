"""服务指标使用共同随机条件，但仍保留经营配置的真实影响。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from saas_bench.simulator.config import CAPACITY_TIERS


def test_service_metrics_ignore_main_rng_position(make_initialized_sim):
    conn, simulator, _ = make_initialized_sim(seed=42)
    try:
        settings = simulator.get_current_config()
        for day in (1, 7, 30):
            simulator.current_day = day
            expected = simulator._compute_service_metrics(0, settings)
            simulator.rng.random(100)
            before = deepcopy(simulator.rng.bit_generator.state)
            assert simulator._compute_service_metrics(0, settings) == expected
            assert simulator.rng.bit_generator.state == before
    finally:
        conn.close()


def test_outage_branch_does_not_change_latency_or_error_noise(make_initialized_sim):
    conn, simulator, config = make_initialized_sim(seed=42)
    try:
        simulator.current_day = 7
        settings = simulator.get_current_config()
        config.ops_outage_min_prob = 0.0
        config.base_outage_prob = 0.0
        without_outage = simulator._compute_service_metrics(0, settings)
        config.base_outage_prob = 1.0
        with_outage = simulator._compute_service_metrics(0, settings)

        assert without_outage[1:3] == (False, 0)
        assert with_outage[1] is True
        assert with_outage[2] in (10, 30, 90)
        assert without_outage[3:] == with_outage[3:]
    finally:
        conn.close()


def test_service_randomness_is_keyed_by_seed_and_day(make_initialized_sim):
    conn, simulator, config = make_initialized_sim(seed=42)
    try:
        settings = simulator.get_current_config()
        simulator.current_day = 7
        initial = simulator._compute_service_metrics(0, settings)
        simulator.current_day = 8
        assert simulator._compute_service_metrics(0, settings) != initial
        simulator.current_day = 7
        config.seed = 43
        assert simulator._compute_service_metrics(0, settings) != initial
        config.seed = 42
        assert simulator._compute_service_metrics(0, settings) == initial
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("usage_ratio", "operations_spend", "expected_outage"),
    [(1.0, 0, True), (1.0, 500, False), (1.5, 500, True)],
)
def test_service_randomness_preserves_operating_formulas(
    make_initialized_sim, monkeypatch, usage_ratio, operations_spend, expected_outage,
):
    conn, simulator, config = make_initialized_sim(seed=42)
    try:
        simulator.current_day = 7
        settings = simulator.get_current_config()
        settings['spend_operations'] = operations_spend
        capacity = CAPACITY_TIERS[settings['capacity_tier']]['capacity_units']
        random_source = SimpleNamespace(
            random=lambda: 0.02,
            choice=lambda values, p: 30,
            normal=lambda mean, std: 5.0 if std == config.p95_noise_std else -0.001,
        )
        monkeypatch.setattr(
            "saas_bench.simulator.simulation.event_rng", lambda *args: random_source,
        )
        overload, outage, downtime, latency, error = simulator._compute_service_metrics(
            int(capacity * usage_ratio), settings,
        )

        assert overload == pytest.approx(usage_ratio - 1.0)
        assert outage is expected_outage
        assert downtime == (30 if expected_outage else 0)
        assert latency == pytest.approx(config.p95_base_ms + config.p95_overload_factor * overload + 5.0)
        assert error == pytest.approx(config.error_rate_base + config.error_rate_overload_factor * overload - 0.001)
    finally:
        conn.close()
