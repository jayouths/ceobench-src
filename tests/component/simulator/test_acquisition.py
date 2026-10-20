"""获客公式与实际随机抽样参数的回归测试。"""

from unittest.mock import Mock

import pytest

from saas_bench.evaluation.fact_recorder import record_segment_day
from saas_bench.simulator.config import AD_CHANNELS, BenchmarkConfig


@pytest.mark.parametrize("source", ["advertising", "network", "both"])
@pytest.mark.parametrize("reputation", [0.0, 0.5, 1.0])
def test_reputation_directly_scales_acquisition(
    make_initialized_sim,
    monkeypatch,
    source,
    reputation,
):
    config = BenchmarkConfig(seed=42)
    conn, simulator, _ = make_initialized_sim(seed=42, config=config)
    if source in ("network", "both"):
        params = simulator._generate_customer_from_group("S1")
        customer_id = simulator._create_customer(params)
        simulator._create_subscription(customer_id, "A", 0.0)
    if source in ("advertising", "both"):
        channel_id = next(
            key for key, channel in AD_CHANNELS.items()
            if channel.leads_per_1000_dollars.get("S1", 0) > 0
        )
        config.targeted_ad_spend = {channel_id: {"S1": 1_000.0}}
    conn.execute(
        "UPDATE group_reputation SET reputation = ? WHERE group_id = 'S1'",
        (reputation,),
    )
    simulator.current_day = 1
    current_config = simulator.get_current_config()
    simulator._cache_step_day_globals(current_config)

    # 只替换 Poisson 抽样，验证真实获客链路传入的期望值，不依赖随机成败。
    rng = Mock()
    rng.poisson.return_value = 0
    monkeypatch.setattr(simulator, "rng", rng)
    facts = simulator._generate_new_customers(current_config)["segment_acquisition"]
    fact = facts["S1"]

    assert (fact.channel_leads_expected > 0) == (source != "network")
    assert (fact.network_leads_expected > 0) == (source != "advertising")
    expected_leads = (
        reputation
        * fact.market_capacity_multiplier
        * fact.calendar_cycle_multiplier
        * fact.macroeconomic_multiplier
        * fact.social_media_multiplier
        * fact.demand_surge_multiplier
        * (fact.channel_leads_expected + fact.network_leads_expected)
    )
    assert fact.total_leads_expected == pytest.approx(expected_leads)
    sample_call = rng.poisson.call_args_list[list(facts).index("S1")]
    assert sample_call.args == pytest.approx((expected_leads,))
    assert fact.actual_leads == 0

    record_segment_day(conn, 1, facts)
    recorded = conn.execute(
        """
        SELECT reputation, total_leads_expected, actual_leads
        FROM _eval_segment_day WHERE day = 1 AND group_id = 'S1'
        """
    ).fetchone()
    assert tuple(recorded) == pytest.approx((reputation, expected_leads, 0))
