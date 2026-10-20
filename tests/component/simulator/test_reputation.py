"""日常声誉与社交抽样权重保持独立。"""

import numpy as np
import pytest

from saas_bench.simulator.config import BenchmarkConfig
from saas_bench.simulator.database import get_group_reputation


def _customer_events():
    return {
        1: {
            'group_id': 'S1', 'new_satisfaction': 0.2,
            'days_subscribed': 90, 'seat_count': 1,
            'events': [], 'satisfaction_change': 0.0,
            'is_contract_locked': False, 'penalties': {},
        },
        2: {
            'group_id': 'S1', 'new_satisfaction': -0.2,
            'days_subscribed': 5, 'seat_count': 200,
            'events': ['outage', 'contract_dissatisfaction'],
            'satisfaction_change': -0.1,
            'is_contract_locked': True, 'penalties': {'outage': 0.2},
        },
    }


@pytest.mark.parametrize('path', ['normal', 'suppressed_fallback', 'suppressed_vectorized'])
def test_daily_reputation_does_not_multiply_social_sampling_weights(
    make_initialized_sim, path,
):
    conn, simulator, config = make_initialized_sim(seed=42)
    try:
        assert config.contract_dissatisfaction_sampling_weight_multiplier == 1.5
        simulator.current_day = 7
        simulator._suppress_customer_posts = path != 'normal'
        if path == 'suppressed_vectorized':
            simulator._last_sat_arrays = {
                'satisfaction': np.array([0.2, -0.2]),
                'group_idx': np.array([0, 0]),
                'unique_groups': ['S1'], 'n_customers': 2,
            }
        initial = get_group_reputation(conn, 'S1')

        posts, _ = simulator._generate_sampled_social_posts(_customer_events(), [])

        # 两名客户按满意度贡献；负向斜率为两倍，再按 log2(2)/2 归一化。
        expected_delta = (0.0005 * 0.2 - 0.0005 * 0.2 * 2) / 2
        assert get_group_reputation(conn, 'S1') == pytest.approx(initial + expected_delta)
        if path == 'normal':
            weights = {post['customer_id']: post['weight'] for post in posts}
            assert weights[2] > weights[1]
        else:
            assert posts == []
    finally:
        conn.close()


@pytest.mark.parametrize('multiplier', [1.5, 3.0])
def test_contract_multiplier_changes_sampling_weight_not_reputation(
    make_initialized_sim, multiplier,
):
    config = BenchmarkConfig(
        seed=42, contract_dissatisfaction_sampling_weight_multiplier=multiplier,
    )
    conn, simulator, _ = make_initialized_sim(config=config, seed=42)
    try:
        simulator.current_day = 7
        initial = get_group_reputation(conn, 'S1')

        posts, _ = simulator._generate_sampled_social_posts(_customer_events(), [])

        weights = {post['customer_id']: post['weight'] for post in posts}
        assert weights[1] == pytest.approx(3.0)
        # 保留原有极端满意度、新客户、变化幅度、事件数和席位的抽样倍率。
        assert weights[2] == pytest.approx(3.0 * 1.8 * 1.5 * 3.0 * 11.0 * multiplier * 200)
        assert get_group_reputation(conn, 'S1') == pytest.approx(initial - 0.00005)
    finally:
        conn.close()
