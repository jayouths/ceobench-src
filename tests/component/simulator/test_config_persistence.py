"""模拟器持久化经营配置回归测试。"""

from saas_bench.runtime.server_entry import _build_resumed_config


def test_resume_restores_latest_advanced_business_config(
    make_initialized_sim,
    make_agent_tools,
):
    conn, _, config = make_initialized_sim(seed=42)
    tools = make_agent_tools(conn, config, seed=42)

    assert tools.set_targeted_ad_spend(
        {"social_media": {"S1": 100}}
    ).success
    # 同一类配置可能修改多次，恢复时必须以最后一次成功快照为准。
    assert tools.set_targeted_ad_spend(
        {"search_ads": {"S1": 234}}
    ).success
    assert tools.set_targeted_ops_spend(
        by_group={"S1": 300},
        by_plan={"A": 200},
        by_group_plan={"E1": {"C": 150}},
        by_customer={"101": 50},
    ).success
    assert tools.set_targeted_dev_spend({"E1": 400}).success
    assert tools.set_ads_strength(
        global_strength=0.1,
        by_group={"S2": 0.2},
        by_customer={"102": 0.3},
    ).success
    assert tools.set_lead_promotion(
        global_promotion=1,
        by_group={"S1": 2},
        by_channel={"search_ads": 3},
        by_channel_group={"social_media": {"S2": 4}},
    ).success
    assert tools.set_promotion(
        global_promotion=5,
        by_group={"E1": 6},
        by_customer={"103": 7},
        by_group_plan={"S1": {"A": 8}},
    ).success

    restored = _build_resumed_config(
        conn,
        {
            "seed": 42,
            "total_days": 497,
            "initial_cash": 1_000_000,
        },
    )

    assert restored.targeted_ad_spend == {"search_ads": {"S1": 234}}
    assert restored.targeted_ops_spend == {"S1": 300.0}
    assert restored.targeted_ops_spend_by_plan == {"A": 200.0}
    assert restored.targeted_ops_spend_by_group_plan == {"E1": {"C": 150.0}}
    assert restored.targeted_ops_spend_by_customer == {101: 50.0}
    assert restored.targeted_dev_spend == {"E1": 400}
    assert restored.ads_strength_global == 0.1
    assert restored.ads_strength_by_group == {"S2": 0.2}
    assert restored.ads_strength_by_customer == {102: 0.3}
    assert restored.lead_promotion_global == 1.0
    assert restored.lead_promotion_by_group == {"S1": 2.0}
    assert restored.lead_promotion_by_channel == {"search_ads": 3.0}
    assert restored.lead_promotion_by_channel_group == {
        "social_media": {"S2": 4.0}
    }
    assert restored.promotion_global == 5.0
    assert restored.promotion_by_group == {"E1": 6.0}
    assert restored.promotion_by_customer == {103: 7.0}
    assert restored.promotion_by_group_plan == {"S1": {"A": 8.0}}


def test_targeted_ops_response_uses_active_config(
    make_initialized_sim,
    make_agent_tools,
):
    """工具回执应说明当前实验系数，不应硬编码默认值。"""
    conn, _, config = make_initialized_sim(seed=42)
    config.individual_ops_scale = 0.7
    config.enterprise_ops_scale = 0.2
    tools = make_agent_tools(conn, config, seed=42)

    result = tools.set_targeted_ops_spend(by_group={"S1": 100})

    assert result.success
    assert "scale_g = 0.7 for individual groups" in result.message
    assert "and 0.2 for enterprise groups" in result.message
