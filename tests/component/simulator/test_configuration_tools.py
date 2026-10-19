"""复合经营配置工具的原子性回归测试。"""

from copy import deepcopy

import pytest


def _override_count(conn):
    return conn.execute("SELECT COUNT(*) FROM config_overrides").fetchone()[0]


@pytest.mark.parametrize("args", [
    {"by_group": {"S1": 100}, "by_plan": {"INVALID": 200}},
    {"by_group": {"S1": True}},
    {"by_group": {"S1": 100}, "by_plan": {"A": float("nan")}},
    {"by_group": {"S1": 100}, "by_group_plan": {"E1": {"C": float("inf")}}},
    {"by_group": {"S1": 100}, "by_customer": {"42": -1}},
    {"by_group": {"S1": 100}, "by_customer": {"0": 10}},
    {"by_group": {"S1": 100}, "by_customer": {"-1": 10}},
    {"by_group": {"S1": 100}, "by_customer": {1.5: 10}},
    {"by_group": {"S1": 100}, "by_customer": {True: 10}},
    {"by_group": {"S1": 1e308}, "by_plan": {"A": 1e308}},
])
def test_targeted_ops_failure_keeps_all_scopes_unchanged(
    make_initialized_sim,
    make_agent_tools,
    args,
):
    conn, _, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)
    config.targeted_ops_spend = {"E1": 10.0}
    config.targeted_ops_spend_by_plan = {"A": 20.0}
    config.targeted_ops_spend_by_group_plan = {"E2": {"C": 30.0}}
    config.targeted_ops_spend_by_customer = {42: 40.0}
    fields = (
        "targeted_ops_spend", "targeted_ops_spend_by_plan",
        "targeted_ops_spend_by_group_plan", "targeted_ops_spend_by_customer",
    )
    before_config = deepcopy([getattr(config, field) for field in fields])
    before_records = _override_count(conn)
    before_ledger = list(conn.execute("SELECT * FROM ledger"))

    result = tools.set_targeted_ops_spend(**args)

    assert not result.success
    assert [getattr(config, field) for field in fields] == before_config
    assert _override_count(conn) == before_records
    assert list(conn.execute("SELECT * FROM ledger")) == before_ledger


def test_ads_strength_failure_keeps_all_scopes_unchanged(
    make_initialized_sim,
    make_agent_tools,
):
    conn, _, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)
    config.ads_strength_global = 0.1
    config.ads_strength_by_group = {"S1": 0.2}
    before_records = _override_count(conn)

    result = tools.set_ads_strength(
        global_strength=0.4,
        by_group={"INVALID": 0.3},
    )

    assert not result.success
    assert config.ads_strength_global == 0.1
    assert config.ads_strength_by_group == {"S1": 0.2}
    assert _override_count(conn) == before_records


def test_lead_promotion_failure_keeps_all_scopes_unchanged(
    make_initialized_sim,
    make_agent_tools,
):
    conn, _, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)
    config.lead_promotion_global = 1.0
    config.lead_promotion_by_group = {"E1": 2.0}
    before_records = _override_count(conn)

    result = tools.set_lead_promotion(
        global_promotion=5,
        by_group={"S1": 6},
        by_channel_group={"social_media": {"INVALID": 7}},
    )

    assert not result.success
    assert config.lead_promotion_global == 1.0
    assert config.lead_promotion_by_group == {"E1": 2.0}
    assert _override_count(conn) == before_records


def test_ongoing_promotion_failure_keeps_all_scopes_unchanged(
    make_initialized_sim,
    make_agent_tools,
):
    conn, _, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)
    config.promotion_global = 1.0
    config.promotion_by_customer = {42: 2.0}
    before_records = _override_count(conn)

    result = tools.set_promotion(
        global_promotion=5,
        by_customer={"7": 6},
        by_group_plan={"S1": {"INVALID": 7}},
    )

    assert not result.success
    assert config.promotion_global == 1.0
    assert config.promotion_by_customer == {42: 2.0}
    assert _override_count(conn) == before_records
