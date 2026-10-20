"""保留作者以消息序号驱动的谈判耐心、让步和退出规则。"""

import pytest

import saas_bench.simulator.enterprise as enterprise
from saas_bench.simulator.database import add_enterprise_turn


@pytest.fixture
def negotiation(make_initialized_sim, make_agent_tools, monkeypatch):
    conn, simulator, config = make_initialized_sim(seed=42)
    params = simulator._generate_customer_from_group("E1")
    customer_id = simulator._create_customer(params)
    simulator._create_enterprise_lead(customer_id, params)
    conn.execute("""
        UPDATE customers SET c_max = 200, q_min = 0.1, q_max = 0.9,
            max_negotiation_turns = 6 WHERE customer_id = ?
    """, (customer_id,))
    conn.execute("""
        UPDATE customer_state SET current_c_max = 200, current_q_min = 0.1,
            current_q_max = 0.9 WHERE customer_id = ?
    """, (customer_id,))
    thread_id = conn.execute(
        "SELECT thread_id FROM enterprise_turns WHERE customer_id = ? LIMIT 1",
        (customer_id,),
    ).fetchone()[0]
    monkeypatch.setattr(simulator, "_get_customer_quality_noise", lambda customer: 1.0)
    monkeypatch.setattr(
        enterprise, "get_qualities_for_all_plans_batch",
        lambda *args, **kwargs: {customer_id: {"A": 1.0, "B": 1.0, "C": 1.0}},
    )
    return conn, simulator, config, make_agent_tools(conn, config, seed=42), customer_id, thread_id


def test_original_patience_distributions_are_retained(make_initialized_sim):
    _, simulator, _ = make_initialized_sim(seed=42)
    expected = {
        "E1": (6.0, 2.0), "E2": (10.0, 3.0), "E3": (14.0, 4.0),
        "D_E01": (14.0, 4.0), "D_E02": (8.0, 3.0), "D_E03": (12.0, 4.0),
        "D_E04": (10.0, 3.0), "D_E05": (8.0, 3.0), "D_E06": (6.0, 2.0),
        "D_E07": (14.0, 4.0), "D_E08": (12.0, 4.0), "D_E09": (6.0, 2.0),
        "D_E10": (10.0, 3.0),
    }
    for gid, distribution in expected.items():
        group = simulator.all_groups[gid]
        assert (group.max_negotiation_turns_mean, group.max_negotiation_turns_std) == distribution


def test_patience_and_concessions_use_message_sequence(negotiation):
    conn, _, config, tools, customer_id, thread_id = negotiation
    assert tools.send_enterprise_deal([[customer_id, [["B", 999]]]]).success
    state = enterprise.get_negotiation_state(conn, thread_id)
    assert state.negotiation_turn == 1
    assert state.max_negotiation_turns == 6
    assert enterprise.get_negotiation_states_batch(conn, [thread_id])[thread_id] == state

    maximum = enterprise.compute_max_accepting_price(state, quality=1.0)
    initial = maximum * state.initial_offer_factor
    expected = round(maximum - (1 - config.enterprise_negotiation_alpha) * (maximum - initial), 2)
    assert enterprise.compute_customer_counter_offer(state, 1.0, 1, config) == expected

    enterprise.add_customer_message(conn, thread_id, 0, "Counter offer", 10)
    add_enterprise_turn(conn, thread_id, 0, "system", message_text="Status update")
    state = enterprise.get_negotiation_state(conn, thread_id)
    assert state.negotiation_turn == 3
    expected = round(maximum - (1 - config.enterprise_negotiation_alpha) ** 3 * (maximum - initial), 2)
    assert enterprise.compute_customer_counter_offer(state, 1.0, 1, config) == expected


@pytest.mark.parametrize("limit,offers,expected_status", [
    (2, (999, 20), "lost"), (3, (999, 20), "lost"),
    (6, (999, 999, 20), "subscribed"), (6, (999, 999, 999, 20), "lost"),
])
def test_message_limit_is_checked_before_offer_acceptance(negotiation, limit, offers, expected_status):
    conn, simulator, _, tools, customer_id, thread_id = negotiation
    conn.execute("UPDATE customers SET max_negotiation_turns = ? WHERE customer_id = ?", (limit, customer_id))
    for price in offers:
        assert tools.send_enterprise_deal([[customer_id, [["B", price]]]]).success
        simulator.current_day = conn.execute(
            "SELECT next_reply_day FROM enterprise_turns WHERE thread_id = ? ORDER BY message_id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()[0]
        simulator._process_scheduled_replies(simulator.get_current_config(), [])
        tools.set_current_day(simulator.current_day)

    assert conn.execute(
        "SELECT status FROM subscriptions WHERE customer_id = ?", (customer_id,),
    ).fetchone()[0] == expected_status


def test_submitting_offers_does_not_add_a_new_round_budget_gate(negotiation):
    conn, _, _, tools, customer_id, thread_id = negotiation
    conn.execute("UPDATE customers SET max_negotiation_turns = 2 WHERE customer_id = ?", (customer_id,))
    # 作者在客户回复时判断消息上限，工具提交本身没有新增的报价次数门槛。
    for _ in range(4):
        assert tools.send_enterprise_deal([[customer_id, [["B", 20]]]]).success
    assert enterprise.get_negotiation_state(conn, thread_id).negotiation_turn == 4


def test_reply_schedule_single_and_batch_use_same_message_identity(negotiation):
    conn, _, _, tools, customer_id, thread_id = negotiation
    assert tools.send_enterprise_deal([[customer_id, [["B", 20]]]]).success
    enterprise.schedule_customer_reply(conn, thread_id, current_day=7, seed=42)
    single = enterprise.get_negotiation_state(conn, thread_id).next_reply_day
    enterprise.batch_schedule_customer_replies(conn, [thread_id], current_day=7, seed=42)
    assert enterprise.get_negotiation_state(conn, thread_id).next_reply_day == single
