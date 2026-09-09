"""公开周度经营快照及 Dashboard 渲染测试。"""

from saas_bench.simulator.public_week_snapshot import (
    build_public_week_snapshot,
    load_public_week_snapshot,
    render_weekly_dashboard,
    save_public_week_snapshot,
)
from saas_bench.simulator.simulation import DayResult


def test_initial_dashboard_is_rendered_from_public_snapshot(make_initialized_sim):
    conn, _, _ = make_initialized_sim()

    snapshot = build_public_week_snapshot(conn, day=0)
    dashboard = render_weekly_dashboard(snapshot)

    assert snapshot.to_dict()["current_state"] == {
        "cash": 1_000_000.0,
        "individual_subscribers": 0,
        "enterprise_subscribed_accounts": 0,
        "enterprise_subscribed_seats": 0,
        "open_issues": 0,
    }
    assert snapshot.weekly_activity is None
    assert dashboard == """=== Week 0 Dashboard (Day 0) ===

Cash: $1,000,000
Individual Subscribers: 0
Enterprise Subscribed Accounts: 0
Enterprise Subscribed Seats: 0
Open Issues: 0

--- Current Config ---
Prices: A=$0, B=$0, C=$0
Model Tiers: A=1, B=1, C=1
Quotas: A=0, B=0, C=0 units/day
Capacity: Tier 0
Daily Spend: Ops=$0, Dev=$0 (ad spend is per (channel, group) — see set_targeted_ad_spend)

--- Delivered Quality (base=0.20, global_bonus=0.0000) ---
Group    Plan A (T1)    Plan B (T1)    Plan C (T1)    Grp Bonus 
E1       0.1200         0.1200         0.1200         0         
E2       0.1200         0.1200         0.1200         0         
E3       0.1200         0.1200         0.1200         0         
S1       0.1200         0.1200         0.1200         0         
S2       0.1200         0.1200         0.1200         0         
S3       0.1200         0.1200         0.1200         0         

--- Inbox ---
  (No new messages)"""


def test_weekly_activity_has_one_structured_source(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    result = DayResult(
        day=7,
        total_usage=1_080,
        overload=0.125,
        outage=True,
        downtime_minutes=15,
        p95_ms=284.4,
        error_rate=0.004,
        new_subscribers=5,
        new_leads=299,
        individual_cancellations=2,
        enterprise_cancellations=1,
        enterprise_cancelled_seats=12,
        individual_upgrades=1,
        enterprise_upgrades=2,
        individual_downgrades=3,
        enterprise_downgrades=4,
        payments_received=0,
        total_costs=0,
        cash=1_000_000,
        mrr=0,
        new_individual_leads=296,
        new_enterprise_leads=3,
        new_individual_subscribers=5,
        new_enterprise_subscriber_accounts=1,
        new_enterprise_subscribers_seats=20,
        total_individual_subscribers=5,
        total_enterprise_subscribers=1,
        total_enterprise_subscription_seats=20,
    )

    snapshot = build_public_week_snapshot(
        conn,
        day=7,
        day_result=result,
        inbox_items=["3 new enterprise leads"],
    )
    dashboard = render_weekly_dashboard(snapshot)

    assert snapshot.weekly_activity is not None
    assert snapshot.weekly_activity.cancellations.enterprise_accounts == 1
    assert snapshot.weekly_activity.cancellations.enterprise_seats == 12
    assert snapshot.weekly_activity.new_enterprise_subscribed_seats == 20
    assert snapshot.inbox_items == ["3 new enterprise leads"]
    assert "Usage: 1,080 units" in dashboard
    assert "New Individual Leads: 296 | New Enterprise Leads: 3" in dashboard
    assert "New Individual Subscribers: 5 | New Enterprise Subscribers: 1 accounts / 20 seats" in dashboard
    assert "Cancellations: Individual=2 | Enterprise=1 accounts / 12 seats" in dashboard
    assert "Plan Changes: Individual Up=1, Down=3 | Enterprise Up=2, Down=4" in dashboard
    assert "Overload (peak): 12.5%" in dashboard
    assert "Outage: YES (15 min total)" in dashboard
    assert "  • 3 new enterprise leads" in dashboard


def test_public_week_snapshot_round_trips_complete_week(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    result = DayResult(
        day=7,
        total_usage=1_080,
        overload=0.125,
        outage=True,
        downtime_minutes=15,
        p95_ms=284.4,
        error_rate=0.004,
        new_subscribers=5,
        new_leads=299,
        individual_cancellations=2,
        enterprise_cancellations=1,
        enterprise_cancelled_seats=12,
        individual_upgrades=1,
        enterprise_upgrades=2,
        individual_downgrades=3,
        enterprise_downgrades=4,
        payments_received=0,
        total_costs=0,
        cash=1_000_000,
        mrr=0,
        new_individual_leads=296,
        new_enterprise_leads=3,
        new_individual_subscribers=5,
        new_enterprise_subscriber_accounts=1,
        new_enterprise_subscribers_seats=20,
        total_individual_subscribers=5,
        total_enterprise_subscribers=1,
        total_enterprise_subscription_seats=20,
    )
    snapshot = build_public_week_snapshot(
        conn,
        day=7,
        day_result=result,
        inbox_items=["研发项目已完成"],
    )

    save_public_week_snapshot(conn, snapshot)
    conn.commit()

    restored = load_public_week_snapshot(conn)
    assert restored == snapshot
    assert restored.weekly_activity == snapshot.weekly_activity
    assert restored.inbox_items == ["研发项目已完成"]
