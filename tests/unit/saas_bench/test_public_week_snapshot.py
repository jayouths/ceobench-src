"""公开周度经营快照及 Dashboard 渲染测试。"""

from saas_bench.public_week_snapshot import (
    build_public_week_snapshot,
    render_weekly_dashboard,
)
from saas_bench.simulation import DayResult


def test_initial_dashboard_is_rendered_from_public_snapshot(make_initialized_sim):
    conn, _, _ = make_initialized_sim()

    snapshot = build_public_week_snapshot(conn, day=0)
    dashboard = render_weekly_dashboard(snapshot)

    assert snapshot.to_dict()["current_state"] == {
        "cash": 1_000_000.0,
        "individual_subscribers": 0,
        "enterprise_subscribed_seats": 0,
        "open_issues": 0,
    }
    assert snapshot.weekly_activity is None
    assert dashboard == """=== Week 0 Dashboard (Day 0) ===

Cash: $1,000,000
Individual Subscribers: 0
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
        cancellations=2,
        upgrades=1,
        downgrades=3,
        payments_received=0,
        total_costs=0,
        cash=1_000_000,
        mrr=0,
        new_individual_leads=296,
        new_enterprise_leads=3,
        new_individual_subscribers=5,
        new_enterprise_subscribers_seats=20,
        total_individual_subscribers=5,
        total_enterprise_subscription_seats=20,
    )

    snapshot = build_public_week_snapshot(
        conn,
        day=7,
        day_result=result,
        calc_outputs={"forecast": "x" * 600},
        inbox_items=["3 new enterprise leads"],
    )
    dashboard = render_weekly_dashboard(snapshot)

    assert snapshot.weekly_activity is not None
    assert snapshot.weekly_activity.new_enterprise_subscribed_seats == 20
    assert snapshot.weekly_calculations == {"forecast": "x" * 500}
    assert snapshot.inbox_items == ["3 new enterprise leads"]
    assert "Usage: 1,080 units" in dashboard
    assert "New Individual Leads: 296 | New Enterprise Leads: 3" in dashboard
    assert "New Individual Subscribers: 5 | New Enterprise Subscribed Seats: 20" in dashboard
    assert "Overload (peak): 12.5%" in dashboard
    assert "Outage: YES (15 min total)" in dashboard
    assert "[forecast]\n" + "x" * 500 in dashboard
    assert "  • 3 new enterprise leads" in dashboard
