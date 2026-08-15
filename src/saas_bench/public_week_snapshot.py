"""供 Dashboard 与 Analysis 复用的公开周度经营快照。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .config import BenchmarkConfig, MODEL_TIERS
from .database import get_cash, get_config, get_discovered_groups
from .simulation import DayResult


@dataclass(frozen=True)
class CurrentBusinessState:
    cash: float
    individual_subscribers: int
    enterprise_subscribed_seats: int
    open_issues: int


@dataclass(frozen=True)
class WeeklyActivity:
    usage_units: int
    new_individual_leads: int
    new_enterprise_leads: int
    new_individual_subscribers: int
    new_enterprise_subscribed_seats: int
    cancellations: int
    upgrades: int
    downgrades: int
    peak_overload: float
    outage: bool
    downtime_minutes: int
    peak_p95_ms: float
    peak_error_rate: float


@dataclass(frozen=True)
class OperatingConfiguration:
    price_a: float
    price_b: float
    price_c: float
    tier_a: int
    tier_b: int
    tier_c: int
    quota_a: int
    quota_b: int
    quota_c: int
    capacity_tier: int
    daily_operations_spend: float
    daily_development_spend: float


@dataclass(frozen=True)
class GroupDeliveredQuality:
    group: str
    plan_a: float
    plan_b: float
    plan_c: float
    group_bonus: float


@dataclass(frozen=True)
class DeliveredQuality:
    base_quality: float
    global_bonus: float
    groups: list[GroupDeliveredQuality] = field(default_factory=list)


@dataclass(frozen=True)
class SocialPostSummary:
    post_id: int
    views: int
    comment_count: int
    comment_post_ids: list[int]
    content_preview: str
    content_truncated: bool


@dataclass(frozen=True)
class PublicWeekSnapshot:
    """某一周结束时，Baseline 已有权限看到的结构化经营事实。"""

    day: int
    week: int
    current_state: CurrentBusinessState
    weekly_activity: WeeklyActivity | None
    configuration: OperatingConfiguration
    delivered_quality: DeliveredQuality
    social_posts: list[SocialPostSummary] = field(default_factory=list)
    weekly_calculations: dict[str, str] = field(default_factory=dict)
    inbox_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_subscriber_counts(
    conn: sqlite3.Connection,
    day_result: DayResult | None,
) -> tuple[int, int]:
    if day_result is not None:
        return (
            day_result.total_individual_subscribers,
            day_result.total_enterprise_subscription_seats,
        )

    # 服务重启后可能没有内存中的 DayResult，此时只恢复可由当前数据库确认的存量。
    individual_subscribers = conn.execute("""
        SELECT COUNT(*) FROM subscriptions s
        JOIN customers c ON s.customer_id = c.customer_id
        WHERE s.status = 'subscribed' AND s.end_day IS NULL
          AND c.customer_type = 'small'
    """).fetchone()[0]
    enterprise_seats = conn.execute("""
        SELECT COALESCE(SUM(CAST(c.seat_count AS INTEGER)), 0) FROM subscriptions s
        JOIN customers c ON s.customer_id = c.customer_id
        WHERE s.status = 'subscribed' AND s.end_day IS NULL
          AND c.customer_type = 'large'
    """).fetchone()[0]
    return individual_subscribers, enterprise_seats


def _build_weekly_activity(day_result: DayResult | None) -> WeeklyActivity | None:
    if day_result is None:
        return None
    return WeeklyActivity(
        usage_units=day_result.total_usage,
        new_individual_leads=day_result.new_individual_leads,
        new_enterprise_leads=day_result.new_enterprise_leads,
        new_individual_subscribers=day_result.new_individual_subscribers,
        new_enterprise_subscribed_seats=day_result.new_enterprise_subscribers_seats,
        cancellations=day_result.cancellations,
        upgrades=day_result.upgrades,
        downgrades=day_result.downgrades,
        peak_overload=day_result.overload,
        outage=day_result.outage,
        downtime_minutes=day_result.downtime_minutes,
        peak_p95_ms=day_result.p95_ms,
        peak_error_rate=day_result.error_rate,
    )


def _build_configuration(config: Mapping[str, Any]) -> OperatingConfiguration:
    return OperatingConfiguration(
        price_a=config.get("price_A", 0),
        price_b=config.get("price_B", 0),
        price_c=config.get("price_C", 0),
        tier_a=config.get("tier_A", 1),
        tier_b=config.get("tier_B", 2),
        tier_c=config.get("tier_C", 3),
        quota_a=config.get("quota_A", 100),
        quota_b=config.get("quota_B", 500),
        quota_c=config.get("quota_C", 2000),
        capacity_tier=config.get("capacity_tier", 0),
        daily_operations_spend=config.get("spend_operations", 0),
        daily_development_spend=config.get("spend_development", 0),
    )


def _build_delivered_quality(
    conn: sqlite3.Connection,
    config: OperatingConfiguration,
) -> DeliveredQuality:
    global_bonus_row = conn.execute(
        "SELECT value FROM global_state WHERE key = 'q_shared_bonus'"
    ).fetchone()
    global_bonus = float(global_bonus_row["value"]) if global_bonus_row else 0.0

    group_bonuses = {}
    rows = conn.execute(
        "SELECT key, value FROM global_state WHERE key LIKE 'q_group_bonus_%'"
    ).fetchall()
    for row in rows:
        group = row["key"][len("q_group_bonus_"):]
        group_bonuses[group] = float(row["value"])

    base_quality = BenchmarkConfig.base_product_quality
    multipliers = (
        MODEL_TIERS[config.tier_a].quality_multiplier,
        MODEL_TIERS[config.tier_b].quality_multiplier,
        MODEL_TIERS[config.tier_c].quality_multiplier,
    )
    groups = []
    for group in sorted(get_discovered_groups(conn)):
        group_bonus = group_bonuses.get(group, 0.0)
        effective_base = base_quality + global_bonus + group_bonus
        groups.append(GroupDeliveredQuality(
            group=group,
            plan_a=effective_base * multipliers[0],
            plan_b=effective_base * multipliers[1],
            plan_c=effective_base * multipliers[2],
            group_bonus=group_bonus,
        ))
    return DeliveredQuality(
        base_quality=base_quality,
        global_bonus=global_bonus,
        groups=groups,
    )


def _build_social_posts(conn: sqlite3.Connection, day: int) -> list[SocialPostSummary]:
    if day <= 7:
        return []

    rows = conn.execute("""
        SELECT asp.agent_post_id, asp.content, asp.views, asp.comment_post_ids,
               COUNT(smp.post_id) AS comment_count
        FROM agent_social_media_posts asp
        LEFT JOIN social_media_posts smp
          ON smp.reply_to_agent_post_id = asp.agent_post_id
        WHERE asp.day > ? AND asp.day <= ?
        GROUP BY asp.agent_post_id
    """, (day - 7, day)).fetchall()

    posts = []
    for row in rows:
        try:
            comment_post_ids = json.loads(row["comment_post_ids"] or "[]")
            if not isinstance(comment_post_ids, list):
                comment_post_ids = []
        except (json.JSONDecodeError, TypeError):
            comment_post_ids = []
        content = row["content"]
        posts.append(SocialPostSummary(
            post_id=row["agent_post_id"],
            views=row["views"],
            comment_count=row["comment_count"],
            comment_post_ids=comment_post_ids,
            content_preview=content[:80],
            content_truncated=len(content) > 80,
        ))
    return posts


def build_public_week_snapshot(
    conn: sqlite3.Connection,
    day: int,
    day_result: DayResult | None = None,
    calc_outputs: Mapping[str, str] | None = None,
    inbox_items: Sequence[str] | None = None,
) -> PublicWeekSnapshot:
    """只组装公开经营事实，不负责文本展示或 Analysis 推理。"""

    config = _build_configuration(get_config(conn, day) or {})
    individual_subscribers, enterprise_seats = _get_subscriber_counts(
        conn, day_result
    )
    open_issues = conn.execute("""
        SELECT COUNT(*) FROM customer_state cs
        JOIN subscriptions s ON cs.customer_id = s.customer_id
        WHERE s.status = 'subscribed' AND s.end_day IS NULL
          AND cs.open_issue_days > 0
    """).fetchone()[0]

    # 这里仅暴露原 Dashboard 已公开的汇总，不把隐藏客户状态交给 Analysis。
    return PublicWeekSnapshot(
        day=day,
        week=(day + 6) // 7,
        current_state=CurrentBusinessState(
            cash=get_cash(conn),
            individual_subscribers=individual_subscribers,
            enterprise_subscribed_seats=enterprise_seats,
            open_issues=open_issues,
        ),
        weekly_activity=_build_weekly_activity(day_result),
        configuration=config,
        delivered_quality=_build_delivered_quality(conn, config),
        social_posts=_build_social_posts(conn, day),
        weekly_calculations={
            name: output[:500] for name, output in (calc_outputs or {}).items()
        },
        inbox_items=list(inbox_items or []),
    )


def render_weekly_dashboard(snapshot: PublicWeekSnapshot) -> str:
    """确定性地渲染快照；Dashboard 不再自行查询或计算指标。"""

    state = snapshot.current_state
    lines = [
        f"=== Week {snapshot.week} Dashboard (Day {snapshot.day}) ===",
        "",
        f"Cash: ${state.cash:,.0f}",
        f"Individual Subscribers: {state.individual_subscribers}",
        f"Enterprise Subscribed Seats: {state.enterprise_subscribed_seats}",
        f"Open Issues: {state.open_issues}",
    ]

    activity = snapshot.weekly_activity
    if activity is not None:
        lines.extend([
            "",
            "--- This Week's Metrics ---",
            f"Usage: {activity.usage_units:,} units",
            f"New Individual Leads: {activity.new_individual_leads} | New Enterprise Leads: {activity.new_enterprise_leads}",
            f"New Individual Subscribers: {activity.new_individual_subscribers} | New Enterprise Subscribed Seats: {activity.new_enterprise_subscribed_seats}",
            f"Cancellations: {activity.cancellations}",
            f"Upgrades: {activity.upgrades} | Downgrades: {activity.downgrades}",
            f"Overload (peak): {activity.peak_overload:.1%}" if activity.peak_overload > 0 else "Overload: None",
            f"Outage: {'YES (' + str(activity.downtime_minutes) + ' min total)' if activity.outage else 'No'}",
            f"P95 Latency (peak): {activity.peak_p95_ms:.0f}ms | Error Rate (peak): {activity.peak_error_rate:.2%}",
        ])

    config = snapshot.configuration
    lines.extend([
        "",
        "--- Current Config ---",
        f"Prices: A=${config.price_a:.0f}, B=${config.price_b:.0f}, C=${config.price_c:.0f}",
        f"Model Tiers: A={config.tier_a}, B={config.tier_b}, C={config.tier_c}",
        f"Quotas: A={config.quota_a}, B={config.quota_b}, C={config.quota_c} units/day",
        f"Capacity: Tier {config.capacity_tier}",
        f"Daily Spend: Ops=${config.daily_operations_spend:.0f}, Dev=${config.daily_development_spend:.0f} (ad spend is per (channel, group) — see set_targeted_ad_spend)",
    ])

    quality = snapshot.delivered_quality
    lines.extend([
        "",
        f"--- Delivered Quality (base={quality.base_quality:.2f}, global_bonus={quality.global_bonus:.4f}) ---",
        f"{'Group':<8} {'Plan A (T'+str(config.tier_a)+')':<14} {'Plan B (T'+str(config.tier_b)+')':<14} {'Plan C (T'+str(config.tier_c)+')':<14} {'Grp Bonus':<10}",
    ])
    for group in quality.groups:
        group_bonus = f"+{group.group_bonus:.4f}" if group.group_bonus > 0 else "0"
        lines.append(
            f"{group.group:<8} {group.plan_a:<14.4f} {group.plan_b:<14.4f} "
            f"{group.plan_c:<14.4f} {group_bonus:<10}"
        )

    if snapshot.social_posts:
        lines.extend(["", "--- Your Social Media Posts (This Week) ---"])
        for post in snapshot.social_posts:
            comment_ids = (
                f" (comment post_ids: {post.comment_post_ids})"
                if post.comment_post_ids else ""
            )
            suffix = "..." if post.content_truncated else ""
            lines.append(
                f"  Post #{post.post_id}: {post.views} views, "
                f"{post.comment_count} comments{comment_ids} — "
                f'"{post.content_preview}{suffix}"'
            )

    if snapshot.weekly_calculations:
        lines.extend(["", "--- Weekly Calculations ---"])
        for name, output in snapshot.weekly_calculations.items():
            lines.extend([f"[{name}]", output])

    lines.extend(["", "--- Inbox ---"])
    if snapshot.inbox_items:
        lines.extend(f"  • {item}" for item in snapshot.inbox_items)
    else:
        lines.append("  (No new messages)")
    return "\n".join(lines)
