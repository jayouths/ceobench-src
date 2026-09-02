"""从模拟器状态记录实验私有事实，不计算论文派生指标。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class SegmentAcquisitionFact:
    """模拟器当日实际使用的客群获客计算量。"""

    market_capacity_multiplier: float
    calendar_cycle_multiplier: float
    macroeconomic_multiplier: float
    social_media_multiplier: float
    demand_surge_multiplier: float
    channel_leads_expected: float
    network_leads_expected: float
    total_leads_expected: float
    actual_leads: int


@dataclass(frozen=True, slots=True)
class PlanQualityFact:
    """某套餐当天实际使用的模型等级和质量倍率。"""

    model_tier: int
    tier_multiplier: float


def record_evaluation_day(
    conn: sqlite3.Connection,
    day: int,
    segment_acquisition: Mapping[str, SegmentAcquisitionFact],
    *,
    base_product_quality: float,
    quality_by_plan: Mapping[str, PlanQualityFact],
) -> None:
    """在每日经营动作完成后统一记录实验事实。"""
    record_subscription_day(conn, day)
    record_segment_day(conn, day, segment_acquisition)
    record_quality_day(
        conn,
        day,
        base_product_quality=base_product_quality,
        quality_by_plan=quality_by_plan,
    )


def record_subscription_day(conn: sqlite3.Connection, day: int) -> None:
    """记录日末有效订阅的账户、席位和 MRR 存量。"""
    # 与模拟器 get_mrr() 使用相同口径：企业席位取 customers 中当前生效值，
    # effective_price 为实际月费。无有效订阅的组合不写行，离线指标层结合
    # service_day 补零；流失率和增长率也只在离线阶段计算。
    conn.execute("DELETE FROM _eval_subscription_day WHERE day = ?", (day,))
    conn.execute(
        """
        INSERT INTO _eval_subscription_day (
            day, customer_type, group_id, plan,
            active_accounts, active_seats, mrr
        )
        SELECT
            ?,
            c.customer_type,
            c.group_id,
            s.plan,
            COUNT(*) AS active_accounts,
            SUM(
                CASE WHEN c.customer_type = 'large'
                     THEN CAST(c.seat_count AS INTEGER)
                     ELSE 1
                END
            ) AS active_seats,
            SUM(
                CASE WHEN c.customer_type = 'large'
                     THEN s.effective_price * CAST(c.seat_count AS INTEGER)
                     ELSE s.effective_price
                END
            ) AS mrr
        FROM subscriptions AS s
        JOIN customers AS c ON c.customer_id = s.customer_id
        WHERE s.status = 'subscribed'
          AND s.end_day IS NULL
          AND s.plan IN ('A', 'B', 'C')
        GROUP BY c.customer_type, c.group_id, s.plan
        """,
        (day,),
    )


def record_segment_day(
    conn: sqlite3.Connection,
    day: int,
    acquisition_by_group: Mapping[str, SegmentAcquisitionFact],
) -> None:
    """记录所有客群的日末状态，并合并当日真实获客计算过程。"""
    conn.execute("DELETE FROM _eval_segment_day WHERE day = ?", (day,))
    # group_parameters 覆盖模拟世界中的全部客群，因此没有客户的客群也不会消失。
    conn.execute(
        """
        INSERT INTO _eval_segment_day (
            day, group_id, info_level, reputation, awareness,
            group_quality_drift, group_budget_drift, global_quality_drift,
            satisfaction_sample_accounts, avg_satisfaction,
            min_satisfaction, max_satisfaction, avg_relationship
        )
        SELECT
            ?,
            gp.group_id,
            COALESCE(gil.info_level, 0),
            COALESCE(gr.reputation, 0.5),
            COALESCE(ga.awareness, 0.0),
            gp.drift_q_bias_total,
            gp.drift_c_max_total,
            gds.global_q_bias_total,
            COALESCE(health.sample_accounts, 0),
            health.avg_satisfaction,
            health.min_satisfaction,
            health.max_satisfaction,
            health.avg_relationship
        FROM group_parameters AS gp
        CROSS JOIN global_drift_state AS gds
        LEFT JOIN group_info_levels AS gil ON gil.group_id = gp.group_id
        LEFT JOIN group_reputation AS gr ON gr.group_id = gp.group_id
        LEFT JOIN group_awareness AS ga ON ga.group_id = gp.group_id
        LEFT JOIN (
            SELECT
                c.group_id,
                COUNT(*) AS sample_accounts,
                AVG(cs.satisfaction) AS avg_satisfaction,
                MIN(cs.satisfaction) AS min_satisfaction,
                MAX(cs.satisfaction) AS max_satisfaction,
                AVG(cs.relationship) AS avg_relationship
            FROM customer_state AS cs
            JOIN customers AS c ON c.customer_id = cs.customer_id
            JOIN subscriptions AS s ON s.customer_id = c.customer_id
            WHERE s.status = 'subscribed' AND s.end_day IS NULL
            GROUP BY c.group_id
        ) AS health ON health.group_id = gp.group_id
        """,
        (day,),
    )

    # 这些量是模拟器生成线索时真正使用的输入和结果，事后不能无损反推。
    conn.executemany(
        """
        UPDATE _eval_segment_day
        SET market_capacity_multiplier = ?,
            calendar_cycle_multiplier = ?,
            macroeconomic_multiplier = ?,
            social_media_multiplier = ?,
            demand_surge_multiplier = ?,
            channel_leads_expected = ?,
            network_leads_expected = ?,
            total_leads_expected = ?,
            actual_leads = ?
        WHERE day = ? AND group_id = ?
        """,
        [
            (
                fact.market_capacity_multiplier,
                fact.calendar_cycle_multiplier,
                fact.macroeconomic_multiplier,
                fact.social_media_multiplier,
                fact.demand_surge_multiplier,
                fact.channel_leads_expected,
                fact.network_leads_expected,
                fact.total_leads_expected,
                fact.actual_leads,
                day,
                group_id,
            )
            for group_id, fact in acquisition_by_group.items()
        ],
    )


def record_quality_day(
    conn: sqlite3.Connection,
    day: int,
    *,
    base_product_quality: float,
    quality_by_plan: Mapping[str, PlanQualityFact],
) -> None:
    """记录客群研发增益和套餐模型共同形成的日末交付质量。"""
    shared_row = conn.execute(
        "SELECT value FROM global_state WHERE key = 'q_shared_bonus'"
    ).fetchone()
    shared_bonus = float(shared_row[0]) if shared_row else 0.0
    group_bonuses = {
        row["key"][len("q_group_bonus_") :]: float(row["value"])
        for row in conn.execute(
            "SELECT key, value FROM global_state WHERE key LIKE 'q_group_bonus_%'"
        ).fetchall()
    }
    group_ids = [
        row["group_id"]
        for row in conn.execute(
            "SELECT group_id FROM group_parameters ORDER BY group_id"
        ).fetchall()
    ]

    conn.execute("DELETE FROM _eval_quality_day WHERE day = ?", (day,))
    rows = []
    for group_id in group_ids:
        group_bonus = group_bonuses.get(group_id, 0.0)
        for plan, quality in quality_by_plan.items():
            delivered_quality = (
                base_product_quality + shared_bonus + group_bonus
            ) * quality.tier_multiplier
            rows.append(
                (
                    day,
                    group_id,
                    plan,
                    base_product_quality,
                    shared_bonus,
                    group_bonus,
                    quality.model_tier,
                    quality.tier_multiplier,
                    delivered_quality,
                )
            )
    conn.executemany(
        """
        INSERT INTO _eval_quality_day (
            day, group_id, plan, base_product_quality,
            shared_quality_bonus, group_quality_bonus,
            model_tier, tier_multiplier, delivered_quality
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def record_channel_effectiveness_event(
    conn: sqlite3.Connection,
    day: int,
    effectiveness: Mapping[tuple[str, str], float],
) -> None:
    """记录渠道与客群组合在某次变化后的真实基础获客效率。"""
    conn.execute(
        "DELETE FROM _eval_channel_effectiveness_event WHERE day = ?",
        (day,),
    )
    conn.executemany(
        """
        INSERT INTO _eval_channel_effectiveness_event (
            day, channel_id, group_id, leads_per_1000_dollars
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (day, channel_id, group_id, value)
            for (channel_id, group_id), value in effectiveness.items()
        ],
    )
