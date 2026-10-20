"""实验私有事实表及其无侵入采集触发器。"""

from __future__ import annotations

import sqlite3


def initialize_evaluation_schema(conn: sqlite3.Connection) -> None:
    """初始化评测事实结构，不向模拟器经营规则增加约束。"""
    conn.executescript(
        """
        -- 日末订阅存量。只保存有有效订阅的组合；缺行表示零而不是数据缺失。
        -- 本表不保存增长率、流失率或集中度等派生指标。
        CREATE TABLE IF NOT EXISTS _eval_subscription_day (
            day INTEGER NOT NULL,
            customer_type TEXT NOT NULL CHECK(customer_type IN ('small', 'large')),
            group_id TEXT NOT NULL,
            plan TEXT NOT NULL CHECK(plan IN ('A', 'B', 'C')),
            active_accounts INTEGER NOT NULL,
            active_seats INTEGER NOT NULL,
            mrr REAL NOT NULL,
            PRIMARY KEY (day, customer_type, group_id, plan)
        );

        -- 真实订阅生命周期事件。事件与订阅状态在同一事务中写入，
        -- 避免根据后续被覆盖的 subscriptions 当前状态反推历史。
        CREATE TABLE IF NOT EXISTS _eval_subscription_event (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            day INTEGER NOT NULL,
            subscription_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            customer_type TEXT NOT NULL CHECK(customer_type IN ('small', 'large')),
            group_id TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN ('started', 'ended')),
            plan TEXT NOT NULL CHECK(plan IN ('A', 'B', 'C')),
            seats INTEGER NOT NULL,
            mrr REAL NOT NULL,
            reason TEXT
        );

        -- 客群日末状态和当日获客过程。每个已定义客群每天保留一行；尚无客户
        -- 时满意度为空，尚未发现时获客过程为空。新增和流失不在此重复记录。
        CREATE TABLE IF NOT EXISTS _eval_segment_day (
            day INTEGER NOT NULL,
            group_id TEXT NOT NULL,
            info_level INTEGER NOT NULL CHECK(info_level BETWEEN 0 AND 5),
            reputation REAL NOT NULL,
            awareness REAL NOT NULL,
            group_quality_drift REAL NOT NULL,
            group_budget_drift REAL NOT NULL,
            global_quality_drift REAL NOT NULL,
            satisfaction_sample_accounts INTEGER NOT NULL,
            avg_satisfaction REAL,
            min_satisfaction REAL,
            max_satisfaction REAL,
            avg_relationship REAL,
            market_capacity_multiplier REAL,
            calendar_cycle_multiplier REAL,
            macroeconomic_multiplier REAL,
            social_media_multiplier REAL,
            demand_surge_multiplier REAL,
            channel_leads_expected REAL,
            network_leads_expected REAL,
            total_leads_expected REAL,
            actual_leads INTEGER,
            PRIMARY KEY (day, group_id)
        );

        -- 各客群、套餐在日末实际形成的产品交付质量。只记录质量组成，
        -- 不混入关系、工单、广告和配额等客户级感知修正。
        CREATE TABLE IF NOT EXISTS _eval_quality_day (
            day INTEGER NOT NULL,
            group_id TEXT NOT NULL,
            plan TEXT NOT NULL CHECK(plan IN ('A', 'B', 'C')),
            base_product_quality REAL NOT NULL,
            shared_quality_bonus REAL NOT NULL,
            group_quality_bonus REAL NOT NULL,
            model_tier INTEGER NOT NULL,
            tier_multiplier REAL NOT NULL,
            delivered_quality REAL NOT NULL,
            PRIMARY KEY (day, group_id, plan)
        );

        -- 渠道获客效率仅在初始日和每次月度随机变化后记录一个状态点。
        CREATE TABLE IF NOT EXISTS _eval_channel_effectiveness_event (
            day INTEGER NOT NULL,
            channel_id TEXT NOT NULL,
            group_id TEXT NOT NULL,
            leads_per_1000_dollars REAL NOT NULL,
            PRIMARY KEY (day, channel_id, group_id)
        );

        CREATE INDEX IF NOT EXISTS idx_eval_subscription_event_day
            ON _eval_subscription_event(day);
        CREATE INDEX IF NOT EXISTS idx_eval_subscription_event_subscription
            ON _eval_subscription_event(subscription_id);

        -- 直接创建为有效订阅时记录开始事件。MAX(service_day.day) 是当前已进入的
        -- 模拟日；初始化阶段尚无 service_day 时退回订阅自身的 start_day。
        CREATE TRIGGER IF NOT EXISTS _eval_subscription_started_after_insert
        AFTER INSERT ON subscriptions
        WHEN NEW.status = 'subscribed' AND NEW.end_day IS NULL
        BEGIN
            INSERT INTO _eval_subscription_event (
                day, subscription_id, customer_id, customer_type, group_id,
                event_type, plan, seats, mrr, reason
            )
            SELECT
                COALESCE((SELECT MAX(day) FROM service_day), NEW.start_day),
                NEW.subscription_id,
                NEW.customer_id,
                c.customer_type,
                c.group_id,
                'started',
                NEW.plan,
                CASE WHEN c.customer_type = 'large'
                     THEN CAST(c.seat_count AS INTEGER) ELSE 1 END,
                CASE WHEN c.customer_type = 'large'
                     THEN NEW.effective_price * CAST(c.seat_count AS INTEGER)
                     ELSE NEW.effective_price END,
                NULL
            FROM customers AS c
            WHERE c.customer_id = NEW.customer_id;
        END;

        -- 企业线索成交时通过 UPDATE 转为有效订阅，需要单独覆盖该状态迁移。
        CREATE TRIGGER IF NOT EXISTS _eval_subscription_started_after_update
        AFTER UPDATE OF status ON subscriptions
        WHEN OLD.status <> 'subscribed'
         AND NEW.status = 'subscribed'
         AND NEW.end_day IS NULL
        BEGIN
            INSERT INTO _eval_subscription_event (
                day, subscription_id, customer_id, customer_type, group_id,
                event_type, plan, seats, mrr, reason
            )
            SELECT
                COALESCE((SELECT MAX(day) FROM service_day), NEW.start_day),
                NEW.subscription_id,
                NEW.customer_id,
                c.customer_type,
                c.group_id,
                'started',
                NEW.plan,
                CASE WHEN c.customer_type = 'large'
                     THEN CAST(c.seat_count AS INTEGER) ELSE 1 END,
                CASE WHEN c.customer_type = 'large'
                     THEN NEW.effective_price * CAST(c.seat_count AS INTEGER)
                     ELSE NEW.effective_price END,
                NULL
            FROM customers AS c
            WHERE c.customer_id = NEW.customer_id;
        END;

        -- 以真实状态迁移发生日记录结束事件，不使用可能指向未来合同到期日的 end_day。
        CREATE TRIGGER IF NOT EXISTS _eval_subscription_ended_after_update
        AFTER UPDATE OF status ON subscriptions
        WHEN OLD.status = 'subscribed' AND NEW.status <> 'subscribed'
        BEGIN
            INSERT INTO _eval_subscription_event (
                day, subscription_id, customer_id, customer_type, group_id,
                event_type, plan, seats, mrr, reason
            )
            SELECT
                COALESCE((SELECT MAX(day) FROM service_day), NEW.end_day, NEW.start_day),
                NEW.subscription_id,
                NEW.customer_id,
                c.customer_type,
                c.group_id,
                'ended',
                OLD.plan,
                CASE WHEN c.customer_type = 'large'
                     THEN CAST(c.seat_count AS INTEGER) ELSE 1 END,
                CASE WHEN c.customer_type = 'large'
                     THEN OLD.effective_price * CAST(c.seat_count AS INTEGER)
                     ELSE OLD.effective_price END,
                NEW.churn_reason
            FROM customers AS c
            WHERE c.customer_id = NEW.customer_id;
        END;
        """
    )
