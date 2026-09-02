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
