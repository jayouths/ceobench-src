"""订阅生命周期活动的统一汇总。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubscriptionActivity:
    """某个事件区间内的订阅取消和套餐变更。"""

    individual_cancellations: int = 0
    enterprise_cancellations: int = 0
    enterprise_cancelled_seats: int = 0
    individual_upgrades: int = 0
    enterprise_upgrades: int = 0
    individual_downgrades: int = 0
    enterprise_downgrades: int = 0


def latest_subscription_activity_event_id(conn: sqlite3.Connection) -> int:
    """返回当前最新的订阅活动事件 ID。"""

    row = conn.execute(
        "SELECT COALESCE(MAX(event_id), 0) FROM _subscription_activity_event"
    ).fetchone()
    return int(row[0])


def subscription_activity_after(
    conn: sqlite3.Connection,
    event_id: int,
) -> SubscriptionActivity:
    """汇总指定事件之后的订阅活动，不移动周报游标。"""

    rows = conn.execute(
        """
        SELECT customer_type, event_type,
               COUNT(*) AS accounts,
               COALESCE(SUM(seats), 0) AS seats
        FROM _subscription_activity_event
        WHERE event_id > ?
        GROUP BY customer_type, event_type
        """,
        (event_id,),
    ).fetchall()
    values = {
        (str(row["customer_type"]), str(row["event_type"])): row
        for row in rows
    }

    def accounts(customer_type: str, event_type: str) -> int:
        row = values.get((customer_type, event_type))
        return int(row["accounts"]) if row else 0

    cancelled_enterprise = values.get(("large", "cancelled"))
    return SubscriptionActivity(
        individual_cancellations=accounts("small", "cancelled"),
        enterprise_cancellations=accounts("large", "cancelled"),
        enterprise_cancelled_seats=(
            int(cancelled_enterprise["seats"]) if cancelled_enterprise else 0
        ),
        individual_upgrades=accounts("small", "upgraded"),
        enterprise_upgrades=accounts("large", "upgraded"),
        individual_downgrades=accounts("small", "downgraded"),
        enterprise_downgrades=accounts("large", "downgraded"),
    )


def consume_weekly_subscription_activity(
    conn: sqlite3.Connection,
) -> SubscriptionActivity:
    """读取上次周报后的全部事件，并持久化新游标。"""

    cursor = conn.execute(
        "SELECT last_event_id FROM _subscription_activity_cursor WHERE id = 1"
    ).fetchone()
    last_event_id = int(cursor["last_event_id"])
    activity = subscription_activity_after(conn, last_event_id)
    latest_event_id = latest_subscription_activity_event_id(conn)
    conn.execute(
        "UPDATE _subscription_activity_cursor SET last_event_id = ? WHERE id = 1",
        (latest_event_id,),
    )
    return activity
