"""主实验 Agent 动作契约与企业 Inbox 汇总。"""

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List


def get_thread_inbox_items(
    conn: sqlite3.Connection,
    day: int,
    week_start_day: int | None = None,
) -> List[str]:
    """Get inbox summary items for enterprise messages in one week."""

    if week_start_day is None:
        week_start_day = max(1, day - 6)

    items = []

    # 新线索规模使用创建谈判时的席位快照，不受后续漂移影响。
    new_threads = conn.execute(
        """
        SELECT COUNT(*) AS cnt,
               COALESCE(SUM(et.seat_count), 0) AS total_seats
        FROM enterprise_turns et
        WHERE et.turn_number = 0 AND et.day >= ? AND et.day <= ?
        """,
        (week_start_day, day),
    ).fetchone()
    if new_threads["cnt"] > 0:
        items.append(
            f"📨 {new_threads['cnt']} new enterprise leads this week "
            f"({new_threads['total_seats']:,} total seats)"
        )

    new_replies = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM enterprise_turns et
        WHERE et.day >= ? AND et.day <= ?
          AND et.turn_number > 0 AND et.sender = 'customer'
        """,
        (week_start_day, day),
    ).fetchone()
    if new_replies["cnt"] > 0:
        items.append(f"✉️ {new_replies['cnt']} new enterprise replies this week")

    awaiting = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM enterprise_turns et
        WHERE et.message_id = (
            SELECT MAX(et2.message_id) FROM enterprise_turns et2
            WHERE et2.thread_id = et.thread_id
        )
          AND et.closed = 0
          AND et._internal_status IS NULL
          AND et.sender = 'customer'
        """
    ).fetchone()
    if awaiting["cnt"] > 0:
        items.append(
            f"⏳ {awaiting['cnt']} enterprise threads awaiting your response"
        )

    return items


@dataclass
class Action:
    """An action in the environment = a tool call."""

    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
