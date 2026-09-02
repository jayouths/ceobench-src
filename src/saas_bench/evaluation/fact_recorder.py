"""从模拟器状态记录实验私有事实，不计算论文派生指标。"""

from __future__ import annotations

import sqlite3


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
