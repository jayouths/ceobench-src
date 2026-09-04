"""Analysis 确定性信号使用的公开只读 SQL。"""

LATEST_MACRO_CONDITION = """
    SELECT day, pmi_value, pmi_trend, pmi_change, cycle_phase, description
    FROM macroeconomic_conditions
    ORDER BY day DESC
    LIMIT 1
"""

LEDGER_MAX_ID = "SELECT COALESCE(MAX(id), 0) AS ledger_max_id FROM ledger"

RESEARCH_PROJECTS = """
    SELECT project_id, tier, status, started_day, expected_completion_day,
           expected_quality_boost, quality_boost_applied
    FROM research_projects
    ORDER BY started_day, project_id
"""

CUSTOMER_BASE = """
    SELECT s.plan, c.customer_type,
           COUNT(DISTINCT c.customer_id) AS accounts,
           SUM(CASE WHEN c.customer_type = 'large' THEN s.seat_count ELSE 0 END) AS seats
    FROM subscriptions s
    JOIN customers c ON c.customer_id = s.customer_id
    WHERE s.status = 'subscribed' AND s.end_day IS NULL
    GROUP BY s.plan, c.customer_type
    ORDER BY s.plan, c.customer_type
"""


def effective_leads(start: int, end: int) -> str:
    """个人以开始订阅为有效线索，企业以正式进入 new_lead 线程为准。"""

    return f"""
        SELECT s.start_day AS day, c.group_id,
               'individual' AS customer_type,
               COALESCE(c.acquisition_source, 'organic') AS acquisition_source,
               COUNT(DISTINCT c.customer_id) AS lead_count
        FROM subscriptions s
        JOIN customers c ON c.customer_id = s.customer_id
        WHERE c.customer_type = 'small' AND s.start_day BETWEEN {start} AND {end}
        GROUP BY s.start_day, c.group_id, c.acquisition_source
        UNION ALL
        SELECT et.day AS day, c.group_id,
               'enterprise' AS customer_type,
               COALESCE(c.acquisition_source, 'organic') AS acquisition_source,
               COUNT(*) AS lead_count
        FROM enterprise_turns et
        JOIN customers c ON c.customer_id = et.customer_id
        WHERE et.thread_type = 'new_lead' AND et.turn_number = 0
          AND et.day BETWEEN {start} AND {end}
        GROUP BY et.day, c.group_id, c.acquisition_source
        ORDER BY day, group_id, acquisition_source
    """


def ad_channels(start: int, end: int) -> str:
    return f"""
        SELECT day, channel_id, group_id,
               SUM(leads_generated) AS raw_leads,
               SUM(spend) AS spend
        FROM ad_channel_leads
        WHERE day BETWEEN {start} AND {end}
        GROUP BY day, channel_id, group_id
        ORDER BY day, channel_id, group_id
    """


def social_posts(start: int, end: int) -> str:
    return f"""
        SELECT post_id, day, content
        FROM social_media_posts
        WHERE day BETWEEN {start} AND {end}
        ORDER BY day, post_id
    """


def ledger_after(previous_max_id: int) -> str:
    """按上周记录的递增 ID 划分流水，避免遗漏周边界日即时支出。"""

    return f"""
        SELECT category, SUM(amount) AS amount
        FROM ledger
        WHERE id > {previous_max_id}
          AND category != 'initial_funding'
        GROUP BY category
        ORDER BY category
    """


def service_days(start: int, end: int) -> str:
    return f"""
        SELECT day, total_usage_units, p95_ms, error_rate,
               downtime_minutes, capacity_units
        FROM service_day
        WHERE day BETWEEN {start} AND {end}
        ORDER BY day
    """


def paid_subscriptions(start: int, end: int) -> str:
    return f"""
        SELECT s.start_day AS day, c.customer_type,
               COUNT(DISTINCT c.customer_id) AS accounts,
               SUM(CASE WHEN c.customer_type = 'large' THEN s.seat_count ELSE 0 END) AS seats
        FROM subscriptions s
        JOIN customers c ON c.customer_id = s.customer_id
        WHERE s.status IN ('subscribed', 'cancelled')
          AND s.start_day BETWEEN {start} AND {end}
        GROUP BY s.start_day, c.customer_type
        ORDER BY s.start_day, c.customer_type
    """


def issue_summary(
    current_start: int,
    current_end: int,
    previous_start: int,
    previous_end: int,
) -> str:
    return f"""
        SELECT
          COALESCE(SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END), 0) AS open_count,
          AVG(CASE WHEN status = 'open' THEN days_open END) AS avg_open_age,
          MAX(CASE WHEN status = 'open' THEN days_open END) AS max_open_age,
          COALESCE(SUM(CASE WHEN status = 'open' AND days_open > 7 THEN 1 ELSE 0 END), 0) AS over_7,
          COALESCE(SUM(CASE WHEN status = 'open' AND days_open > 14 THEN 1 ELSE 0 END), 0) AS over_14,
          COALESCE(SUM(CASE WHEN open_day BETWEEN {current_start} AND {current_end} THEN 1 ELSE 0 END), 0) AS current_opened,
          COALESCE(SUM(CASE WHEN open_day BETWEEN {previous_start} AND {previous_end} THEN 1 ELSE 0 END), 0) AS previous_opened,
          COALESCE(SUM(CASE WHEN resolution_type = 'ops_resolved' AND resolved_day BETWEEN {current_start} AND {current_end} THEN 1 ELSE 0 END), 0) AS current_resolved,
          COALESCE(SUM(CASE WHEN resolution_type = 'ops_resolved' AND resolved_day BETWEEN {previous_start} AND {previous_end} THEN 1 ELSE 0 END), 0) AS previous_resolved,
          AVG(CASE WHEN resolution_type = 'ops_resolved' AND resolved_day BETWEEN {current_start} AND {current_end} THEN resolved_day - open_day END) AS current_resolution_days,
          AVG(CASE WHEN resolution_type = 'ops_resolved' AND resolved_day BETWEEN {previous_start} AND {previous_end} THEN resolved_day - open_day END) AS previous_resolution_days
        FROM issues
    """


def _active_thread_cte() -> str:
    """仅保留每个线程的最新消息，并用公开订阅状态排除失效线程。"""

    return """
        WITH latest AS (
          SELECT et.*
          FROM enterprise_turns et
          WHERE et.message_id = (
            SELECT MAX(et2.message_id) FROM enterprise_turns et2
            WHERE et2.thread_id = et.thread_id
          )
        ), active AS (
          SELECT latest.*, c.group_id, CAST(c.seat_count AS INTEGER) AS seat_count
          FROM latest
          JOIN customers c ON c.customer_id = latest.customer_id
          JOIN subscriptions s ON s.customer_id = latest.customer_id
          WHERE latest.closed = 0 AND (
            (latest.thread_type = 'new_lead' AND s.status = 'lead') OR
            (latest.thread_type != 'new_lead' AND s.status = 'subscribed' AND s.end_day IS NULL)
          )
        )
    """


def negotiation_summary(day: int) -> str:
    return _active_thread_cte() + f"""
        SELECT COUNT(*) AS open_threads,
               COALESCE(SUM(seat_count), 0) AS open_seats,
               SUM(CASE WHEN sender = 'customer' THEN 1 ELSE 0 END) AS awaiting_agent,
               AVG({day} - day) AS avg_waiting_days,
               MAX({day} - day) AS max_waiting_days
        FROM active
    """


def negotiation_details(day: int, limit: int) -> str:
    return _active_thread_cte() + f"""
        SELECT thread_id, customer_id, thread_type, group_id, seat_count,
               {day} - day AS waiting_days, day AS latest_day,
               sender AS latest_sender, message_text AS latest_message
        FROM active
        ORDER BY waiting_days DESC, thread_id
        LIMIT {limit}
    """


def negotiation_outcomes(start: int, end: int) -> str:
    """只统计每个线程最终一条公开消息上的明确接受或 Agent 拒绝。"""

    return f"""
        SELECT et.day, et.close_reason, COUNT(*) AS outcome_count
        FROM enterprise_turns et
        WHERE et.message_id = (
            SELECT MAX(et2.message_id) FROM enterprise_turns et2
            WHERE et2.thread_id = et.thread_id
        )
          AND et.close_reason IN ('accepted', 'agent_rejected')
          AND et.day BETWEEN {start} AND {end}
        GROUP BY et.day, et.close_reason
        ORDER BY et.day, et.close_reason
    """
