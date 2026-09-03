"""论文口径使用的 Analysis 信号来源与计算目录。"""

from __future__ import annotations

from dataclasses import dataclass


SIGNAL_CATALOG_VERSION = "1.0"


@dataclass(frozen=True)
class SignalDefinition:
    meaning: str
    sources: tuple[str, ...]
    formula: str
    window: str


SIGNAL_CATALOG = {
    "market.effective_leads": SignalDefinition(
        "进入个人选择结果或企业正式谈判的有效线索",
        ("public_week_snapshot.weekly_activity", "subscriptions", "enterprise_turns", "customers"),
        "个人取 subscription.start_day；企业取 new_lead 的 turn_number=0；按账户计数",
        "最近7天与前7天",
    ),
    "market.acquisition_mix": SignalDefinition(
        "有效线索的自然、网络和付费来源结构",
        ("subscriptions", "enterprise_turns", "customers.acquisition_source"),
        "各来源有效线索数 / 同窗口有效线索总数",
        "最近7天与前7天",
    ),
    "market.paid_acquisition": SignalDefinition(
        "广告投入产生原始线索和有效线索的效率",
        ("ad_channel_leads", "customers.acquisition_source"),
        "CPL=广告支出/对应线索数；原始与有效口径分别计算",
        "最近7天与前7天",
    ),
    "market.social_feedback": SignalDefinition(
        "公开客户社交反馈的数量和原文",
        ("social_media_posts",),
        "保留公开原文，不生成无监督情绪分数",
        "最近7天与前7天",
    ),
    "market.macro_condition": SignalDefinition(
        "当前可见的最近一期滞后宏观读数",
        ("macroeconomic_conditions",),
        "取最新已发布读数，并记录观察日与测量日间隔",
        "最新公开读数",
    ),
    "finance.current_cash": SignalDefinition(
        "当前可支配现金余额",
        ("public_week_snapshot.current_state.cash",),
        "直接读取当前周边界的公开现金快照",
        "当前时点",
    ),
    "finance.operating_revenue": SignalDefinition(
        "订阅收入与广告收入",
        ("ledger.subscription_payment", "ledger.ad_revenue"),
        "相邻 Ledger ID 边界内的订阅收入+广告收入",
        "相邻决策周与前一组相邻决策周",
    ),
    "finance.net_cash_flow": SignalDefinition(
        "企业现金在窗口内的真实净变化",
        ("public_week_snapshot.current_state.cash", "历史 signals.json"),
        "当前现金快照-上周现金快照",
        "相邻周边界与前一组相邻周边界",
    ),
    "finance.costs": SignalDefinition(
        "服务、获客、运营、开发和一次性投资成本",
        ("ledger", "历史 signals.json.finance.ledger_max_id"),
        "按相邻周度 Ledger ID 边界取增量，再按经济用途汇总负向流水并转为正成本",
        "相邻决策周与前一组相邻决策周",
    ),
    "finance.service_delivery_margin": SignalDefinition(
        "收入扣除计算与容量成本后的交付利润率",
        ("ledger",),
        "(经营收入-计算成本-容量成本)/经营收入",
        "最近7天与前7天",
    ),
    "finance.runway": SignalDefinition(
        "按经常性净消耗估算的现金可维持天数",
        ("ledger", "public_week_snapshot.current_state.cash"),
        "当前现金/最近28天平均每日经常性净消耗",
        "最近28个完整模拟日",
    ),
    "product.usage": SignalDefinition(
        "服务使用总量和日均使用量",
        ("service_day.total_usage_units",),
        "窗口求和与日均",
        "最近7天与前7天",
    ),
    "product.capacity": SignalDefinition(
        "连续容量利用率和超载天数",
        ("service_day.total_usage_units", "service_day.capacity_units"),
        "利用率=使用量/容量；计算均值、峰值、峰值超载幅度和大于1的天数",
        "最近7天与前7天",
    ),
    "product.reliability": SignalDefinition(
        "延迟、错误率和宕机表现",
        ("service_day",),
        "分别计算均值、峰值、宕机分钟和宕机天数",
        "最近7天与前7天",
    ),
    "product.configuration": SignalDefinition(
        "产品配置逐项变化",
        ("public_week_snapshot.configuration", "历史 signals.json"),
        "按套餐和配置项比较相邻周边界公开快照",
        "当前时点与上周时点",
    ),
    "product.research_pipeline": SignalDefinition(
        "在研、已完成项目及新完成数量",
        ("research_projects", "历史 signals.json"),
        "当前完成项目集合与上周公开集合做差",
        "当前时点与最近7天",
    ),
    "product.delivered_quality": SignalDefinition(
        "各公开客群和套餐的当前交付质量",
        ("public_week_snapshot.delivered_quality",),
        "直接复用 Dashboard 同源公开快照",
        "当前时点",
    ),
    "customer.customer_base": SignalDefinition(
        "当前个人账户、企业账户、企业席位和套餐结构",
        ("public_week_snapshot.current_state", "subscriptions", "customers"),
        "当前有效订阅按客户类型和套餐汇总",
        "当前时点与上周时点",
    ),
    "customer.new_paid_subscriptions": SignalDefinition(
        "窗口内开始付费的个人账户、企业账户和企业席位",
        ("subscriptions", "customers"),
        "按 start_day 统计 subscribed/cancelled，排除 lead/lost",
        "最近7天与前7天",
    ),
    "customer.churn": SignalDefinition(
        "取消、升降级及账户流失率",
        ("public_week_snapshot.weekly_activity", "历史 signals.json"),
        "周流失率=本周取消/周初活跃账户；28天口径累计四周取消",
        "最近7天、前7天与最近28天",
    ),
    "customer.issues": SignalDefinition(
        "工单存量、积压年龄、新开和解决效率",
        ("issues", "public_week_snapshot.current_state.open_issues"),
        "按 open_day/resolved_day 汇总，解决时长=resolved_day-open_day",
        "当前时点、最近7天与前7天",
    ),
    "customer.enterprise_negotiations": SignalDefinition(
        "有效开放谈判、等待压力和公开谈判结果",
        ("enterprise_turns", "subscriptions", "customers"),
        "取每个线程最新公开消息，并结合公开订阅状态排除失效线程",
        "当前时点、最近7天与前7天",
    ),
}
