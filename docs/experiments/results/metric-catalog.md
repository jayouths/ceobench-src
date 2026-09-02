# 实验指标目录

## 1. 用途

本文件先定义“研究什么”，图表文档再决定“如何展示”。同一指标可以形成终态、全周期汇总或时序统计，但只有回答不同研究问题时才重复展示。

优先级含义：

- `P0`：核心结论必须报告，不根据结果好坏删减。
- `P1`：重要解释指标，正式实验前根据独立预实验锁定部分指标。
- `P2`：机制或附录候选，只在能够解释独立现象时成图。

## 2. 核心经营结果

| 指标 | 优先级 | 计算口径 | 数据来源 | 可用阶段 |
|---|---|---|---|---|
| 破产率 | P0 | 破产运行数除以全部合规运行数 | 运行结果 + `ledger` | Analysis |
| 生存天数 | P0 | 运行实际完成的模拟天数 | 运行结果 | Analysis |
| 最终现金 | P0 | 截止终止日全部 `ledger.amount` 累计值 | `ledger` | Analysis |
| 现金轨迹 | P0 | 按日累计 `ledger.amount`，主图按周取样 | `ledger` | Analysis |
| 最大现金回撤 | P1 | 历史现金峰值到后续最低点的最大降幅 | `ledger` | Analysis |

最终现金是 CEO-Bench 的北极星指标；破产率和生存天数优先于提前破产运行的终态比较。

## 3. 持续经营能力

| 指标 | 优先级 | 计算口径 | 数据来源 | 可用阶段 |
|---|---|---|---|---|
| 最终 MRR | P0 | 终止日所有有效个人和企业订阅的实际月费 | `subscriptions` + `customers` | Analysis |
| 活跃个人订阅数 | P0 | 终止日有效个人订阅数量 | `subscriptions` + `customers` | Analysis |
| 企业订阅席位数 | P0 | 终止日有效企业订阅覆盖席位之和 | `subscriptions` + `customers` | Analysis |
| 末四周平均每周净现金流 | P0 | 最后 28 天账本净额除以 4 | `ledger` | Analysis |
| 末四周实际订阅收入 | P1 | 最后 28 天 `subscription_payment` 合计 | `ledger` | Analysis |
| 末四周个人订阅净增长 | P1 | 终止日个人订阅数减去 28 天前数值 | `_eval_subscription_day` | Analysis |
| 末四周企业席位净增长 | P1 | 终止日企业席位减去 28 天前数值 | `_eval_subscription_day` | Analysis |
| 个人订阅流失率 | P1 | 窗口内结束订阅数除以期初有效订阅数加窗口内新增数 | `_eval_subscription_event` + `_eval_subscription_day` | Analysis |
| 企业席位流失率 | P1 | 窗口内结束订阅的席位数除以期初有效席位加窗口内新增席位 | `_eval_subscription_event` + `_eval_subscription_day` | Analysis |
| 活跃客户群数量 | P1 | 终止日仍有有效订阅的 `group_id` 数量 | `subscriptions` + `customers` | Analysis |
| MRR 客群集中度 | P1 | 各客群 MRR 占比的 HHI 或最大客群占比 | `_eval_subscription_day` | Analysis |

MRR 表示当前订阅结构未来每月可持续产生的收入，与历史累计形成的现金不同。

## 4. 服务质量与经营风险

| 指标 | 优先级 | 计算口径 | 数据来源 | 可用阶段 |
|---|---|---|---|---|
| 累计宕机时间 | P1 | 全周期 `downtime_minutes` 合计 | `service_day` | Analysis |
| 严重故障周数 | P1 | 超过预先锁定错误率或宕机阈值的周数 | `service_day` | Analysis |
| 平均错误率 | P2 | 先在 run 内按周期聚合，再比较实验组 | `service_day` | Analysis |
| P95 延迟 | P2 | 按周或全周期聚合，必须注明均值或峰值 | `service_day` | Analysis |
| 容量利用率 | P2 | `total_usage_units / capacity_units` | `service_day` | Analysis |
| 过载天数或周数 | P2 | `overload > 0` 的周期数量 | `service_day` | Analysis |
| 未解决工单数 | P1 | 终止日仍为未解决状态的工单数 | `issues` | Analysis |
| 未解决工单平均积压天数 | P1 | 终态未解决工单 `days_open` 均值 | `issues` | Analysis |

错误率与 P95 延迟通常受同一过载状态驱动。若二者高度相关，正文只保留业务含义更明确的一项，完整统计仍进入结果表或附录。

## 5. 经营行为与现金来源

| 指标 | 优先级 | 计算口径 | 数据来源 | 可用阶段 |
|---|---|---|---|---|
| 各账本类别现金贡献 | P1 | 实验组与 Baseline 的同类账本累计金额均值之差 | `ledger` | Analysis |
| 各类经营支出 | P2 | 按周或全周期累计支出绝对值 | `ledger` | Analysis |
| 广告投入产出比 | P2 | 广告相关收入与获客支出的预先固定口径 | `ledger` | Analysis |
| 定向研发投入占比 | P2 | 定向研发支出除以全部研发支出 | `ledger` + 配置历史 | Analysis |
| 客群发现数量与研究等级 | P2 | 已发现客群数及研究等级汇总 | `group_info_levels` | Analysis |

所有账本分类必须覆盖正式运行中出现的全部 `ledger.category`。未识别类别不得静默丢弃。

## 6. 预测与认知能力

| 指标 | 优先级 | 计算口径 | 数据来源 | 可用阶段 |
|---|---|---|---|---|
| 现金预测绝对百分比误差 | P1 | run 内按期限计算 APE 均值，再比较实验组 | `predictions` + `ledger` | Analysis |
| 现金预测有符号误差 | P2 | 保留误差方向，只用于判断高估或低估偏差 | `predictions` + `ledger` | Analysis |
| 95% 预测区间覆盖率 | P2 | 真实现金落入预测区间的成熟预测比例 | `predictions` + `ledger` | Analysis |
| 预测区间相对宽度 | P2 | 区间宽度除以真实现金绝对值 | `predictions` + `ledger` | Analysis |
| 经营指标方向命中率 | P1 | 到期方向与预先定义的真实方向一致的比例 | 待新增预测记录 | Deliberation + Reflection |

只评价目标日期已经真实发生的成熟预测，不外推第 497 天之后的现金，也不把同一 run 的多次预测伪装成独立重复实验。

## 7. 模型调用与成本

| 指标 | 优先级 | 计算口径 | 数据来源 | 可用阶段 |
|---|---|---|---|---|
| 各模块 LLM 调用次数 | P1 | 按 `bash_agent`、`analysis`、`deliberation`、`reflection` 分组 | 性能日志 | Analysis 起 |
| 输入、缓存、输出、推理 Token | P1 | 按模型和模块分别累计，不跨模型直接合并比较效率；未上报的推理 Token 记为 `null` | 性能日志 | Analysis 起 |
| 各模块 API 成本 | P0 | 使用配置中锁定的模型官方单价计算 | 性能日志 | Analysis 起 |
| 整次实验 Agent API 成本 | P0 | 决策 Agent 与创新模块成本合计 | 性能日志 | Analysis 起 |
| 模拟器环境 LLM 成本 | P1 | 与 Agent 成本分开报告 | `api_costs` 或环境日志 | Analysis 起 |
| 增量模拟现金 | P0 | 实验组平均最终现金减去 Baseline 平均最终现金 | `ledger` | Analysis 起 |
| 增量 API 成本 | P0 | 实验组平均 Agent 成本减去 Baseline 平均 Agent 成本 | 性能日志 | Analysis 起 |
| 单位成本现金增量 | P1 | 增量模拟现金除以增量 API 成本 | 上述两项 | Analysis 起 |

模拟企业现金与真实 API 支出属于不同语义的货币量。论文中可以报告“每增加 1 美元 API 成本对应的模拟现金增量”，但不能把它表述为真实商业利润或直接回本。

## 8. 采集状态与后续缺口

经营结果统一以 `world.nmdb` 中的数据表为准；轨迹和性能日志只记录运行过程、模型调用与成本，不承担经营状态采集。

- TODO：实现 Deliberation 时记录逐次结构化经营指标方向预测。
- TODO：实现 Reflection 时记录到期真实方向和预测命中结果。
- TODO：Deliberation 与 Reflection 接入统一的调用、Token 和成本日志。
- `_eval_subscription_day` 已记录日末订阅、席位和 MRR 存量；`_eval_subscription_event` 已记录真实订阅开始与结束事件。两表均不向 Agent 开放。
- 正式实验前需要锁定严重故障阈值、广告投入产出比口径和方向预测阈值。
- 图表代码尚未开发；本阶段先保证原始指标可采集、可追溯和可复算。
