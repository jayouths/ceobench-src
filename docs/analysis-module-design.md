# Analysis 模块设计

## 1. 模块定位

Analysis 是“隐性经营状态识别”模块，只负责识别企业当前的经营状态。它不提供行动建议，不读取历史策略，不修改模拟环境，不包含后续的战略协商和历史复盘。

```text
公开周度经营快照 + 公开经营明细
→ 四个职能视角分别分析
→ 统一重构经营状态
→ 生成状态简报
→ 作为本周 ReAct Agent 的初始输入
```

## 2. 每周执行流程

Analysis 在每个新模拟周开始时执行一次，包括初始的 day 0。

1. Runner 读取 `game-status` 和 `/dashboard` 返回的 `public_week_snapshot`；Dashboard 文本继续原样提供给 ReAct Agent。
2. Runner 通过现有 `/query` 接口读取公开经营数据。
3. 程序计算最近 7 天、之前 7 天及环比变化，不让 LLM 自行完成算术。
4. 市场、财务、产品和客户四个角色按固定顺序生成报告。
5. 状态重构 LLM 合并四份报告，生成统一经营画像。
6. 程序将经营画像确定性地格式化为 `STRATEGY_BRIEF.md`。
7. Runner 将 `Dashboard + 状态简报` 一起交给原始 ReAct Agent。

正常情况下，每周包含 4 次角色报告调用和 1 次状态重构调用，共 5 次 Analysis LLM 调用。70 天实验约调用 50 次，497 天完整实验约调用 355 次；非法输出的修复调用另行计数和计费。

## 3. 数据边界与统计信号

Analysis 只能通过现有 `/query` 接口查询结构化经营明细。该接口继续负责禁止写操作、隐藏表、隐藏字段和 Schema 探测。除此之外，模块只使用 Baseline 已公开的 `public_week_snapshot`，以及此前由 Analysis 保存的公开统计快照；不直接打开 `world.nmdb`，不为创新模块增加更高的数据权限。

Dashboard 不再作为一段需要解析的文本输入。模拟器在每个周边界只生成一次 `public_week_snapshot`，Dashboard 渲染器和 Analysis 信号计算器分别消费其中需要的字段：

```text
数据库当前状态 + 本周 DayResult
            ↓
  public_week_snapshot
       ↙           ↘
Dashboard 文本      Analysis 统计信号
```

快照统一保存当前经营状态、本周经营活动、当前配置、交付质量、社交帖子摘要、周度计算和收件箱。现金、客户基础、取消数、使用量和服务质量等字段可由 Dashboard 与 Analysis 共用；周度计算、收件箱等内容主要用于 Dashboard；环比、现金跑道、获客效率和利润率等 Analysis 专用指标则由快照与 `/query` 返回的公开明细进一步确定性计算。字段本身不写入消费者标签，由 Dashboard 渲染器和 Analysis 信号生成器显式声明各自依赖，避免展示职责污染数据模型。

`/dashboard` 当前同时返回 Dashboard 文本和结构化快照：

```json
{
  "dashboard": "...",
  "day": 7,
  "public_week_snapshot": {}
}
```

`modules.analysis.enabled = false` 时，Runner 仍只把原有 Dashboard 文本交给决策 Agent，不执行额外查询或 LLM 调用。该结构化改造只统一数据口径，不改变模拟器规则、随机数状态或 Baseline 上下文。

字段不照搬交付原型，而是从业务含义和状态识别目标反向选择。每个候选字段必须满足以下要求：

1. **语义准确**：先确认字段在模拟器中的真实含义。例如，`customers.created_day` 记录潜在对象创建，其中可能包含未被 Dashboard 认定为有效线索的企业对象，更不等同于成功订阅或付费获客。
2. **状态相关**：字段必须能支持五个经营状态维度中的至少一个，或用于解释状态变化；只反映 Agent 信息掌握程度的字段不进入经营状态。
3. **权限一致**：只使用原始 Agent 可通过 `/query` 读取的公开表和公开字段，不读取隐藏状态，不提升实验组权限。
4. **时间有效**：明确存量、流量和事件的区别，固定最近 7 天与之前 7 天窗口，并保留实际覆盖天数和反馈滞后。
5. **计算确定**：求和、均值、比例和环比由程序完成；LLM 只解释已计算信号，不自行承担可确定的算术。
6. **证据可追溯**：每个派生值都能还原到来源表、字段、查询窗口和计算公式，便于复现实验和撰写论文。
7. **避免重复和误导**：不因为字段容易查询就加入，也不把相关关系包装成因果关系。

每个正式信号都要进入信号字典，至少记录：信号名称、业务含义、来源表和字段、计算公式、时间窗口、方向解释、对应状态维度、反馈滞后、缺失值处理、与 Dashboard 的关系，以及最终保留理由。

| 角色 | 实现状态 | 公开原始数据 | 当前信号设计 |
| --- | --- | --- | --- |
| 市场 | 已实现并测试 | `subscriptions`、`customers`、`enterprise_turns`、`ad_channel_leads`、`social_media_posts`、`macroeconomic_conditions` | 有效线索、来源结构、付费获客效率、外部反馈和宏观环境 |
| 财务 | 已实现并测试 | `ledger`、`public_week_snapshot`、历史 `signals.json` | 现金、收入、支出、净现金流、成本结构、交付利润率和现金跑道 |
| 产品 | 已实现并测试 | `service_day`、`config_history`、`research_projects`、`public_week_snapshot`、历史 `signals.json` | 使用量、容量压力、可靠性、产品配置和研发管线 |
| 客户 | 已实现并测试 | `subscriptions`、`customers`、`issues`、`enterprise_turns`、`public_week_snapshot`、历史 `signals.json` | 客户基础、新增、流失、工单压力和企业谈判状态 |

### 3.1 市场信号设计

市场角色用于识别需求动量，并为经营状态变化提供外部环境解释。当前保留五类输入：

| 信号组 | 来源 | 设计说明 |
| --- | --- | --- |
| 有效线索 | 个人取 `subscriptions.start_day`；企业取 `enterprise_turns` 中 `new_lead` 线程的首次日期 | 比较两个 7 天窗口。不能直接使用 `customers.created_day`，因为模拟器会创建不满足任何方案的企业对象，但它们不属于 Dashboard 的有效线索。 |
| 来源结构 | 有效线索集合联结 `customers.acquisition_source` | 分别统计自然、网络效应和付费渠道带来的有效线索，排除只有客户行、没有订阅记录或 `new_lead` 线程的无效企业对象。 |
| 付费获客效率 | `ad_channel_leads` 与有效线索集合 | `ad_channel_leads.leads_generated` 表示广告生成的原始线索，可能包含不具备可行方案的企业对象。分别计算原始线索 CPL 和有效线索 CPL，不能把两种口径混用。零分母时结果为未定义。 |
| 外部反馈 | `social_media_posts.day/content` | 保留近期公开帖子原文供市场角色分析。程序不预先生成缺乏验证标准的情绪分数。 |
| 宏观环境 | `macroeconomic_conditions` | 使用当前已经公开的最新读数及其测量日期，明确约 30 天发布滞后，避免将宏观需求变化错误归因于企业策略。 |

`group_info_levels` 不进入 Analysis。它表示 Agent 对客群参数的了解程度，属于认知状态而非企业经营状态。Dashboard 已提供当周线索总量；Analysis 仅补充前一窗口、来源结构和外部解释，不重复制造同义指标。

### 3.2 财务信号设计

财务角色用于识别现金健康度和单位经济性。收入、支出和净现金流比较最近 7 天与之前 7 天；现金跑道使用最近 28 天，降低按月订阅付款在短窗口内造成的波动。

| 信号组 | 来源或公式 | 设计说明 |
| --- | --- | --- |
| 当前现金 | `public_week_snapshot.current_state.cash` | 这是与 Dashboard 同源的时点存量。`initial_funding` 计入现金，但不计入经营收入。 |
| 经营收入 | `ledger.subscription_payment + ledger.ad_revenue` | 分项保留订阅收入和广告收入，并与成本共用相邻决策周的 Ledger ID 边界。 |
| 净现金流 | 当前现金快照减去上周现金快照 | 反映相邻决策周之间的真实现金增减，包含周边界发生的一次性研究投入。 |
| 成本结构 | 按经济用途重新归类 | Ledger 中成本是负数；展示时转换为正的绝对成本。服务交付为 `compute + capacity`；获客为 `advertising + lead_acquisition_cost`；运营、开发分别保留；市场研究、客群研究和研发项目归为一次性投资。 |
| 服务交付利润率 | `(经营收入 + compute + capacity) / 经营收入` | `compute` 和 `capacity` 使用 Ledger 中的负值。等价写法是收入减去两项绝对成本。收入为零时不计算。 |
| 经常性现金消耗 | `max(0, -(经营收入 + 各项经常性负向流水))` | 包含服务交付、获客、运营和开发，排除 `market_research`、`group_research` 和 `research_project`，避免一次性投资扭曲持续经营速度。 |
| 现金跑道 | `当前现金 / 最近 28 天平均每日经常性净消耗` | 仅在经常性现金流为负且覆盖完整 28 天时计算；否则标记为不适用或数据不足。 |

模型调用成本不属于模拟企业经营流水。Bash Agent、Analysis 以及环境 LLM 的 Token 和金额继续独立记录，不进入现金、利润率或现金跑道。

财务流水不能只按 `ledger.day BETWEEN ...` 划分。Agent 会在周边界日完成 Analysis 后立即决策，该日产生的研究投入等流水应属于接下来的一周。为避免遗漏，`signals.json` 保存当时公开的 `ledger_max_id`；下一周读取 `id > 上周 ledger_max_id` 的流水。该 ID 只用于确定增量边界，不作为经营证据交给 LLM。经营收入、成本和利润率使用这批增量流水，净现金流使用相邻现金快照差做权威口径。

### 3.3 产品信号设计

产品角色主要识别服务压力，并为客户健康度和单位经济性提供产品侧证据。

| 信号组 | 来源或公式 | 设计说明 |
| --- | --- | --- |
| 使用量趋势 | `service_day.total_usage_units` | 比较两个 7 天窗口的总量和日均值，作为容量压力的需求侧背景。 |
| 容量利用 | `total_usage_units / capacity_units` | 计算平均值、峰值、超载天数，以及 `max(利用率 - 1, 0)` 的峰值。保留连续值，不在信号层提前分类。 |
| 服务可靠性 | `p95_ms`、`error_rate`、`downtime_minutes` | 分别保留两个窗口的平均值、峰值、宕机总分钟和宕机天数，避免只看一次极值。 |
| 产品配置 | `config_history` | 比较当前日与一周前有效的模型档位、配额、容量档位、运营投入和开发投入，并列出期间变化。价格不归入产品状态。 |
| 研发管线 | `research_projects` | 区分进行中与已完成项目。进行中项目的预计完成日和预计提升只表示未来管线；只有已完成项目的 `quality_boost_applied` 能证明能力已经释放。 |
| 当前交付质量 | `public_week_snapshot.delivered_quality` | 这是 Baseline 已通过 Dashboard 公开给 Agent 的当前能力信息，可以作为证据；Analysis 不查询隐藏质量状态。 |

`config_history` 实际保存配置变更后的快照，不能当作连续的逐日观测求平均。延迟、错误率和宕机包含随机波动，产品角色可以提出原因假设，但不得仅凭时间先后认定某次配置修改造成了指标变化。

同一天多次配置操作会覆盖为该日最终快照，因此产品配置不能仅查询自然日 `1..7`。实现以上周 `signals.json` 中的公开配置为基线，并比较包含周初边界日的 `config_history`；这样 day 0 或 day 7 在 Analysis 之后发生的修改，会进入下一周信号。无法还原同一天内被覆盖的中间步骤，只记录最终有效配置变化。

### 3.4 客户信号设计

客户角色用于识别客户健康度，并发现尚未完全反映到收入中的流失风险。

| 信号组 | 来源或公式 | 设计说明 |
| --- | --- | --- |
| 当前客户基础 | `subscriptions` 联结 `customers` | 分开统计活跃个人订阅数、企业客户数、企业席位数和当前套餐结构，不把个人账户与企业席位直接相加。 |
| 新增付费订阅 | `start_day` 与订阅状态 | 只统计当前状态为 `subscribed` 或 `cancelled` 的记录并比较两个 7 天窗口。`lead` 和 `lost` 从未形成付费订阅，必须排除；已经取消的历史付费订阅仍应计入其开始窗口。 |
| 取消订阅 | 当前 `public_week_snapshot.weekly_activity` 与历史公开统计快照 | 快照中的 `cancellations` 是 Dashboard 同源的本周权威值。企业谈判超时时，`subscriptions.end_day` 可能写为未来合同到期日，因此不能用它重建所有历史取消事件。 |
| 客户基础净变化 | 当前与上周公开客户基础快照之差 | 分别比较活跃个人订阅数和企业席位数；不把账户与席位混成同一个净增指标。 |
| 流失率 | 周度取消数 / 周初活跃账户数 | 7 天流失率使用相邻快照；28 天流失率累计最近四次完整周度取消数，并使用 28 天前的活跃账户快照。公开数据不能可靠拆分所有历史取消席位和类型，因此不输出这些指标。 |
| 工单压力 | `issues` | 统计当前未解决数量、平均和最长积压时间、超 7 天与超 14 天工单，并比较两个窗口的新开、解决数量和解决时长。 |
| 企业谈判 | `enterprise_turns` 联结 `subscriptions` | 统计当前有效线程的类型、席位、等待时间和最新公开消息。不能只用 `closed=0`，还要结合公开订阅状态排除已经流失或取消但内部超时标记不可见的线程。 |
| 谈判结果 | 最新公开线程状态 | 比较两个窗口内 `accepted` 和 `agent_rejected` 的数量；内部超时原因不可见，不推断为已知事实。 |

订阅表只保留当前套餐，不能可靠还原历史升级和降级路径，因此历史套餐趋势不从 SQL 伪造，直接使用 Dashboard 已公开的本周升级、降级汇总。`churn_reason`、客户满意度和关系值均为隐藏状态，Analysis 只能通过公开的客户基础、流失、工单、谈判和反馈识别潜在客户状态。

### 3.5 历史公开统计快照

为了比较 Dashboard 独有的周度指标，Analysis 可以读取此前生成的 `signals.json`，但只复用当时已经公开的确定性统计值，包括取消数、升级数、降级数、活跃个人订阅、活跃企业账户和企业席位。

该机制只形成统计时间序列，不读取或复用历史 `role_reports.json`、`state_portrait.json`、`STRATEGY_BRIEF.md`，因此不承担模块三的策略复盘和经验记忆功能。历史快照缺失时，对应比较必须标记为 `insufficient_data`，不得改用含义不一致的字段近似补齐。

可比较的流量信号必须保留原始值、时间窗口、最近窗口值、对照窗口值和派生变化；现金等时点存量则明确记录观察日。day 0 等数据不足的时点必须标注实际覆盖天数或 `insufficient_data`，不得伪造完整的两周对比。

### 3.6 `signals.json` 数据契约

确定性信号层已实现于 `analysis/signal_models.py`、`analysis/signals.py` 和 `analysis/signal_catalog.py`。每周产物的顶层结构固定为：

```json
{
  "schema_version": "1.0",
  "signal_catalog_version": "1.0",
  "day": 7,
  "week": 1,
  "windows": {},
  "public_week_snapshot": {},
  "market": {},
  "finance": {},
  "product": {},
  "customer": {}
}
```

每个可比较数值分别保存 `current`、`previous`、绝对变化、相对变化、方向和比较状态。`current` 与 `previous` 各自携带 `available / insufficient_data / not_applicable` 状态，避免把“数据不足”和“零值”混淆。最近 7 天为 day 1 至 day 7 这类完整模拟日窗口，前 7 天为其前一个完整窗口；现金和配置等周边界状态按上文的相邻公开快照规则处理。

Runner 在 `modules.analysis.enabled = true` 时生成 `analysis/day_XXX/signals.json`，同一日期已有合法产物则直接复用。断点恢复会保留断点日及以前的产物并删除更晚的孤立目录。当前提交只完成确定性信号层，四角色 LLM、经营画像和 `STRATEGY_BRIEF.md` 仍按后续步骤实现。

## 4. 角色报告

四个角色都只分析与自身职能相关的信号，不提供经营行动建议。结构化输出包含：

- `evidence`：可由公开统计信号支持的经营事实，包含唯一证据 ID、原始指标、变化方向、证据强度和反馈滞后说明。
- `hypotheses`：对事实成因的暂时解释，必须引用证据 ID，并给出置信度和可执行的后续验证方式。
- `risks`：尚未充分暴露但已有早期信号的风险，包含早期指标、预计时间范围和严重程度。

Prompt 必须包含字段语义、枚举范围、数量上限和最小合法输出示例，不能只给字段名让模型猜测。

## 5. 统一经营画像

状态重构必须输出五个固定维度，每个维度包含标签、置信度、证据 ID 和判断理由。

| 维度 | 标签空间 |
| --- | --- |
| 现金健康度 | `healthy / watch / stressed / critical / insufficient_data` |
| 需求动量 | `contracting / stable / growing / surging / insufficient_data` |
| 单位经济性 | `healthy / marginal / loss_making / insufficient_data` |
| 服务压力 | `underutilized / balanced / pressured / overloaded / insufficient_data` |
| 客户健康度 | `healthy / watch / deteriorating / critical / insufficient_data` |

不保留 `state_label`。五个固定维度及各自的离散标签共同构成唯一的机器可比较状态，避免再增加一个语义重复且难以验证的综合标签。`diagnosis` 仅保留一句自然语言总结，供决策 Agent 阅读，不参与状态分类。

经营画像同时包含：

- `facts`：由至少两个独立信号，或一个无歧义的财务、服务指标支持的事实。
- `hypotheses`：包含支持证据、反对证据、竞争性解释和验证方式的原因假设。
- `latent_risks`：尚未体现在头部指标中的滞后风险。
- `causal_chain`：结构化的因果步骤，每步包含原因、结果、证据 ID 和置信度，不使用无结构字符串列表。

## 6. 输出校验与失败处理

工程层使用 Pydantic 校验角色报告和经营画像的 JSON 结构。第一次输出不合法时，将原始回答、校验错误和目标 Schema 一起交给模型修复。修复次数由实验配置显式声明：

```toml
[modules.analysis]
enabled = true
max_schema_retries = 1
```

达到上限后仍无法校验通过时，实验直接失败，不用空数组、默认标签或旧报告静默降级。首次调用和修复调用都必须记录 Token、成本、耗时和原始响应。

本节只规定工程 Schema 校验。利用隐藏环境状态评价识别准确率，属于论文实验层的效果校验，暂留 TODO。

## 7. 过程产物

每个模拟周保存一份不可混淆的独立产物：

```text
run_<run_id>/analysis/day_007/
├── signals.json
├── role_reports.json
├── state_portrait.json
└── STRATEGY_BRIEF.md
```

- `signals.json`：SQL 结果及程序计算的统计信号。
- `role_reports.json`：四个职能角色的结构化报告。
- `state_portrait.json`：统一经营状态画像。
- `STRATEGY_BRIEF.md`：由程序格式化、实际注入本周 ReAct Agent 上下文的内容。

四份产物应使用原子写入，防止中途崩溃留下外观完整但内容截断的文件。Analysis LLM 的每次调用还要进入现有原始响应和耗时日志，并标记 `component=analysis`、角色和任务。

## 8. Token、成本与断点恢复

Analysis 必须独立统计：

- LLM 调用次数。
- `input_tokens`、`output_tokens`、`cached_tokens` 和 `reasoning_tokens`。
- 按市场、财务、产品、客户和状态重构拆分的用量。
- 按供应商结算币种分开累计的成本。

Analysis 用量不得与 Bash Agent 或社交环境 LLM 混合。推理 Token 只作为观测指标；如果供应商已将其包含在 `output_tokens` 中，不重复计费。

Analysis 模型配置、模块开关、累计用量和已完成的模拟周必须进入 `config.json`、Checkpoint 和 `result.json`。Analysis 完成后，在 ReAct Agent 开始本周决策前生成一个同日稳定断点。因此，恢复同一模拟周时应复用已完成产物，不重复调用 Analysis LLM，不重复计费。

## 9. 消融边界

`modules.analysis.enabled = false` 时必须严格保持当前 Baseline：

- 不查询 Analysis 数据。
- 不创建 Analysis LLM 客户端。
- 不调用 Analysis 模型。
- 不生成状态简报。
- Dashboard 原样交给 ReAct Agent。

实验组只增加 Analysis，决策模型、社交环境模型、随机种子、模拟天数和模拟器配置必须与 Baseline 一致。等成本对照组和利用隐藏状态的离线准确性评价暂留 TODO。

## 10. 代码改动边界

| 位置 | 计划改动 |
| --- | --- |
| `src/saas_bench/public_week_snapshot.py` | 统一构建公开周度经营事实并确定性渲染 Dashboard |
| `src/saas_bench/api_server.py` 与 `environment.py` | 在 `/dashboard` 暴露结构化快照，Dashboard 复用同一数据来源 |
| `src/saas_bench/llm_provider.py` | 在统一 LLM 返回结构中补充推理 Token |
| `src/saas_bench/agents/bash_agent/analysis/` | 新增 Schema、统计信号、Prompt、流程编排和简报生成 |
| `src/saas_bench/agents/bash_agent/run_test.py` | 接入独立模型、每周调用、简报注入、用量与恢复 |
| `src/saas_bench/experiment_config.py` 与 `experiments/*.toml` | 补齐模块和模型配置 |
| `tests/` | 分别覆盖 Schema、信号、流程、产物、消融和恢复 |

本模块不修改 `simulation.py`、`database.py`、`config.py` 等模拟器底层规则，不修改 Agent 可见的公开工具权限。
