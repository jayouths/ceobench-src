# Analysis 模块设计

## 1. 模块定位

Analysis 是“隐性经营状态识别”模块，只负责识别企业当前的经营状态。它不提供行动建议，不读取历史策略，不修改模拟环境，不包含后续的战略协商和历史复盘。

```text
Dashboard + 公开经营数据
→ 四个职能视角分别分析
→ 统一重构经营状态
→ 生成状态简报
→ 作为本周 ReAct Agent 的初始输入
```

## 2. 每周执行流程

Analysis 在每个新模拟周开始时执行一次，包括初始的 day 0。

1. Runner 读取 `game-status` 和 Dashboard。
2. Runner 通过现有 `/query` 接口读取公开经营数据。
3. 程序计算最近 7 天、之前 7 天及环比变化，不让 LLM 自行完成算术。
4. 市场、财务、产品和客户四个角色按固定顺序生成报告。
5. 状态重构 LLM 合并四份报告，生成统一经营画像。
6. 程序将经营画像确定性地格式化为 `STRATEGY_BRIEF.md`。
7. Runner 将 `Dashboard + 状态简报` 一起交给原始 ReAct Agent。

正常情况下，每周包含 4 次角色报告调用和 1 次状态重构调用，共 5 次 Analysis LLM 调用。70 天实验约调用 50 次，497 天完整实验约调用 355 次；非法输出的修复调用另行计数和计费。

## 3. 数据边界与统计信号

Analysis 只能通过现有 `/query` 接口查询数据。该接口继续负责禁止写操作、隐藏表、隐藏字段和 Schema 探测。Analysis 不直接打开 `world.nmdb`，不为创新模块增加更高的数据权限。

| 角色 | 公开原始数据 | 程序生成的统计信号 |
| --- | --- | --- |
| 市场 | `customers`、`config_history`、`social_media_posts`、`group_info_levels` | 近两周获客、渠道结构、广告投入、市场信息变化 |
| 财务 | `ledger` | 现金、收入、支出、净现金流、分类成本和环比 |
| 产品 | `service_day`、`config_history`、`research_projects` | 延迟、错误率、宕机、容量利用、产品配置和研发进展 |
| 客户 | `subscriptions`、`customers`、`issues`、`enterprise_turns` | 新增、流失、套餐结构、工单压力和企业客户状态 |

每个统计信号必须保留原始值、时间窗口、最近窗口值、对照窗口值和派生变化。day 0 等数据不足的时点必须明确标注覆盖天数或 `insufficient_data`，不得伪造完整的两周对比。

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
| `src/saas_bench/llm_provider.py` | 在统一 LLM 返回结构中补充推理 Token |
| `src/saas_bench/agents/bash_agent/analysis/` | 新增 Schema、统计信号、Prompt、流程编排和简报生成 |
| `src/saas_bench/agents/bash_agent/run_test.py` | 接入独立模型、每周调用、简报注入、用量与恢复 |
| `src/saas_bench/experiment_config.py` 与 `experiments/*.toml` | 补齐模块和模型配置 |
| `tests/` | 分别覆盖 Schema、信号、流程、产物、消融和恢复 |

本模块不修改 `simulation.py`、`database.py`、`config.py` 等模拟器底层规则，不修改 Agent 可见的公开工具权限。
