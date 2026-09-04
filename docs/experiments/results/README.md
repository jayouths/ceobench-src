# 实验结果设计

## 1. 设计原则

实验结果统一按照以下顺序设计：

```text
研究问题 -> 指标定义 -> 数据与统计口径 -> 图表形式
```

先确定论文需要回答什么问题、哪些指标能够形成证据，再决定使用折线图、点图、瀑布图或结果表。图表使用自身实验产物和统一指标口径生成。

当前只进行指标和结果方案设计。正式实验结果必须从 `outputs/runs/` 的原始产物统一生成，不手工维护中间 CSV。

单次运行结束后使用以下命令计算指标：

```bash
uv run --frozen python scripts/evaluate_run.py <run_dir>
```

默认生成：

```text
<run_dir>/evaluation/
├── metrics.json       # 保留完整标量、日级时序和分组明细
└── metrics_long.csv   # 统一长表，供多次运行拼接、统计和画图
```

多次运行完成后，按实验组目录聚合并与 Baseline 比较：

```bash
uv run --frozen python scripts/evaluate_experiment.py \
  outputs/runs/baseline \
  outputs/runs/analysis \
  --baseline baseline \
  --output-dir outputs/evaluation/opening
```

默认生成：

```text
outputs/evaluation/opening/
├── experiment_metrics.json  # 完整的组内描述统计和组间差异
├── group_summary.csv         # 各组标量指标的均值、标准差和有效样本数
├── group_comparisons.csv     # 破产率、标量指标和账本类别的组间差异
└── group_series.csv          # 按模拟日聚合的经营时序
```

提前破产运行只进入破产率、生存天数、回撤和实际存续期时序；终态现金、终态经营指标和累计账本差额只聚合跑满统一期限的运行。
每组只有一次运行时也可使用同一入口：输出保留 `n=1`，标准差为 `null`，绘图时不生成误差带或显著性结论。

评价程序只读 `world.nmdb`、运行配置、终态结果和调用日志，不修改模拟器状态。未满 28 天的运行无法形成完整末四周指标，对应值明确记录为 `null`。

## 2. 文档结构

| 文档 | 用途 |
|---|---|
| [metric-catalog.md](metric-catalog.md) | 统一定义候选指标、计算口径、数据来源、优先级和可用阶段 |
| [plans/final.md](plans/final.md) | 规划最终论文的研究问题、候选指标、证据关系和图表形式 |
| [plans/opening.md](plans/opening.md) | 从最终方案中收敛开题答辩可呈现的问题、指标和图表 |
| `figures/` | 保存两套方案共用的核心图、候选机制图和后续模块图的具体设计 |

## 3. 通用实验口径

- 最终论文计划比较 `baseline`、`analysis`、`analysis-deliberation` 和 `full`；开题阶段只比较前两组。
- 各组使用相同的决策模型、环境模型、模型参数、模拟器随机种子和实验时长，只改变创新模块开关。
- 主结果使用完整 497 天运行，不用 Smoke Test 代替正式实验。
- 单次运行不绘制置信区间或标准误；误差必须来自同一实验组的多次独立运行。
- 仅固定 `seed=42` 重复运行时，波动只表示 LLM 和 Provider 的随机性，不代表对不同模拟环境的泛化能力。
- `world.nmdb` 是经营状态和经营结果的权威数据源；运行日志负责记录 LLM 调用、模块成本和过程事件。
- 运行提前破产时保留真实生存轨迹，不补零、不外推，也不把破产日终态与第 497 天终态直接混算。
- 正式结果汇总所有满足实验协议的运行，不挑选表现最好的一次。
- 主结果使用 CEO-Bench 默认难度。统一调整模拟器难度的场景只作为敏感性或稳健性分析，与默认主结果分开报告。

## 4. 当前结果方案

### 4.1 开题核心证据

| 方案 | 主要回答的问题 | 状态 |
|---|---|---|
| 核心结果表 | Analysis 是否提高最终现金和生存表现，代价是多少 | 待正式实验 |
| [现金轨迹](figures/cash-trajectory.md) | 现金优势何时形成，是否稳定 | 核心图 |
| [多维经营结果](figures/operating-outcomes.md) | 是否同时改善持续收入、客户规模和近期盈利能力 | 核心图 |
| [现金差额来源](figures/cash-gap-waterfall.md) | 现金优势来自增收还是节支 | 优先机制图 |
| [现金预测准确性](figures/cash-forecast-accuracy.md) | Analysis 是否改善经营状态理解 | 优先机制图 |
| [模型成本收益](figures/model-cost-benefit.md) | 经营提升是否值得额外 API 成本 | 核心成本分析 |

### 4.2 候选机制图

这些指标应完整采集，但是否单独成图取决于它们能否解释稳定、重要的经营现象：

| 方案 | 独立价值成立的条件 | 状态 |
|---|---|---|
| [客户与持续收入轨迹](figures/business-growth-trajectory.md) | 两组增长阶段或转折点存在稳定差异 | 候选 |
| [每日实际收入轨迹](figures/daily-revenue-trajectory.md) | 需要解释 MRR、实际收款与现金流为何不同 | 候选 |
| [经营支出策略](figures/spending-strategy.md) | 投入时点和结构能够解释现金差额 | 候选 |
| [服务可靠性](figures/service-reliability.md) | Analysis 能稳定改善过载、宕机、错误率或延迟表现 | 候选 |

若候选图没有提供超出汇总指标的新信息，则只在结果表报告对应指标，或移至附录。

### 4.3 后续模块

[经营指标方向预测](figures/direction-forecast-accuracy.md)依赖 Deliberation 生成预测、Reflection 到期验证和复盘。它不属于开题阶段，也不能用当前 Baseline 构造虚假的预测准确率对照。

## 5. 图表确定规则

- P0 指标无论结果是否有利都必须报告。
- 正式实验前使用独立预实验判断指标是否重复、能否稳定测量，并锁定主结果指标。
- 正式实验后可以根据机制现象决定候选图进入正文还是附录，但不得删除已经锁定的负向或无显著结果。
- 同一指标只有在回答不同问题时才采用多种展示形式。例如最终 MRR 回答终态差异，MRR 轨迹回答增长时点与稳定性。
- 图表保留原始业务单位，不构造缺乏理论依据的综合得分，也不把不同模型的 Token 数直接相加比较。
