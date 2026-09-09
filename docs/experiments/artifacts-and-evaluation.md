# 实验产物与离线评价

## 1. 运行目录

每次实验写入独立目录：

```text
outputs/runs/<experiment_name>/<北京时间>_seed-<seed>_<run_id>/
├── world.nmdb
├── config.json
├── checkpoint.json
├── result.json                     # 运行到终态后生成
├── logs/
│   ├── trajectory_<run_id>.jsonl
│   └── performance_<run_id>.jsonl
├── analysis/day_XXX/             # 仅 Analysis 组存在
└── evaluation/                   # 执行离线评价后生成
```

目录名用于人工识别，实验身份和参数以目录内的 `config.json` 为准。

## 2. 产物职责

| 产物 | 职责 | 是否为经营指标来源 |
|---|---|---:|
| `world.nmdb` | 保存公开经营数据、模拟器内部状态和实验评价事实 | 是 |
| `config.json` | 保存本次运行实际采用的完整配置 | 否 |
| `checkpoint.json` | 保存可恢复的运行边界和状态索引 | 否 |
| `result.json` | 保存终态结果和累计模型用量，便于快速查看 | 部分；正式复算仍以数据库和日志为准 |
| `logs/trajectory_*.jsonl` | 按发生顺序记录周边界、模型调用、工具执行和重试 | 否 |
| `logs/performance_*.jsonl` | 记录模型调用、耗时、Token 和成本汇总 | 仅运行与成本指标 |
| `analysis/day_XXX/` | 保存信号、角色报告、经营画像和战略简报 | 仅模块过程分析 |

经营结果统一从 `world.nmdb` 复算；日志不承担现金、订阅或服务状态采集。`result.json` 是运行结束时的便捷摘要，不替代原始事实。

## 3. 数据可见性

数据库同时包含三类信息：

| 信息层 | 普通 Agent / Analysis | Oracle | 离线评价器 |
|---|---:|---:|---:|
| 公开经营数据 | 可通过公开工具读取 | 可读 | 可读 |
| 模拟器内部状态 | 不可读 | 可读 | 按评价需要读取 |
| `_eval_*` 实验评价事实 | 不可读 | 可读 | 可读 |

正常执行链路中，普通 Agent 和 Analysis 不直接打开 `world.nmdb`，只能经公开 API 查询已登记的表和字段。Oracle 是显式启用的白盒上界模式，仍然只读。离线评价发生在运行结束后，结果不得回流当次决策。客户端文件的物理访问限制见 [数据库加密与访问边界](../engineering/database-encryption.md)。

## 4. 单次运行评价

评价一个运行目录：

```bash
uv run --frozen python scripts/evaluate_run.py \
  outputs/runs/<experiment_name>/<run_dir>
```

默认生成：

```text
<run_dir>/evaluation/
├── metrics.json
└── metrics_long.csv
```

`metrics.json` 保留完整层级结构；`metrics_long.csv` 将标量、时序、账本分类和模型用量展开成长表，供 pandas、统计程序和绘图脚本直接读取。指标定义统一维护在 [实验指标目录](results/metric-catalog.md)。

## 5. 多次运行聚合

按实验组目录聚合多次已经终结的独立运行：

```bash
uv run --frozen python scripts/evaluate_experiment.py \
  outputs/runs/baseline \
  outputs/runs/analysis \
  --baseline baseline \
  --output-dir outputs/evaluation/<evaluation_name>
```

输出包括：

| 文件 | 内容 |
|---|---|
| `experiment_metrics.json` | 全部描述统计与组间差异 |
| `group_summary.csv` | 各组标量指标的样本数、均值、标准差和范围 |
| `group_comparisons.csv` | 各实验组相对 Baseline 的差异 |
| `group_series.csv` | 按模拟日聚合的时序指标 |

当前聚合器提供描述统计，不自动选择显著性检验。正式论文的统计检验需在实验重复次数和分布特征确定后锁定。

## 6. 开题结果绘图

开题阶段的 Baseline 与 Analysis 图使用两份单次运行指标生成：

```bash
uv run --frozen python scripts/plot_opening_results.py \
  --baseline <baseline_metrics.json> \
  --analysis <analysis_metrics.json> \
  --output-dir outputs/evaluation/<evaluation_name>
```

模拟数据必须通过 `--watermark` 明确标记。真实实验图不得手工改写中间 CSV；需要变更口径时，应修改统一评价或绘图代码后重新生成。

## 7. 直接检查数据库

需要排查底层事实时，可将 SQLCipher 数据库解密到临时目录：

```bash
uv run --frozen python scripts/decode_db.py \
  <run_dir>/world.nmdb \
  -o outputs/tmp/world.db
```

解密副本仅用于本地复核，不提交仓库。数据库格式和密钥管理见 [数据库加密与访问边界](../engineering/database-encryption.md)。
