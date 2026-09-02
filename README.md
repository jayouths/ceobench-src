# CEO-Bench 企业战略决策 Agent 实验仓库

本项目基于 [CEO-Bench](https://github.com/zlab-princeton/ceobench-src)，研究多 Agent 协同机制能否提升大语言模型在长期企业经营模拟中的战略决策能力。

CEO-Bench 模拟约 500 天的 SaaS 企业经营过程。决策 Agent 可以查询公开经营信息并调用管理工具，其行动会影响现金、收入、客户、服务质量和市场竞争状态。本项目的研究改动集中在 Agent Harness、实验配置、日志和结果分析层。

自动化编码代理修改本仓库前必须阅读 [AGENTS.md](AGENTS.md)。

## 研究设计

四个实验组采用递进消融：

| 实验组 | Analysis | Deliberation | Reflection |
|---|---:|---:|---:|
| `baseline` | 关闭 | 关闭 | 关闭 |
| `analysis` | 开启 | 关闭 | 关闭 |
| `analysis-deliberation` | 开启 | 开启 | 关闭 |
| `full` | 开启 | 开启 | 开启 |

- **Analysis**：提取结构化经营信号，通过市场、财务、产品和客户角色重构经营状态，形成战略简报。
- **Deliberation**：围绕当前经营状态协同推演方案，处理行动依赖和角色冲突。
- **Reflection**：验证历史预测与行动结果，将经验反馈给后续推演。

当前优先完成和验证 Analysis，开题阶段只比较 `baseline` 与 `analysis`。

## 研究边界

主实验保持 CEO-Bench 默认经营规则，以保证 Baseline 和创新组可公平比较，并保留与上游结果的可比性。

后续可以设置统一的低、中、高难度场景作为敏感性或稳健性检验。所有实验组必须使用相同场景参数，场景结果与默认主实验分开报告。若默认环境导致所有模型快速破产，较低难度场景可以用于观察模块机制，但不能在查看结果后替代默认主结果。

模拟器难度参数位于 `src/saas_bench/simulator/config.py`。修改 Agent 可见的模拟器内容后，需要重新构建 `public/`：

```bash
uv run --frozen python scripts/build_public.py
```

## 运行实验

安装锁定版本依赖：

```bash
uv sync --frozen
```

根据 [配置模板](config/config_template.toml) 创建实验配置并启动：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.cli \
  --config config/<experiment>.toml
```

恢复已有实验：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.cli \
  --resume <run_id-or-directory>
```

配置说明见 [实验配置](docs/engineering/configuration.md)。

## 实验产物

运行结果写入：

```text
outputs/runs/<experiment_name>/<北京时间>_seed-<seed>_<run_id>/
```

| 产物 | 内容 |
|---|---|
| `result.json` | 最终经营结果和累计模型用量 |
| `world.nmdb` | 经营状态与完整账本数据库 |
| `config.json` | 本次运行的完整配置 |
| `checkpoint.json` | 断点恢复状态 |
| `analysis/day_XXX/` | Analysis 的信号、角色报告、经营画像和战略简报 |
| `logs/trajectory_<run_id>.jsonl` | 按顺序记录 LLM、工具和周边界事件 |
| `logs/performance_<run_id>.jsonl` | 周度模型调用、模块性能和实验级成本汇总 |

结果提取方式见 [运行产物分析指南](docs/experiments/analyze-trajectory.md)。

## 测试

```bash
make test       # 日常快速测试
make test-all   # 正式实验前完整回归
```

`Makefile` 还提供配置、模型调用、Agent、Analysis、模拟器和恢复机制的分模块测试入口。

## 项目结构

| 目录 | 作用 |
|---|---|
| `config/` | 实验 TOML 配置模板和具体配置 |
| `docs/` | 配置、测试、模块设计、产物分析和论文结果设计 |
| `outputs/` | 本地实验产物，不提交版本库 |
| `public/` | Agent 可见的工具、API 和公开说明 |
| `scripts/` | `public/` 构建、文档生成和数据库解密脚本 |
| `src/saas_bench/agents/` | 当前主实验 Agent 和创新模块 |
| `src/saas_bench/experiment/` | 实验配置、模型调用和成本计算 |
| `src/saas_bench/runtime/` | API 服务、数据库保护和运行时能力 |
| `src/saas_bench/simulator/` | 企业经营模拟规则 |
| `src/saas_bench/novamind_api/` | Agent 可调用的企业管理 API 客户端 |
| `src/saas_bench/legacy/` | 保留的扩展 Agent 和参考实验实现 |
| `tests/` | 单元、组件、集成、系统和外部 API 测试 |

## 文档导航

`README.md` 是项目的人类文档入口；`AGENTS.md` 专供自动化编码代理读取。其他需要长期维护的说明统一放在 `docs/`。

```text
docs/
├── engineering/                      # 工程实现和开发约定
│   ├── code-structure.md            # 代码模块职责和依赖边界
│   ├── configuration.md             # 实验配置使用方式
│   ├── testing.md                   # 测试分层、入口和编写规则
│   └── database-encryption.md       # world.nmdb 加密与解密说明
├── modules/                          # 论文创新模块设计
│   └── analysis.md                  # Analysis 模块设计
└── experiments/                      # 实验分析与论文结果
    ├── analyze-trajectory.md             # 运行产物分析指南
    └── results/                          # 指标与结果呈现设计
        ├── README.md                      # 结果设计总览
        ├── metric-catalog.md             # 统一指标目录
        ├── plans/                         # 分阶段的结果方案
        │   ├── opening.md                 # 开题阶段结果方案
        │   └── final.md                   # 最终论文结果方案
        └── figures/                       # 两套方案共用的候选图表设计
```

### 核心文档

| 文档 | 作用 |
|---|---|
| [实验配置](docs/engineering/configuration.md) | 说明配置模板、新实验启动和断点恢复方式 |
| [测试体系](docs/engineering/testing.md) | 说明测试分层、固定命令入口和编写规则 |
| [代码结构](docs/engineering/code-structure.md) | 说明 `src/saas_bench/` 中各模块职责、Harness 文件分工和 Legacy 边界 |
| [Analysis 模块设计](docs/modules/analysis.md) | 记录第一个创新模块的数据边界、执行流程和消融设计 |
| [运行产物分析指南](docs/experiments/analyze-trajectory.md) | 说明如何解密和分析 `world.nmdb` 及运行日志 |
| [数据库加密](docs/engineering/database-encryption.md) | 记录 SQLCipher 的使用方式和安全边界 |
| [实验结果设计](docs/experiments/results/README.md) | 汇总研究问题、候选指标和图表呈现方案 |

### 结果设计文档

| 文档 | 作用 |
|---|---|
| [指标目录](docs/experiments/results/metric-catalog.md) | 统一维护指标优先级、计算口径、数据来源和适用阶段 |
| [最终论文方案](docs/experiments/results/plans/final.md) | 规划四个递进实验组的结果分析和证据组合 |
| [开题阶段方案](docs/experiments/results/plans/opening.md) | 收敛 `baseline` 与 `analysis` 当前可以回答的问题和图表 |
| [候选图表](docs/experiments/results/figures/) | 按研究内容分开记录指标、数据来源、图形和保留条件 |

当前开发和开题阶段，建议依次阅读 [Analysis 模块设计](docs/modules/analysis.md)、[指标目录](docs/experiments/results/metric-catalog.md)、[开题阶段方案](docs/experiments/results/plans/opening.md) 和 [运行产物分析指南](docs/experiments/analyze-trajectory.md)。

文档维护遵循三条原则：稳定的跨模块知识才写入 `docs/`；指标口径统一在 `metric-catalog.md` 维护；实验原始数据和生成结果只放入 `outputs/runs/`。

## 上游出处

- 论文：[CEO-Bench: Can Agents Play the Long Game?](https://arxiv.org/abs/2606.18543)
- 官方仓库：[zlab-princeton/ceobench-src](https://github.com/zlab-princeton/ceobench-src)

```bibtex
@misc{chen2026ceobenchagentsplaylong,
  title={CEO-Bench: Can Agents Play the Long Game?},
  author={Haozhe Chen and Karthik Narasimhan and Zhuang Liu},
  year={2026},
  eprint={2606.18543},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2606.18543}
}
```
