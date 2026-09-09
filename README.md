# CEO-Bench 企业战略决策 Agent 实验仓库

本项目基于 [CEO-Bench](https://github.com/zlab-princeton/ceobench-src)，研究战略增强 Agent Harness 能否提升大语言模型在长期企业经营模拟中的决策能力。

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

结果提取方式见 [实验产物与离线评价](docs/experiments/artifacts-and-evaluation.md)。

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
| `src/saas_bench/evaluation/` | Agent 不可见的实验事实采集、指标计算和结果导出 |
| `src/saas_bench/runtime/` | API 服务、数据库保护和运行时能力 |
| `src/saas_bench/simulator/` | 企业经营模拟规则 |
| `src/saas_bench/novamind_api/` | Agent 可调用的企业管理 API 客户端 |
| `src/saas_bench/legacy/` | 保留的扩展 Agent 和参考实验实现 |
| `tests/` | 单元、组件、集成、系统和外部 API 测试 |

## 文档

[文档导航](docs/README.md) 统一说明工程、创新模块、实验和审计文档的职责及推荐阅读顺序。常用入口：

- [工程架构](docs/engineering/architecture.md)
- [实验配置](docs/engineering/configuration.md)
- [Analysis 模块设计](docs/modules/analysis.md)
- [实验产物与离线评价](docs/experiments/artifacts-and-evaluation.md)
- [实验结果设计](docs/experiments/results/README.md)

`README.md` 只提供项目总览；具体设计和口径以 `docs/` 中对应的权威文档为准。自动化编码代理还必须遵守 [AGENTS.md](AGENTS.md)。

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
