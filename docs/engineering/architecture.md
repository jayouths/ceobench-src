# 工程架构

## 1. 主包职责

| 目录 | 职责 | 当前主实验是否依赖 |
|---|---|---:|
| `src/saas_bench/simulator/` | 经营状态、数据库、每日推进规则、客户行为和经营动作 | 是 |
| `src/saas_bench/runtime/` | API 服务、查询授权、数据库保护和宿主运行时 | 是 |
| `src/saas_bench/experiment/` | 实验配置、模型协议、用量归一化和成本计算 | 是 |
| `src/saas_bench/evaluation/` | 私有评价事实、离线指标、跨运行聚合和结果导出 | 是 |
| `src/saas_bench/agents/` | 决策 Agent、Analysis 模块和运行 Harness | 是 |
| `src/saas_bench/novamind_api/` | Agent 调用企业管理能力的公开客户端 | 是 |
| `src/saas_bench/legacy/` | 暂未纳入主实验的旧 Agent、消融和参考实现 | 否 |

主依赖方向为：Agent Harness 调用公开客户端和运行时接口，运行时驱动模拟器；评价层旁路读取实验事实。`evaluation` 可以观察，但不能改变经营状态，也不能把私有事实注入普通 Agent 或创新模块。`legacy` 不得被当前主实验导入。

## 2. 数据边界

模拟世界中的信息分为三层：

| 信息层 | 用途 | 普通 Agent / Analysis | Oracle | 离线评价器 |
|---|---|---:|---:|---:|
| 公开经营数据 | Dashboard、经营查询和工具决策 | 可读 | 可读 | 可读 |
| 模拟器内部状态 | 驱动客户、市场和随机过程 | 不可读 | 可读 | 按需要读取 |
| `_eval_*` 实验评价事实 | 复盘、指标计算和绘图 | 不可读 | 可读 | 可读 |

普通 Agent 和 Analysis 具有相同的数据权限。Oracle 是显式白盒上界，只放宽信息可见性，不获得写数据库的能力。离线评价只发生在运行之外，其结果不得反馈给同一次经营决策。

该表描述公开 API 的逻辑可见性。客户端数据库加密的实际威胁边界另见 [数据库加密与访问边界](database-encryption.md)。

## 3. Bash Agent Harness

| 路径 | 职责 |
|---|---|
| `agents/bash_agent/cli.py` | 解析 `--config` 和 `--resume` |
| `agents/bash_agent/run_config.py` | 创建新实验或从原配置恢复 Runner |
| `agents/bash_agent/runner.py` | 编排实验生命周期和每周 ReAct 决策循环 |
| `agents/bash_agent/simulator_server.py` | 管理模拟器服务进程、HTTP 和 Unix Socket |
| `agents/bash_agent/workspace.py` | 管理 Agent 可写工作区和 Git 时间线 |
| `agents/bash_agent/checkpoint.py` | 保存和恢复数据库、对话与日志边界 |
| `agents/bash_agent/experiment_logs.py` | 写入轨迹日志和性能日志 |
| `agents/bash_agent/analysis/` | 实现 Analysis 的信号、角色报告、状态重构和简报 |

`runner.py` 只负责编排调用时机，不应重新实现文件持久化、服务进程管理或模拟器经营规则。Analysis 的详细数据流见 [Analysis 模块设计](../modules/analysis.md)。

## 4. Legacy 边界

`src/saas_bench/legacy/` 中的内容只用于追溯和后续设计，不保证能够直接运行：

- `agents/`：旧 Agent、旧消融入口和 Replay 实现；
- `reference/`：已经被当前实现替代的提示词模板和工具文档。

确认某项能力重新进入主实验时，应将其迁回正式模块、接入统一配置和日志，并补齐相应测试；不要直接从 `legacy` 导入。
