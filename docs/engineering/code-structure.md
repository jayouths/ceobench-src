# 代码结构

## 主包职责

| 目录 | 职责 | 是否属于当前主实验 |
|---|---|---|
| `src/saas_bench/simulator/` | 经营状态、数据库、每日推进规则、客户行为和经营动作 | 是 |
| `src/saas_bench/runtime/` | API Server、公开 CLI、会话管理、数据库保护和文档生成 | 是 |
| `src/saas_bench/experiment/` | TOML 实验配置、模型 Provider 接入和实验文件写入 | 是 |
| `src/saas_bench/agents/` | 当前 Bash Agent、Analysis 创新模块和运行 Harness | 是 |
| `src/saas_bench/novamind_api/` | Agent 在隔离环境内调用模拟器的公开 SDK | 是 |
| `src/saas_bench/legacy/` | 暂未使用的旧 Agent、消融、Replay 和参考材料 | 否 |

主实验依赖方向为 `agents -> experiment/runtime -> simulator`。`legacy` 不得被主实验导入。

## Bash Agent Harness

| 路径 | 职责 |
|---|---|
| `src/saas_bench/agents/bash_agent/cli.py` | 仅解析 `--config` 和 `--resume` 命令行参数 |
| `src/saas_bench/agents/bash_agent/run_config.py` | 创建新实验或从原配置恢复 Runner |
| `src/saas_bench/agents/bash_agent/runner.py` | 主实验生命周期和每周决策循环 |
| `src/saas_bench/agents/bash_agent/simulator_server.py` | 模拟器会话、服务进程、HTTP 和 Unix Socket |
| `src/saas_bench/agents/bash_agent/workspace.py` | Agent 可写工作区的 Git 时间线和公开文件 |
| `src/saas_bench/agents/bash_agent/checkpoint.py` | 数据库、对话、日志边界等断点文件 |
| `src/saas_bench/agents/bash_agent/experiment_logs.py` | 轨迹日志与性能日志 |
| `src/saas_bench/agents/bash_agent/analysis/` | Analysis 创新模块的数据、提示词与编排 |

`runner.py` 只负责决定何时调用各组件，不应重新实现文件持久化、进程管理或模拟器经营规则。

## Legacy 边界

`src/saas_bench/legacy/` 保存暂未纳入当前主实验的旧实现和参考材料：

- `agents/`：旧 Agent、旧消融入口和 Replay 实现。
- `reference/`：已经被当前实现替代的提示词模板和工具文档。

这些内容仅用于追溯和后续重新设计，不保证可以直接运行。重新启用时，应先迁回正式模块并补齐测试。
