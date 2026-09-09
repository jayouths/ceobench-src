# 文档导航

本目录只保存需要长期维护的工程、模块、实验和审计知识。临时诊断、模拟数据和一次性图表放入 `outputs/tmp/`，不进入文档体系。

## 目录职责

| 目录 | 维护内容 | 不维护的内容 |
|---|---|---|
| [`engineering/`](engineering/) | 当前代码架构、配置、测试和数据库访问边界 | 实验结论、历史问题流水账 |
| [`modules/`](modules/) | 已实现或已明确设计的论文创新模块 | 通用工程操作、结果图表 |
| [`experiments/`](experiments/) | 实验运行、产物、指标和结果呈现 | 模拟器实现细节 |
| [`audits/`](audits/) | 问题复核证据、处理状态和历史判断 | 当前工程行为的唯一说明 |

## 推荐阅读顺序

1. [工程架构](engineering/architecture.md)：先理解代码模块和数据边界。
2. [实验配置](engineering/configuration.md)：创建或恢复实验。
3. [Analysis 模块设计](modules/analysis.md)：理解当前已实现的创新模块。
4. [实验产物与离线评价](experiments/artifacts-and-evaluation.md)：复算指标和生成分析结果。
5. [实验结果设计](experiments/results/README.md)：查看研究问题、指标和图表方案。

## 权威范围

- 代码模块和可见性边界以 [工程架构](engineering/architecture.md) 为准。
- 具体模块输入、输出和消融行为以 `modules/` 下对应文档为准。
- 指标名称、公式和数据来源以 [实验指标目录](experiments/results/metric-catalog.md) 为准。
- 审计文档只记录“当时发现什么、如何判断、是否处理”，不替代上述当前设计文档。
- 实际运行配置最终以每个运行目录中的 `config.json` 为准，文档和模板不替代实验产物。
