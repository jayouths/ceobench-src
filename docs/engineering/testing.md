# 测试体系

测试按三个维度组织：顶层目录表示测试层级，内部目录镜像生产模块，测试文件表示具体行为。

```text
tests/
├── unit/          # 单个函数或类，不使用真实外部 I/O
├── component/     # 单个组件，可使用 SQLite、临时目录等本地依赖
├── integration/   # 多个生产组件通过真实接口协作
├── system/        # 完整实验链路
└── external/      # 真实模型 API 等付费或联网测试
```

`slow`、`linux`、`bwrap` 和 `external` Marker 只描述运行条件，不用于表达模块归属。

## 固定入口

| 场景 | 命令 |
| --- | --- |
| 日常快速回归 | `make test` |
| 全部单元测试 | `make test-unit` |
| 全部组件测试 | `make test-component` |
| 实验配置 | `make test-config` |
| LLM 兼容和计费 | `make test-llm` |
| Bash Agent | `make test-agent` |
| Analysis 创新模块 | `make test-analysis` |
| Checkpoint | `make test-checkpoint` |
| 断点恢复 | `make test-resume` |
| API Server | `make test-api` |
| 模拟器规则 | `make test-simulator` |
| 跨组件集成 | `make test-integration` |
| 完整回归 | `make test-all` |
| 只检查收集数量 | `make test-collect` |

开发时优先运行被修改模块对应的入口。跨模块改动完成后、合并到主分支前和正式实验前，再执行完整回归。

## 编写规则

- 新测试放入对应层级和生产模块目录，不再继续扩充根目录中的历史测试文件。
- Unit 测试不得启动服务、访问网络或依赖真实时间。
- Component 测试只验证一个组件，跨组件真实通信放入 Integration。
- 外部 API 测试必须标记 `external`，不得进入默认入口。
- 共享 Fake 和构造器放入 `tests/support/`，禁止从其他测试文件导入辅助代码。
