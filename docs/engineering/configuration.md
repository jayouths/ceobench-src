# 实验配置

## 1. 配置入口

[`config/config_template.toml`](../../config/config_template.toml) 是当前代码支持的完整配置模板，只说明字段结构，不代表具体实验方案，不应直接运行。

新实验必须创建独立 TOML，并通过命令行显式传入：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.cli \
  --config config/<experiment>.toml
```

配置中的 `experiment.name` 同时决定运行分组目录：

```text
outputs/runs/<experiment.name>/<北京时间>_seed-<seed>_<run_id>/
```

## 2. 配置块职责

| 配置块 | 内容 |
|---|---|
| `[experiment]` | 实验名称、随机种子、模拟天数、场景、初始现金、运行目录和决策轮次限制 |
| `[modules.analysis]` | Analysis 开关、Schema 修复次数、企业谈判明细上限和角色并发数 |
| `[models.decision_agent]` | 决策模型协议、模型名、端点、采样参数、工具选择、超时和输出上限 |
| `[models.social_llm]` | 社交环境模型及各类帖子任务覆盖参数 |
| `[models.analysis]` | Analysis 模型及角色报告、状态重构任务覆盖参数 |
| `pricing_model_map` | 请求模型名到官方计价模型的显式映射 |
| `pricing.<model>` | 官方币种和每百万 Token 单价 |
| `request_options` | 经官方文档确认后透传的厂商私有参数 |

## 3. 显式参数原则

- 决策 Agent、社交环境 LLM 和已启用模块的模型都必须显式配置。
- `api_type` 只允许代码已支持的协议，不根据端点 URL 猜测。
- 模型名必须通过枚举校验；缺少价格或计价映射时直接失败。
- 服务端回执模型名只用于审计，不改变配置中锁定的计价口径。
- `reasoning_effort`、`temperature`、`top_p` 和厂商私有参数未配置时不发送，不静默补成某个默认值。
- 模块关闭时不得创建对应模型客户端或产生调用成本。

正式实验配置应在注释中记录价格来源、生效日期及采用的高峰或低谷口径。密钥只通过 `api_key_env` 指向环境变量，不写入 TOML。

## 4. 恢复行为

恢复实验时只传运行 ID 或运行目录：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.cli \
  --resume <run_id-or-directory>
```

恢复过程只读取原运行目录中的 `config.json`，不会重新读取当前 TOML，也不允许通过 CLI 覆盖模型或实验参数。这样可以保证恢复前后的模型、价格、模块开关和模拟规则一致。

断点只支持同一 Git 版本的完整状态；随机流、客群随机状态或宏观状态缺失时中止，不使用初始默认值补齐。原世界快照不会在失败恢复时被覆盖。请求日志证明有动作尚未确认或已确认动作超出世界快照时，也拒绝恢复，不能通过回退旧断点重新执行。

服务生命周期由宿主 Runner 管理。独立公开 CLI 只会为 `created` 会话执行首次启动，已停止或不可恢复会话不会隐式重启；不提供绕过宿主断点预检的 Agent `resume` 命令。
