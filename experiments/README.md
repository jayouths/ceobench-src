# 实验运行与模型调用配置

`*.toml` 只配置实验参数和三类 LLM 参数，不修改 CEO-Bench 的经济规则、客户行为规则和评测逻辑。

启动实验时必须通过 `--config` 显式指定 TOML。以下三个配置块都必须存在，并且必须分别填写 `provider` 和 `model`：

```toml
[models.decision_agent]
[models.social_llm]
[models.enterprise_llm]
```

代码不提供任何模型或 Provider 的兜底值。缺少模型配置时，程序会在实验启动前直接报错。

初步消融实验（70 天，10 周）：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.run_test \
  --config experiments/experiment.toml
```

本地一周 Smoke Test：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.run_test \
  --config experiments/smoke.toml
```

最终完整实验（500 天，Runner 实际执行到最近的完整周）：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.run_test \
  --config experiments/full.toml
```

配置优先级：

```text
CLI 显式覆盖 > TOML 中的模型身份与实验参数 > 通用调用参数默认值
```

API Key 不写入 TOML。`api_key_env` 保存环境变量名，实际密钥保存在仓库根目录的 `.env` 或系统环境变量中。
本地无鉴权端点必须显式配置 `api_key_required = false`；付费 API 保持默认值 `true`。

`input_cost_per_million` 和 `output_cost_per_million` 记录每百万 Token 的价格。本地模型填写 `0.0`；付费 API 应填写实验运行时的真实价格。

`reasoning_effort` 省略时，请求不发送 `reasoning` 参数；写为字符串 `"none"` 时，会向 API 显式传递 `none`。
