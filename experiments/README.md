# 实验运行与模型调用配置

`*.toml` 只配置实验参数和两类 LLM 参数，不修改 CEO-Bench 的经济规则、客户行为规则和评测逻辑。企业客户谈判由官方的结构化规则处理，不调用 LLM。

`experiment.max_decision_turns_per_batch` 限制单个决策批次的 Agent 行动次数。达到上限仍未执行 `next-week` 时，本次调用会保存可恢复断点并以 `incomplete` 结束，不会替 Agent 强制推进环境。

`experiment.max_invalid_responses_per_turn` 限制同一次行动中连续出现无工具调用或非法工具参数的次数。达到上限后实验直接失败，避免模型异常时无限请求。

启动新实验时不接受 CLI 配置参数，程序固定读取 `experiments/experiment.toml`。以下两个配置块都必须存在，并且必须分别填写 `provider`、`api_type`、`model` 和 `pricing`：

```toml
[models.decision_agent]
[models.social_llm]
```

代码不提供模型、Provider 或输出上限的兜底值。两类模型都必须显式填写 `max_output_tokens`，缺少时程序会在实验启动前直接报错。任务级 `max_output_tokens` 可以省略，此时继承所属模型的上限。

初步消融实验（70 天，10 周）：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.run_test
```

`experiments/smoke.toml` 和 `experiments/full.toml` 是一周连通性测试与 500 天完整实验的配置模板。需要切换实验方案时，先把相应内容写入 `experiments/experiment.toml`，再使用上面的无参数命令启动；不能通过 CLI 覆盖实验或模型配置。

本地一周 Smoke Test 配置模板：

```bash
cp experiments/smoke.toml experiments/experiment.toml
uv run --frozen python -m saas_bench.agents.bash_agent.run_test
```

最终完整实验配置模板（500 天，Runner 实际执行到最近的完整周）：

```bash
cp experiments/full.toml experiments/experiment.toml
uv run --frozen python -m saas_bench.agents.bash_agent.run_test
```

断点恢复只指定原实验的 run id 或 run 目录：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.run_test \
  --resume <run_id>
```

恢复时程序读取原目录中的 `config.json`，不读取当前 `experiments/experiment.toml`，也不比较配置是否变化。CLI 仅保留 `--resume`，不支持覆盖实验或模型参数。

`config.json` 会记录实验首次启动时的 `harness_git_commit`、`harness_git_dirty` 和 `harness_source_sha256`。恢复时如果当前 Harness 源码哈希与首次启动不同，程序会警告但继续使用当前代码；`result.json` 记录本次实际执行使用的 Harness 身份。正式论文实验应在 `harness_git_dirty = false` 时启动。

`provider` 只表示客户端类型：`openai`、`openai_compatible`、`anthropic` 或 `bedrock`。Google、Together、xAI、Ollama 等兼容服务统一使用 `openai_compatible`，并显式配置 `base_url` 和 `api_type`。`api_type` 可选 `openai_responses`、`openai_chat_completions` 或 `anthropic_messages`。

API Key 不写入 TOML。`api_key_env` 保存环境变量名，实际密钥保存在仓库根目录的 `.env` 或系统环境变量中。
本地无鉴权端点必须显式配置 `api_key_required = false`；付费 API 保持默认值 `true`。

`pricing` 按服务端实际返回的模型名记录每百万 Token 价格。本地模型填写 `0.0`；可能发生模型 fallback 时，必须把所有可能返回的模型都写入价格表。服务端返回未知模型时实验直接失败。

项目产物采用统一的内部用量口径：`input_tokens` 表示本次计费涉及的总输入量，`cached_tokens` 是其中命中缓存的部分，`output_tokens` 表示按输出单价计费的总输出量。推理 Token 通常已包含在 `output_tokens` 中，只单独记录用于分析，不重复计费。

这不是所有供应商的原始 API 口径。OpenAI 和 DeepSeek 的返回基本可以直接映射；Anthropic 原始 `input_tokens` 不包含缓存读取量，兼容层会将缓存读取量合并进项目的 `input_tokens`，并同时记为 `cached_tokens`。未来接入 Gemini 原生 API 或其他 Provider 时，必须依据该供应商当时的官方计费说明单独适配，不能仅按字段名推断包含关系。

TODO：当前价格模型尚未支持 Anthropic 独立的缓存写入价格及缓存期限档位。检测到 `cache_creation_input_tokens > 0` 时程序会直接报错，不会将其误算为普通输入成本；正式使用 Anthropic Prompt Caching 前必须先扩展价格配置和计费公式。

计费时，未命中量 `input_tokens - cached_tokens` 使用 `uncached_input_cost_per_million`，命中量使用 `cached_input_cost_per_million`；输出按 `output_cost_per_million` 计算。所有金额只记录供应商实际结算币种，不在实验过程中换算汇率。

```toml
[models.decision_agent.pricing."qwen3-coder:30b"]
currency = "CNY"
uncached_input_cost_per_million = 0.0
cached_input_cost_per_million = 0.0
output_cost_per_million = 0.0
```

模型级标准参数包括 `reasoning_effort`、`temperature`、`top_p`、`max_output_tokens` 和 `timeout_seconds`。社交 LLM 可在 `[models.social_llm.tasks.<task>]` 中覆盖前四项。供应商私有参数放入 `request_options`；OpenAI 兼容协议可使用 `extra_body`、`extra_headers` 和 `extra_query`，Anthropic Messages 可使用原生 `thinking` 和 `output_config`。

`reasoning_effort` 省略时不发送对应参数；显式配置时按 OpenAI Responses 或 Chat Completions 协议映射。Anthropic Messages 不使用该抽象字段；如需 thinking，必须在 `request_options` 中显式填写原生参数。
