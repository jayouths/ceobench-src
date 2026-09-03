# AGENTS.md

本文件适用于整个仓库。任何自动化编码代理开始分析、修改、测试或运行实验前，都必须先阅读本文件，再阅读任务涉及的模块文档。

## 1. 必须遵守的研究边界

1. **不得让创新模块改变 CEO-Bench 的经营世界规则。** Analysis、Deliberation 和 Reflection 的实现只能改变 Agent 获取、组织和利用公开信息的方式，不能直接修改客户行为、收入、成本、竞争事件或状态转移规则。
2. **Baseline 必须保持原样。** `modules.analysis.enabled = false` 时，不得执行 Analysis 查询、创建 Analysis 客户端、生成战略简报或改变决策 Agent 的原始 Dashboard 上下文。后续模块也必须遵守同样的关闭即无影响原则。
3. **四个实验组采用固定的递进关系。**

   ```text
   baseline
   analysis
   analysis-deliberation
   full
   ```

   Deliberation 依赖 Analysis，Reflection 依赖前两个模块。不要为了形式上的完全组合而创建没有有效输入的 `deliberation-only` 或 `reflection-only`。
4. **默认难度承担主实验结论。** 可以统一调整模拟器参数建立低、中、高难度场景，但只能作为明确标记的敏感性或稳健性实验。同一场景内所有实验组必须使用相同规则，降低难度不能在查看结果后替代默认主结果。
5. **不得把模拟企业现金解释为真实利润。** API 成本是真实调用成本，CEO-Bench 现金是模拟经营指标。可以报告二者的关系，但必须明确语义不同。

## 2. 配置是实验的唯一入口

1. 新实验必须通过 `--config config/<experiment>.toml` 显式传入完整配置。不要重新增加模型、Provider 或实验参数的 CLI 覆盖项。
2. 决策 Agent、社交环境 LLM 和已启用创新模块的模型必须显式配置。代码不得为缺失的模型名称、端点或价格静默兜底。
3. `api_type`、模型名、最大输出 Token 和计价信息必须在 TOML 中明确。当前主实验统一使用 OpenAI SDK 及其兼容端点，协议显式选择 `openai_chat_completions` 或 `openai_responses`，不得根据 URL 猜测。
4. `reasoning_effort`、`temperature`、`top_p` 等可选参数遵循“未配置就不发送，配置后原样透传”。不得把未配置静默转换成 `none`、`low` 或供应商默认值。
5. 厂商私有参数只能通过 `request_options` 显式配置，并根据官方文档验证实际请求与返回。仅仅请求成功不代表参数已经生效。
6. 模型计价必须通过请求模型名和服务端返回模型名映射到明确的官方计价模型。缺少价格时应中止实验，不得套用其他模型的默认单价。
7. 恢复实验只能使用 `--resume <run_id-or-directory>`，并读取原运行目录中的 `config.json`。不得用当前 TOML 或 CLI 参数覆盖断点实验的原配置。

完整配置结构见 [config/config_template.toml](config/config_template.toml) 和 [docs/engineering/configuration.md](docs/engineering/configuration.md)。

## 3. 依赖与生成产物

1. 使用 `uv` 管理 Python 环境，安装依赖时使用：

   ```bash
   uv sync --frozen
   ```

2. **不得让依赖安装命令重写 `uv.lock`。** 切换镜像只允许改变下载来源，不能重新解析或更新锁文件。若锁文件被意外修改而依赖定义没有变化，应恢复锁文件，不需要重建 `.venv`。
3. 执行任何 `uv` 依赖命令后都要检查 `git diff -- uv.lock`。若差异只把公开 PyPI URL 改成公司或其他镜像 URL，必须立即恢复，禁止提交；除非 `pyproject.toml` 的依赖定义确实发生了预期变化，否则不得更新锁文件。
4. 确需更新依赖和锁文件时，当前 `uv` 应使用 `uv lock --no-config --default-index https://pypi.org/simple`。不要使用已废弃的 `UV_INDEX_URL` 覆盖本机 `default-index`，其优先级不足，仍可能把锁文件改写为公司镜像。
5. `src/` 是实现源代码，`public/` 是构建生成的 Agent 可见产物。不要把手工修改 `public/` 当作源代码修改。
6. MOCK 数据生成、一次性绘图、临时诊断和探索脚本统一放入不提交的 `outputs/tmp/`。验证完成后删除；只有确认会长期复用的通用能力才能进入 `src/` 或 `scripts/`。
7. 修改 Agent 可见的模拟器、API 客户端、工具说明或运行内容后，执行：

   ```bash
   uv run --frozen python scripts/build_public.py
   ```

8. 不涉及 Agent 可见内容的论文文档、结果设计或宿主侧日志改动，不需要重建 `public/`。

## 4. 代码目录职责

| 目录 | 职责 | 修改原则 |
|---|---|---|
| `src/saas_bench/agents/` | 主实验 Agent、Analysis 及后续创新模块 | 论文创新的主要实现位置 |
| `src/saas_bench/experiment/` | TOML 配置、Provider 协议、用量归一化和成本计算 | 同类模型调用问题必须在统一层解决 |
| `src/saas_bench/evaluation/` | Agent 不可见的实验事实、离线指标和结果导出 | 只观察经营状态，不得改变模拟规则或进入 Agent 查询结果 |
| `src/saas_bench/runtime/` | API 服务、数据库保护和宿主运行时 | 可以为主实验完善，但不得泄露隐藏状态 |
| `src/saas_bench/simulator/` | 企业经营环境、数据库和状态转移规则 | 主实验保持默认；只在明确的场景实验中统一调整 |
| `src/saas_bench/novamind_api/` | Agent 可使用的企业管理 API 客户端 | 视为 Agent 与世界的稳定接口，修改需评估公平性 |
| `src/saas_bench/legacy/` | 暂未进入主实验的扩展 Agent 和参考实现 | 不主动扩展或删除；任务明确涉及时再启用 |

修改功能设计、模块职责、实验含义或评价口径前，必须先与用户对齐。纯架构整理、命名改进和不改变行为的可读性重构可以直接进行，但不得顺带扩大到无关链路。

## 5. LLM 调用与成本口径

1. 当前主实验的统一兼容层位于 `src/saas_bench/experiment/llm_provider.py`。主链只配置 API 协议和端点，不增加恒定、无路由作用的 Provider 名称；新增协议差异时优先在该层归一化。
2. 决策 Agent 的 API `tool_choice` 与 Harness 的业务约束是两个层次。修改工具调用规则时必须同时检查请求参数、模型返回和 ReAct 循环行为。
3. 用量字段口径固定为：

   - `input_tokens`：计费总输入。
   - `cached_tokens`：输入 Token 中命中缓存的部分。
   - `output_tokens`：计费总输出，推理 Token 通常包含其中。
   - `reasoning_tokens`：独立观测字段，不从输出中扣除，也不得重复计费；供应商未上报时记录为 `null`，不得伪记为 `0`。

4. Agent 成本与模拟器环境 LLM 成本必须分开。Agent 侧至少按 `bash_agent`、`analysis`、`deliberation` 和 `reflection` 记录；不存在的模块不要提前伪造数据。
5. 不同模型的 Token 数不能直接合并成效率结论。跨模型比较应使用明确单价计算后的成本，并同时保留原始 Token 字段。

## 6. 实验产物与结果可信度

1. `world.nmdb` 是现金、订阅、客户、预测和服务状态等经营结果的权威来源。`ledger` 是现金计算的唯一权威账本。
2. `logs/trajectory_<run_id>.jsonl` 记录按真实顺序发生的原子事件；`logs/performance_<run_id>.jsonl` 记录周级、模块级和实验级聚合。不要让两个文件重复保存同一份大对象。
3. Analysis 每周产物固定保存在 `analysis/day_XXX/`：

   ```text
   signals.json
   role_reports.json
   state_portrait.json
   STRATEGY_BRIEF.md
   ```

4. 正式图表和 CSV 必须从 `outputs/runs/` 原始产物统一生成。不得手工填写结果，或在缺少原始 run ID 时声称结果可复现。
5. 单次运行的周度点不是独立重复实验。均值、标准差和标准误必须基于独立 run，不能用一个 run 内的时间窗口伪造重复样本。
6. 提前破产是实验结果，不能删掉、补零或外推到第 497 天。基础设施失败与 Agent 自身决策失败必须区分记录。
7. 结果设计和指标口径见 [docs/experiments/results/README.md](docs/experiments/results/README.md)；修改统计口径时同步更新 [指标目录](docs/experiments/results/metric-catalog.md)。

## 7. 隔离与隐藏状态

1. 普通决策 Agent 和三个创新模块只能通过公开 Dashboard、文档和工具了解经营世界。不得把数据库密钥、模拟器内部状态、实验评价事实、未来事件或宿主源码暴露给它们。Oracle 白盒上界实验可以读取模拟器内部状态和 `_eval_*` 实验事实。
2. 正式实验应在 Linux 沙箱中运行，禁止 Agent 任意联网，只开放完成实验所需的模拟器本地接口。macOS 本机调试不能被当作正式隔离证明。
3. 修改挂载、文件权限、网络或模拟器接口前，必须检查 Agent 是否能够读取 `world.nmdb`、密钥、宿主环境变量或实验配置中的敏感信息。
4. 数据库加密和威胁边界见 [docs/engineering/database-encryption.md](docs/engineering/database-encryption.md) 与 [docs/experiments/analyze-trajectory.md](docs/experiments/analyze-trajectory.md)。

## 8. 测试要求

1. 优先运行与改动模块对应的最小测试入口，不要因一个局部修改默认执行全量测试。
2. 跨模块改动、合并前和正式实验前执行完整回归。
3. 常用入口：

   ```bash
   make test
   make test-config
   make test-llm
   make test-agent
   make test-analysis
   make test-simulator
   make test-all
   ```

4. 外部模型 API 测试必须标记为 `external`，不能进入默认测试。未经用户明确要求，不运行可能产生费用的测试。
5. 新测试按 `unit`、`component`、`integration`、`system` 和 `external` 分层，并镜像生产模块。共享 Fake 和构造器放在 `tests/support/`。
6. 不为几乎不可能发生、且不影响实验可信度的极端状态堆积防御代码和测试。删除已有测试前必须确认对应功能确实无效或不再需要。

完整测试约定见 [docs/engineering/testing.md](docs/engineering/testing.md)。

## 9. 文档与注释

1. [README.md](README.md) 是项目和文档体系的统一入口。新增稳定文档后必须更新其中的文档导航。
2. Analysis 设计以 [docs/modules/analysis.md](docs/modules/analysis.md) 为准。修改模块输入、输出、流程或消融边界时必须同步该文档。
3. 关键业务边界、恢复逻辑、隔离逻辑、计费口径和不直观算法应保留简洁中文注释。不要为显而易见的赋值和流程增加旁白式注释。
4. 文档和代码不得声称尚未实现的 Deliberation、Reflection 或正式隔离能力已经完成。

## 10. Git 与交付

1. 不直接在 `main` 上开发，使用独立开发分支。不得修改或强推官方 `upstream`；个人 fork 的 `origin` 才是推送目标。
2. 工作区可能包含用户或其他任务的改动。不要回退、覆盖或提交与当前任务无关的内容。
3. 提交信息使用英文类型加中文说明，例如：

   ```text
   feat: 增加经营状态分析
   fix: 修复断点恢复配置读取
   refactor: 整理主实验日志结构
   docs: 完善实验结果设计
   test: 拆分模型调用测试
   ```

4. 一个提交应对应一个完整、可解释的逻辑改动。不要把每个微小字段调整拆成独立提交，也不要把无关修改混在一起。
5. 未经用户要求不要提交或推送。提交前说明改动面和验证结果；无法执行的测试必须明确报告。

## 11. 完成任务前检查

- 变更是否保持 Baseline 和模拟器主规则不变。
- 配置是否显式、可追溯，恢复是否继续使用原配置。
- Agent、创新模块和环境 LLM 的 Token 与成本是否正确分开。
- 是否需要重建并检查 `public/`。
- 是否运行了对应模块测试，是否需要完整回归。
- 是否同步更新设计文档、指标目录和文档导航。
- 是否误改、删除或提交了用户已有内容。
