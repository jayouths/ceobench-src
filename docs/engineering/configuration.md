# 实验配置

[`config/config_template.toml`](../../config/config_template.toml) 是当前主实验支持的完整配置模板，不代表任何具体模型或实验方案，不应直接运行。

每份配置必须通过 `experiment.name` 声明实验类型。运行目录将写入
`outputs/runs/<experiment.name>/<北京时间>_seed-<seed>_<run_id>/`。

开始一组新实验时，在 `config/` 目录创建对应的 TOML 文件，并显式传入：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.cli \
  --config config/<experiment>.toml
```

断点恢复不重新读取 TOML，只使用原运行目录中的 `config.json`：

```bash
uv run --frozen python -m saas_bench.agents.bash_agent.cli \
  --resume <run_id-or-directory>
```
