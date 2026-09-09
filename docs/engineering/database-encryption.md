# 数据库加密与访问边界

## 1. 用途

每次运行的 `world.nmdb` 是 SQLCipher 加密的 SQLite 数据库。加密用于保护落盘产物、避免普通 SQLite 工具直接打开，并支持宿主侧持久化与恢复。

SQLCipher 不是正式实验的唯一隔离边界。正式隔离还依赖 Linux 沙箱、文件系统挂载、环境变量清理、网络限制，以及查询 API 的表级和字段级只读授权。即使数据库加密有效，也不能因此放宽这些边界。

## 2. 密钥来源

运行时由 `src/saas_bench/runtime/db_protection.py` 统一取得密钥：

1. 构建产物中的 `saas_bench.runtime._embedded_key._NMDB_KEY`；
2. 私有开发和测试环境中的 `NMDB_KEY` 环境变量。

缺少两者时直接失败，不使用默认密钥。密钥不得写入实验配置、日志或 Agent 进程环境。

当前 `novamind-operation` 会把编译后的密钥模块打包进 Agent 可见的公开 CLI。因此，SQLCipher 可以阻止误操作和普通 `sqlite3` 直接读取，但不能抵御专门提取可执行文件内容的 Agent。正式实验仍需依赖沙箱限制宿主文件和网络访问，并通过轨迹审计排查直接检查 CLI 或数据库的行为；不能把加密文件本身表述为不可突破的安全边界。

## 3. 可见性与职责

- 普通 Agent 和创新模块按实验协议只能通过公开工具读取授权后的经营数据，不应直接检查数据库文件、公开 CLI 内部或宿主源码。
- Oracle 通过显式白盒模式放宽查询可见性，但仍不能写数据库。
- 离线评价器在宿主侧读取完整数据库，用于复算指标和绘图，评价结果不得回流当次实验。

查询 API 对普通 Agent 只开放 `TABLE_DOCS` 登记的实体表，并在 SQLite 读取阶段拦截隐藏字段、未公开表、Schema 探测和写操作。Oracle 只绕过隐藏信息过滤，不绕过只读限制。

## 4. 本地复核

优先使用仓库脚本把数据库解密到不提交的临时目录：

```bash
uv run --frozen python scripts/decode_db.py \
  <run_dir>/world.nmdb \
  -o outputs/tmp/world.db
```

也可以在 Python 中直接读取：

```python
from saas_bench.runtime.db_protection import load_session_db

conn = load_session_db("<run_dir>/world.nmdb")
cash = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM ledger").fetchone()[0]
print(cash)
```

完整产物和评价流程见 [实验产物与离线评价](../experiments/artifacts-and-evaluation.md)。

## 5. 变更规则

- 不为普通实验手工轮换密钥；已有运行必须继续能够恢复和复核。
- 确需轮换时，应同步更新构建密钥、重新构建 `public/`，并明确标记旧产物所需的旧密钥。
- 修改密钥注入、沙箱挂载或查询授权后，必须重新验证普通 Agent 无法读取完整事实，而 Oracle 和离线评价仍可只读访问所需数据。
