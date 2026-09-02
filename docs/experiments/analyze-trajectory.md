# 📈 Analyzing an Agent Trajectory

Each finished CEO-Bench run leaves two primary result sources:

```
<run_dir>/world.nmdb
<run_dir>/logs/performance_<run_id>.jsonl
```

`world.nmdb` is a **SQLCipher** database (page-level AES-256 encrypted SQLite)
and is the authoritative record of the simulated business. Decrypt it to
analyze cash, subscriptions, customers, service quality, and business events.
The performance log records runtime measurements such as model usage, API
cost, and elapsed time; it is not a business-state data source.

---

## 🔑 Decrypt and open

The SQLCipher key is fixed and bundled into the published `novamind-operation`
zipapp. The value is in `docs/engineering/database-encryption.md`, or import it from the
compiled `saas_bench.runtime._embedded_key` module.

**One-shot decrypt to plain SQLite:**

```bash
uv run --frozen python scripts/decode_db.py <run_dir>/world.nmdb -o /tmp/world.db
sqlite3 /tmp/world.db ".tables"
```

`scripts/decode_db.py` also supports `--dump` (JSON), `--csv-dir`, `--shell`,
and `--summary`.

**Or query it directly in Python (no plaintext copy on disk):**

```python
from pathlib import Path
from saas_bench.runtime.db_protection import load_session_db

conn = load_session_db(Path("<run_dir>/world.nmdb"))   # in-memory sqlite3 conn
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()
print([t[0] for t in tables])
```

---

## 🧭 What's in the database

Run `.schema <table>` in `sqlite3` (or `PRAGMA table_info(<table>)`) for exact
column types. The tables most useful for trajectory analysis:

| Table                     | What it holds                                                      |
|---------------------------|--------------------------------------------------------------------|
| `ledger`                  | Every income/cost event; the source of truth for cash-on-hand      |
| `config_history`          | Daily snapshot of every agent-configurable setting (prices, tiers, spend, quotas) |
| `predictions`             | Cash forecasts the agent submitted at each `next-week` call (point + 95% CI, 4 horizons) |
| `subscriptions`           | Customer subscriptions, current + historical, with plan/price/status |
| `customers`               | All customers with persona, company profile, acquisition source, group |
| `_eval_subscription_day`  | Private daily account, seat, and MRR stock by customer type, segment, and plan |
| `_eval_subscription_event`| Private subscription start/end events with event-time seats and MRR |
| `_eval_segment_day`       | Private daily segment state, customer health, and actual lead-generation inputs/results |
| `_eval_quality_day`       | Private daily delivered-quality components by segment and plan |
| `_eval_channel_effectiveness_event` | Private initial/monthly channel effectiveness state changes |
| `enterprise_turns`        | Enterprise negotiation messages: offers, seat counts, close reason |
| `ad_channel_leads`        | Per-(day, channel, group) leads generated and ad spend             |
| `daily_usage`             | Per-customer daily usage units                                    |
| `research_projects`       | R&D tier invocations: status and quality boost                    |
| `service_day`             | Daily service metrics (quality, uptime, capacity)                 |
| `social_media_posts`      | Public customer feedback posts                                    |
| `agent_social_media_posts`| Social posts the agent authored                                  |
| `notifications`           | The agent's inbox: alerts, competitor moves, events               |
| `macroeconomic_conditions`| PMI business-cycle index over time                               |

The `ledger` table is the most important; every monetary event flows through
it:

| column     | type    | meaning                                                              |
|------------|---------|----------------------------------------------------------------------|
| `day`      | INTEGER | Simulated day of the entry (day 0 = initial funding)                 |
| `category` | TEXT    | `subscription_payment`, `compute`, `capacity`, `advertising`, `operations`, `development`, `lead_acquisition_cost`, `initial_funding`, `market_research`, `group_research`, `research_project`, `ad_revenue` |
| `amount`   | REAL    | Positive for income, negative for cost                               |
| `note`     | TEXT    | Free-form description                                                |

**Cash-on-hand on day _d_** is the running sum of `amount` over all entries
with `day ≤ d`. The headline benchmark score is the final cash:

```sql
SELECT COALESCE(SUM(amount), 0) AS final_cash FROM ledger;
```

---

## 📊 Analysis examples

**Per-day cash trajectory:**

```sql
WITH daily AS (
    SELECT
        day,
        SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS revenue,
        SUM(CASE WHEN amount < 0 THEN -amount ELSE 0 END) AS costs,
        SUM(amount) AS net_change
    FROM ledger
    GROUP BY day
)
SELECT
    day, revenue, costs, net_change,
    SUM(net_change) OVER (ORDER BY day) AS cash_on_hand
FROM daily
ORDER BY day;
```

**Cash trajectory in pandas:**

```python
import pandas as pd
from saas_bench.runtime.db_protection import load_session_db

conn = load_session_db("<run_dir>/world.nmdb")
ledger = pd.read_sql_query("SELECT day, category, amount FROM ledger", conn)

daily = (
    ledger.groupby("day", as_index=False)["amount"]
          .sum()
          .rename(columns={"amount": "net_change"})
          .sort_values("day")
)
daily["cash_on_hand"] = daily["net_change"].cumsum()
print(daily.tail())
```

**Spend breakdown by category:**

```python
spend = (
    ledger[ledger.amount < 0]
    .groupby("category")["amount"].sum().abs()
    .sort_values(ascending=False)
)
print(spend)
```

**Daily subscription and recurring-revenue trajectory:**

```sql
SELECT
    day,
    SUM(mrr) AS mrr,
    SUM(CASE WHEN customer_type = 'small' THEN active_accounts ELSE 0 END)
        AS individual_subscriptions,
    SUM(CASE WHEN customer_type = 'large' THEN active_seats ELSE 0 END)
        AS enterprise_seats
FROM _eval_subscription_day
GROUP BY day
ORDER BY day;
```

`_eval_subscription_day` stores end-of-day stock. `_eval_subscription_event`
stores actual transitions into and out of the active subscription state, so
gross acquisition and churn do not need to be inferred from mutable rows.
All `_eval_*` tables are unavailable to normal agents and innovation modules;
they are intended for post-run analysis and the explicit Oracle upper-bound
experiment only.

**Forecast accuracy**: compare what the agent predicted at each `next-week`
call against the cash it actually had at that horizon. Each submission inserts
4 rows (`horizon_days` = 7, 28, 84, 182); the target day is
`submit_day + horizon_days`:

```python
preds  = pd.read_sql_query("SELECT * FROM predictions", conn)
actual = daily.set_index("day")["cash_on_hand"]

preds["target_day"] = preds["submit_day"] + preds["horizon_days"]
preds["actual"]     = preds["target_day"].map(actual)
preds["pct_error"]  = (preds["predicted_value"] - preds["actual"]) / preds["actual"]
print(preds[["submit_day", "horizon_days", "predicted_value", "actual", "pct_error"]])
```

**Replaying the agent's decisions**: `config_history` has one row per day with
every agent-configurable setting (prices, tiers, per-category spend, ad spend
per channel, quotas), so diffing consecutive rows shows exactly when and how the
agent changed strategy:

```python
cfg = pd.read_sql_query("SELECT * FROM config_history ORDER BY day", conn)
changes = cfg[cfg.drop(columns="day").diff().ne(0).any(axis=1)]
print(changes)
```

---

## 🛡️ Preventing the agent from cheating

`world.nmdb` is encrypted so the agent **cannot easily access information that
is hidden from it**: hidden customer segments, upcoming competitor events,
future market shocks. Vanilla `sqlite3` (the only SQLite the agent's sandbox
has) cannot open a SQLCipher file at all, so the agent must learn the world by
acting in it rather than by reading the ledger.

The catch: the decryption key is **bundled inside the `novamind-operation`
zipapp** (so the engine can write the ledger). An agent with shell access on
the same machine could unpack the zipapp, extract the key, and decrypt
`world.nmdb` to peek ahead.

**To be safe, remove the agent's read access to the `novamind-operation`
zipapp.** Run it behind a wrapper or proxy so the agent can invoke the CLI
without being able to `cat`, `unzip`, or `python -m zipfile` the executable
itself. Without read access to the zipapp, the agent has no path to the key
and the encrypted ledger stays sealed.
