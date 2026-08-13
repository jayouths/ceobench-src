# CEOBench CLI Reference

```text
usage: novamind-operation [-h]
                          {new-session,next-week,python,python-c,query,status,history,list-sessions,stop} ...

CEOBench Simulation CLI

positional arguments:
  {new-session,next-week,python,python-c,query,status,history,list-sessions,stop}
    new-session         Create a new simulation session
    next-week           Advance simulation by one week (7 days). Requires a
                        rationale string + 12 cash forecasts.
    python              Execute a Python script with novamind_api
    python-c            Execute inline Python code with novamind_api
    query               Execute a SQL query
    status              Get session status
    history             View action history
    list-sessions       List all sessions
    stop                Stop the simulation server

options:
  -h, --help            show this help message and exit

Examples:
  ./novamind-operation new-session --days 365 --seed 42
  ./novamind-operation next-week "Holding prices, raising ad spend on E1 to push enterprise pipeline"                                   1050000 1000000 1100000  1200000 1050000 1400000  1800000 1400000 2300000  3000000 2000000 4500000
                                  # rationale (required, non-empty) + 12 cash forecasts:
                                  # per horizon (+7d/+28d/+84d/+182d), submit point + 95% CI low/high
  ./novamind-operation python my_strategy.py
  ./novamind-operation python-c "import novamind_api as nm; nm.pricing.set_prices(A=25)"
  ./novamind-operation query "SELECT * FROM subscriptions LIMIT 10"
  ./novamind-operation status
  ./novamind-operation history --tail 20
  ./novamind-operation list-sessions
  ./novamind-operation stop
```

## `./novamind-operation new-session`

```text
usage: novamind-operation new-session [-h] [--days DAYS] [--seed SEED]
                                      [--cash CASH]

options:
  -h, --help   show this help message and exit
  --days DAYS  Total simulation days (default: 365)
  --seed SEED  Random seed (default: 42)
  --cash CASH  Initial cash (default: 1000000)
```

## `./novamind-operation next-week`

```text
usage: novamind-operation next-week [-h] [--session SESSION]
                                    rationale cash_1wk_point cash_1wk_lower
                                    cash_1wk_upper cash_4wk_point
                                    cash_4wk_lower cash_4wk_upper
                                    cash_12wk_point cash_12wk_lower
                                    cash_12wk_upper cash_26wk_point
                                    cash_26wk_lower cash_26wk_upper

Advance the simulation by 7 days. You MUST submit: 1. A rationale string (your
strategic reasoning for this week's actions, non-empty). 2. Cash forecasts at
four horizons (+7d, +28d, +84d, +182d). For EACH horizon submit a point
estimate plus 95% CI lower and upper bounds (lower <= point <= upper). 12
numbers total. Scored on point-percent-error, CI coverage, and sharpness at
each horizon. Rationale replaces the old standalone log_rationale tool — it is
now a required argument here.

positional arguments:
  rationale          Your strategic reasoning for this week's actions
                     (required, non-empty)
  cash_1wk_point     Point estimate of cash +7 days
  cash_1wk_lower     95% CI lower bound, +7 days
  cash_1wk_upper     95% CI upper bound, +7 days
  cash_4wk_point     Point estimate of cash +28 days
  cash_4wk_lower     95% CI lower bound, +28 days
  cash_4wk_upper     95% CI upper bound, +28 days
  cash_12wk_point    Point estimate of cash +84 days
  cash_12wk_lower    95% CI lower bound, +84 days
  cash_12wk_upper    95% CI upper bound, +84 days
  cash_26wk_point    Point estimate of cash +182 days (~6 months)
  cash_26wk_lower    95% CI lower bound, +182 days
  cash_26wk_upper    95% CI upper bound, +182 days

options:
  -h, --help         show this help message and exit
  --session SESSION  Session ID (default: latest)
```

## `./novamind-operation python`

```text
usage: novamind-operation python [-h] [--session SESSION] script

positional arguments:
  script             Path to Python script

options:
  -h, --help         show this help message and exit
  --session SESSION  Session ID (default: latest)
```

## `./novamind-operation python-c`

```text
usage: novamind-operation python-c [-h] [--session SESSION] code

positional arguments:
  code               Python code to execute

options:
  -h, --help         show this help message and exit
  --session SESSION  Session ID (default: latest)
```

## `./novamind-operation query`

```text
usage: novamind-operation query [-h] [--session SESSION] sql

positional arguments:
  sql                SQL query string

options:
  -h, --help         show this help message and exit
  --session SESSION  Session ID (default: latest)
```

## `./novamind-operation status`

```text
usage: novamind-operation status [-h] [--session SESSION]

options:
  -h, --help         show this help message and exit
  --session SESSION  Session ID (default: latest)
```

## `./novamind-operation history`

```text
usage: novamind-operation history [-h] [--session SESSION] [--tail TAIL]

options:
  -h, --help         show this help message and exit
  --session SESSION  Session ID (default: latest)
  --tail TAIL        Number of recent entries (default: 50)
```

## `./novamind-operation list-sessions`

```text
usage: novamind-operation list-sessions [-h]

options:
  -h, --help  show this help message and exit
```

## `./novamind-operation stop`

```text
usage: novamind-operation stop [-h] [--session SESSION]

options:
  -h, --help         show this help message and exit
  --session SESSION  Session ID (default: latest)
```
