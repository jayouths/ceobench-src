#!/usr/bin/env python3
"""Decode a NovaMind session database for analysis.

This tool is NOT included in the public repo. It's for private analysis
of simulation runs — viewing all tables, hidden columns, internal state.

Usage:
    # Decode .nmdb to plain SQLite
    uv run --frozen python scripts/decode_db.py sessions/<id>/world.nmdb -o decoded.db

    # Decode and dump all tables as JSON
    uv run --frozen python scripts/decode_db.py sessions/<id>/world.nmdb --dump

    # Decode and dump specific tables
    uv run --frozen python scripts/decode_db.py sessions/<id>/world.nmdb --dump --tables customer_state,ledger

    # Decode and open interactive sqlite3 shell
    uv run --frozen python scripts/decode_db.py sessions/<id>/world.nmdb --shell

    # Export all data as CSV files
    uv run --frozen python scripts/decode_db.py sessions/<id>/world.nmdb --csv-dir output_csvs/

    # Summary statistics
    uv run --frozen python scripts/decode_db.py sessions/<id>/world.nmdb --summary
"""

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from saas_bench.db_protection import unprotect_db


def get_all_tables(conn: sqlite3.Connection) -> list:
    """Get all table names."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row[0] for row in cursor.fetchall()]


def get_table_columns(conn: sqlite3.Connection, table: str) -> list:
    """Get column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def dump_table(conn: sqlite3.Connection, table: str, limit: int = None) -> dict:
    """Dump a table as a dict with columns and rows."""
    columns = get_table_columns(conn, table)
    query = f"SELECT * FROM {table}"
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    return {
        "table": table,
        "columns": columns,
        "row_count": count,
        "rows_shown": len(rows),
        "rows": rows,
    }


def print_summary(conn: sqlite3.Connection):
    """Print a summary of all tables and their row counts."""
    tables = get_all_tables(conn)

    print(f"\n{'='*60}")
    print(f"NovaMind Database Summary")
    print(f"{'='*60}")
    print(f"\n{'Table':<35} {'Rows':>10}  {'Columns':>8}")
    print(f"{'-'*35} {'-'*10}  {'-'*8}")

    total_rows = 0
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols = len(get_table_columns(conn, table))
        total_rows += count
        print(f"{table:<35} {count:>10,}  {cols:>8}")

    print(f"{'-'*35} {'-'*10}  {'-'*8}")
    print(f"{'TOTAL':<35} {total_rows:>10,}  {len(tables):>8} tables")

    # Key metrics
    try:
        cash = conn.execute("SELECT SUM(amount) FROM ledger").fetchone()[0] or 0
        subs = conn.execute("SELECT COUNT(*) FROM subscriptions WHERE status='active'").fetchone()[0]
        days = conn.execute("SELECT MAX(day) FROM service_day").fetchone()[0] or 0
        print(f"\n📊 Key Metrics:")
        print(f"  Cash balance (net ledger): ${cash:,.2f}")
        print(f"  Active subscribers: {subs:,}")
        print(f"  Days simulated: {days}")
    except Exception:
        pass

    # Dividends
    try:
        divs = conn.execute("SELECT SUM(founder_payout) FROM dividends").fetchone()[0] or 0
        print(f"  Founder dividends: ${divs:,.2f}")
    except Exception:
        pass

    print()


def export_csv(conn: sqlite3.Connection, output_dir: Path):
    """Export all tables as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = get_all_tables(conn)

    for table in tables:
        columns = get_table_columns(conn, table)
        cursor = conn.execute(f"SELECT * FROM {table}")

        csv_path = output_dir / f"{table}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in cursor:
                writer.writerow(row)

        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  ✅ {table}.csv ({count:,} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Decode NovaMind session databases for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("nmdb_path", type=str, help="Path to .nmdb file")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for decoded .db file")
    parser.add_argument("--dump", action="store_true",
                        help="Dump all tables as JSON to stdout")
    parser.add_argument("--tables", type=str, default=None,
                        help="Comma-separated table names to dump (with --dump)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit rows per table in dump")
    parser.add_argument("--shell", action="store_true",
                        help="Open interactive sqlite3 shell")
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="Export all tables as CSV to this directory")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics")

    args = parser.parse_args()

    nmdb_path = Path(args.nmdb_path)
    if not nmdb_path.exists():
        print(f"Error: File not found: {nmdb_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        db_path = Path(args.output)
    else:
        # Use temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        tmp.close()

    # Decode
    print(f"Decoding {nmdb_path} → {db_path}")
    unprotect_db(nmdb_path, db_path)
    print(f"✅ Decoded ({db_path.stat().st_size / 1024 / 1024:.1f} MB)")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    if args.summary or (not args.dump and not args.shell and not args.csv_dir and not args.output):
        print_summary(conn)

    if args.dump:
        tables = args.tables.split(",") if args.tables else get_all_tables(conn)
        result = {}
        for table in tables:
            result[table] = dump_table(conn, table, args.limit)
        print(json.dumps(result, indent=2, default=str))

    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        print(f"\nExporting CSVs to {csv_dir}/")
        export_csv(conn, csv_dir)
        print(f"✅ All tables exported")

    if args.shell:
        conn.close()
        print(f"\nOpening sqlite3 shell on {db_path}")
        print("Type .tables to see all tables, .quit to exit\n")
        subprocess.run(["sqlite3", str(db_path)])

    conn.close()

    # Clean up temp file if no output specified
    if not args.output and not args.shell:
        db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
