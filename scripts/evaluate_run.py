#!/usr/bin/env python3
"""计算一个 CEO-Bench 运行目录的单次实验指标。"""

from __future__ import annotations

import argparse
from pathlib import Path

from saas_bench.evaluation.exports import export_run_metrics
from saas_bench.evaluation.run_metrics import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="包含 world.nmdb 的运行目录")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；默认写入 <run_dir>/evaluation",
    )
    args = parser.parse_args()

    metrics = evaluate_run(args.run_dir)
    output_dir = args.output_dir or args.run_dir / "evaluation"
    json_path, csv_path = export_run_metrics(metrics, output_dir)
    print(f"Metrics JSON: {json_path}")
    print(f"Long-table CSV: {csv_path}")
    print(f"Outcome: {metrics['summary']['outcome']}")
    print(f"Final cash: {metrics['summary']['final_cash']:.2f}")


if __name__ == "__main__":
    main()
