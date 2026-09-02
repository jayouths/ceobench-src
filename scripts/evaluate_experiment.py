#!/usr/bin/env python3
"""聚合多个实验目录，并比较各组与 Baseline 的描述统计。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from saas_bench.evaluation.experiment_metrics import aggregate_experiments
from saas_bench.evaluation.exports import (
    export_experiment_metrics,
    export_run_metrics,
)
from saas_bench.evaluation.run_metrics import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=Path,
        help="一个或多个实验组目录；子目录必须是独立运行目录",
    )
    parser.add_argument("--baseline", required=True, help="Baseline 的实验名称")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs_by_group: defaultdict[str, list[dict]] = defaultdict(list)
    for experiment_dir in args.experiment_dirs:
        for run_dir in _discover_runs(experiment_dir):
            metrics = evaluate_run(run_dir)
            export_run_metrics(metrics, run_dir / "evaluation")
            group = metrics["run"].get("experiment_name")
            if not group:
                raise ValueError(f"Run has no experiment name: {run_dir}")
            runs_by_group[str(group)].append(metrics)

    aggregated = aggregate_experiments(
        runs_by_group,
        baseline_group=args.baseline,
    )
    paths = export_experiment_metrics(aggregated, args.output_dir)
    for path in paths:
        print(path)


def _discover_runs(experiment_dir: Path) -> list[Path]:
    experiment_dir = experiment_dir.expanduser().resolve()
    if (experiment_dir / "config.json").is_file():
        return [experiment_dir]
    runs = sorted(
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    )
    if not runs:
        raise ValueError(f"No run directories found in {experiment_dir}")
    return runs


if __name__ == "__main__":
    main()
