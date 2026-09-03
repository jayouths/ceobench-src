#!/usr/bin/env python3
"""根据两组单次运行指标生成开题阶段结果图。"""

from __future__ import annotations

import argparse
from pathlib import Path

from saas_bench.evaluation.opening_plots import load_metrics, plot_opening_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--watermark",
        help="可选图面水印；使用模拟数据时必须显式传入",
    )
    args = parser.parse_args()

    paths = plot_opening_results(
        load_metrics(args.baseline),
        load_metrics(args.analysis),
        args.output_dir,
        watermark=args.watermark,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
