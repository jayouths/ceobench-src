#!/usr/bin/env python3
"""参考真实短期结果，生成仅用于绘图验链路的五周模拟数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from saas_bench.evaluation.opening_plots import load_metrics, plot_opening_results
from saas_bench.evaluation.opening_preview import build_mock_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    analysis = load_metrics(args.analysis)
    baseline, analysis_mock = build_mock_pair(analysis)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "baseline_mock_metrics.json"
    analysis_path = args.output_dir / "analysis_mock_metrics.json"
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n")
    analysis_path.write_text(json.dumps(analysis_mock, ensure_ascii=False, indent=2) + "\n")
    _write_readme(args.output_dir, args.analysis, baseline_path, analysis_path)

    paths = plot_opening_results(
        baseline,
        analysis_mock,
        args.output_dir,
        watermark="PREVIEW - ALL DATA MOCK",
    )
    print(baseline_path)
    print(analysis_path)
    for path in paths:
        print(path)


def _write_readme(
    output_dir: Path,
    reference_path: Path,
    baseline_path: Path,
    analysis_path: Path,
) -> None:
    output_dir.joinpath("README.md").write_text(
        "# Opening Figure Preview\n\n"
        "> **MOCK DATA WARNING:** Baseline 与 Analysis 均为五周模拟数据，仅用于验证绘图链路和版式，"
        "不得作为实验结论或论文证据。\n\n"
        f"- 趋势与量级参考：真实 14 天运行 `{reference_path}`\n"
        f"- Baseline 模拟数据：`{baseline_path.name}`\n"
        f"- Analysis 模拟数据：`{analysis_path.name}`\n"
        "- 所有图片均带有 `PREVIEW - ALL DATA MOCK` 水印。\n"
        "- 两组均覆盖 35 天，因此末四周平均每周净现金流可以完整展示。\n"
    )


if __name__ == "__main__":
    main()
