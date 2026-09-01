#!/usr/bin/env python3
"""Bash Agent 主实验的命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .run_config import create_new_runner, create_resumed_runner


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run bash agent for SaaS Bench")
    # 新实验必须明确选择配置；断点恢复只读取原运行目录的 config.json。
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--config",
        type=Path,
        help="Complete TOML configuration for a new experiment",
    )
    mode.add_argument(
        "--resume",
        help="Resume a run by run id or run directory using its saved configuration",
    )
    args = parser.parse_args(argv)
    runner = (
        create_resumed_runner(args.resume)
        if args.resume
        else create_new_runner(args.config)
    )
    result = runner.run(verbose=True)
    print(f"\nResult: {result['outcome']}")
    print(f"Final Cash: ${result['final_cash']:,.0f}")
    print(f"Workspace: {result['workspace_dir']}")


if __name__ == "__main__":
    main()
