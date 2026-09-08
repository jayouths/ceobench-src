#!/usr/bin/env python3
"""Generate all public documentation from authoritative source definitions."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from saas_bench.runtime.docs_generator import generate_public_docs


def main():
    parser = argparse.ArgumentParser(description="Generate public documentation")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "public" / "docs",
        help="Output directory (default: public/docs)",
    )
    args = parser.parse_args()
    generate_public_docs(args.output)


if __name__ == "__main__":
    main()
