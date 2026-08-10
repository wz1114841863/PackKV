#!/usr/bin/env python3
"""Export deterministic BRISK-KV Format v0 vectors for Chisel tests."""

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(PROJECT_ROOT))
os.environ.setdefault("PACKKV_GLOBAL_K_STATS", "0")

from utils.golden_vectors import directed_cases, export_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "hardware" / "golden_vectors" / "generated",
        help="Golden-vector output root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the named generated case directories",
    )
    args = parser.parse_args()

    manifest = export_cases(args.output, directed_cases(), args.overwrite)
    print(f"Golden vectors exported: {args.output.resolve()}")
    print(f"Cases: {manifest['case_count']}")
    for case in manifest["cases"]:
        print(f"  {case['case']}: {case['file_count']} files")


if __name__ == "__main__":
    main()
