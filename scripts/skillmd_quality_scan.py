#!/usr/bin/env python3
"""Run deterministic quality checks on a submission directory.

Checks description quality, broken references, file completeness,
imprecise instructions, unfinished content, and generic advice.

Produces a JSON report compatible with the QualityGate interface.

Exit codes: 0 = scan completed, 1 = scan error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from abevalflow.quality.skillmd_quality_scanner import scan_directory

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic quality checks on skill submissions",
    )
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Path to the submission directory to check",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSON report",
    )
    args = parser.parse_args(argv)

    if not args.submission_dir.is_dir():
        logger.error("Not a directory: %s", args.submission_dir)
        return 1

    try:
        result = scan_directory(args.submission_dir)
    except Exception:
        logger.exception("Quality scan failed")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    findings = result.get("findings", [])
    logger.info("Quality scan complete: %d findings", len(findings))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.exit(main())
