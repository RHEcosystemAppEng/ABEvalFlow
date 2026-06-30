#!/usr/bin/env python3
"""Run SKILL.md security scan on a submission directory.

Deterministic regex checks run always. LLM semantic review runs by default
and can be disabled with --no-llm.

Produces a JSON report compatible with the SecurityGate interface.

Exit codes: 0 = scan completed, 1 = scan error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from abevalflow.security.skillmd_scanner import (
    llm_security_review,
    scan_directory,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan SKILL.md files for security risks",
    )
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Path to the submission directory to scan",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSON report",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM semantic security review",
    )
    args = parser.parse_args(argv)

    if not args.submission_dir.is_dir():
        logger.error("Not a directory: %s", args.submission_dir)
        return 1

    try:
        result = scan_directory(args.submission_dir)
    except Exception:
        logger.exception("Scan failed")
        return 1

    if not args.no_llm:
        try:
            llm_findings = llm_security_review(args.submission_dir)
            result["findings"].extend(llm_findings)
        except Exception:
            logger.exception("LLM review failed, continuing with deterministic results")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    findings = result.get("findings", [])
    severity_counts: dict[str, int] = {}
    for f in findings:
        sev = f.get("severity", "info")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    logger.info(
        "Scan complete: %d findings (%s)",
        len(findings),
        ", ".join(f"{k}={v}" for k, v in sorted(severity_counts.items())),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.exit(main())
