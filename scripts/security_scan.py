"""Security scan a submission directory for prompt injection and credential access.

Checks SKILL.md files for:
  1. Prompt injection patterns (17 regex patterns, context-aware severity)
  2. Credential access patterns (sensitive paths, env vars, dangerous commands)

Exit codes: 0 = pass (no errors, warnings only), 1 = blocked (ERROR findings).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from abevalflow.security_scanner import scan_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Security scan a submission directory"
    )
    parser.add_argument(
        "submission_dir", type=Path, help="Path to the submission directory"
    )
    args = parser.parse_args(argv)

    submission_dir: Path = args.submission_dir
    if not submission_dir.is_dir():
        result = {
            "passed": False,
            "findings": [],
            "summary": f"Not a directory: {submission_dir}",
        }
        print(json.dumps(result, indent=2))
        return 1

    scan_result = scan_submission(submission_dir)
    print(json.dumps(scan_result.model_dump(), indent=2))
    return 0 if scan_result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
