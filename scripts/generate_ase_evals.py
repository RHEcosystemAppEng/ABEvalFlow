#!/usr/bin/env python3
"""Generate evals/evals.json for ASE evaluation from SKILL.md.

This script is called by the generate-tests Tekton task when:
- eval-engine is 'ase' or 'both'
- evals/evals.json does not exist

It uses an LLM to analyze the skill definition and generate evaluation
prompts with assertions that test skill-specific knowledge.

Usage:
    python generate_ase_evals.py /path/to/submission

Output (JSON to stdout):
    {"generated": ["evals/evals.json"]}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Import from generate_tests to reuse the existing infrastructure
from scripts.generate_tests import _load_submission, generate_ase_evals

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate evals/evals.json from SKILL.md for ASE evaluation")
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Path to the submission directory",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum generation retry attempts (default: 3)",
    )
    args = parser.parse_args(argv)

    if not args.submission_dir.is_dir():
        result = {"generated": [], "error": f"Not a directory: {args.submission_dir}"}
        print(json.dumps(result, indent=2))
        return 1

    try:
        metadata, skill_content = _load_submission(args.submission_dir)
        if not skill_content.strip():
            raise ValueError("skills/SKILL.md is empty or missing")

        generate_ase_evals(
            args.submission_dir,
            metadata.name,
            skill_content,
            max_retries=args.max_retries,
        )
        result = {"generated": ["evals/evals.json"]}
        print(json.dumps(result, indent=2))
        return 0

    except Exception as exc:
        logger.error("ASE evals generation failed: %s", exc)
        result = {"generated": [], "error": str(exc)}
        print(json.dumps(result, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
