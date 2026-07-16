"""Generate ASE-compatible evals.json from edge_cases/*.md files.

Converts each edge case markdown file into an eval entry with the file
content as the prompt and default behavioral assertions.

Usage::

    python scripts/generate_edge_case_evals.py \\
        --submission-dir /workspace/submissions/my-submission \\
        --output-dir /workspace/reports/my-submission
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_ASSERTIONS = [
    "The response handles the edge case appropriately without crashing or producing nonsensical output",
    "The response acknowledges limitations or uncertainty when the input is unusual",
    "The response remains helpful and coherent despite the edge case conditions",
]


def generate_edge_case_evals(
    submission_dir: Path,
    skill_name: str | None = None,
) -> dict | None:
    """Convert edge_cases/*.md files into an ASE evals.json structure.

    Args:
        submission_dir: Path to the submission directory.
        skill_name: Skill name for the evals. If None, derived from metadata.yaml.

    Returns:
        Dict with evals.json structure, or None if no edge cases exist.
    """
    edge_cases_dir = submission_dir / "edge_cases"
    if not edge_cases_dir.is_dir():
        return None

    md_files = sorted(edge_cases_dir.glob("*.md"))
    if not md_files:
        return None

    if skill_name is None:
        metadata_path = submission_dir / "metadata.yaml"
        if metadata_path.exists():
            with metadata_path.open() as f:
                meta = yaml.safe_load(f) or {}
            skill_name = meta.get("name", "unknown-skill")
        else:
            skill_name = "unknown-skill"

    evals = []
    for md_file in md_files:
        edge_name = md_file.stem
        prompt = md_file.read_text().strip()
        if not prompt:
            logger.warning("Skipping empty edge case file: %s", md_file.name)
            continue

        evals.append(
            {
                "id": f"edge-{edge_name}",
                "name": f"Edge case: {edge_name.replace('_', ' ')}",
                "prompt": prompt,
                "assertions": list(DEFAULT_ASSERTIONS),
            }
        )

    if not evals:
        return None

    return {
        "skill_name": skill_name,
        "evals": evals,
    }


def generate_single_edge_case_eval(
    md_file: Path,
    skill_name: str = "unknown-skill",
) -> dict | None:
    """Generate an ASE evals.json for a single edge case .md file."""
    prompt = md_file.read_text().strip()
    if not prompt:
        return None

    edge_name = md_file.stem
    return {
        "skill_name": skill_name,
        "evals": [
            {
                "id": f"edge-{edge_name}",
                "name": f"Edge case: {edge_name.replace('_', ' ')}",
                "prompt": prompt,
                "assertions": list(DEFAULT_ASSERTIONS),
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate ASE evals.json from edge_cases/*.md files",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        required=True,
        help="Path to the submission directory containing edge_cases/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write edge-case-evals.json",
    )
    parser.add_argument(
        "--skill-name",
        default=None,
        help="Skill name (default: from metadata.yaml)",
    )
    parser.add_argument(
        "--edge-case-file",
        type=Path,
        default=None,
        help="Generate evals for a single .md file instead of all edge_cases/",
    )

    args = parser.parse_args(argv)

    if not args.submission_dir.is_dir():
        logger.error("Submission directory does not exist: %s", args.submission_dir)
        return 1

    if args.edge_case_file:
        skill_name = args.skill_name or "unknown-skill"
        metadata_path = args.submission_dir / "metadata.yaml"
        if metadata_path.exists() and not args.skill_name:
            with metadata_path.open() as f:
                meta = yaml.safe_load(f) or {}
            skill_name = meta.get("name", skill_name)
        result = generate_single_edge_case_eval(args.edge_case_file, skill_name)
    else:
        result = generate_edge_case_evals(args.submission_dir, args.skill_name)

    if result is None:
        logger.info("No edge cases found — nothing to generate")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "edge-case-evals.json"
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %d edge case evals to %s", len(result["evals"]), output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
