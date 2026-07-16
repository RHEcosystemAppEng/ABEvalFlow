"""Aggregate ASE edge case evaluation results into per-edge-case pass/fail.

Reads grading.json files from the ASE edge case run and produces a JSON
array of per-edge-case results compatible with the EdgeCaseGate and
report.json's edge_case_results field.

Usage::

    python scripts/aggregate_edge_case_evals.py \\
        --results-dir /workspace/eval-results/my-submission/edge-cases \\
        --output-dir /workspace/reports/my-submission
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from abevalflow.certification import DEFAULT_EDGE_CASE_PASS_THRESHOLD

logger = logging.getLogger(__name__)


def aggregate_edge_case_results(results_dir: Path) -> list[dict]:
    """Parse per-edge-case ASE results into pass/fail.

    Each edge case ran as a separate ASE invocation under its own
    subdirectory: results_dir/<edge-name>/iteration-1/...

    For each edge case subdirectory, finds the with_skill grading.json
    and extracts the overall pass rate as the edge case score.

    Args:
        results_dir: Directory containing per-edge-case result subdirs
            (e.g. eval-results/<name>/edge-cases/)

    Returns:
        List of dicts with {name, passed, score} per edge case.
    """
    edge_results: list[dict] = []

    for edge_dir in sorted(results_dir.iterdir()):
        if not edge_dir.is_dir():
            continue

        edge_name = edge_dir.name
        grading_files = sorted(edge_dir.rglob("grading.json"))
        with_skill_gradings = [g for g in grading_files if "with_skill" in g.parts]

        if not with_skill_gradings:
            logger.warning("Edge case '%s': no with_skill grading.json — counted as failure", edge_name)
            edge_results.append({"name": edge_name, "passed": False, "score": 0.0})
            continue

        total_passed = 0
        total_count = 0
        parse_failed = False
        for grading_path in with_skill_gradings:
            try:
                data = json.loads(grading_path.read_text())
                summary = data.get("summary", {})
                total_passed += summary.get("passed", 0)
                total_count += summary.get("total", 0)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to parse %s — counted as failure: %s", grading_path, e)
                parse_failed = True
                break

        if parse_failed or total_count == 0:
            edge_results.append({"name": edge_name, "passed": False, "score": 0.0})
        else:
            score = total_passed / total_count
            passed = score >= DEFAULT_EDGE_CASE_PASS_THRESHOLD
            edge_results.append({"name": edge_name, "passed": passed, "score": score})

    return edge_results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Aggregate ASE edge case results into per-edge-case pass/fail",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to ASE edge case results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write edge-case-results.json",
    )

    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        logger.info("No edge case results directory found — nothing to aggregate")
        return 0

    results = aggregate_edge_case_results(args.results_dir)

    if not results:
        logger.info("No edge case results to aggregate")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "edge-case-results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %d edge case results to %s", len(results), output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
