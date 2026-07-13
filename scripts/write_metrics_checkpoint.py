"""Write a metrics checkpoint from quality review token data.

Reads _ai_review.json for token usage and writes a MetricsContext
checkpoint to the report directory.

Usage::

    python scripts/write_metrics_checkpoint.py \\
        --run-id <pipeline-run-name> \\
        --submission-name <name> \\
        --report-dir /workspace/source/reports/<name> \\
        --review-file /workspace/source/_ai_review.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from abevalflow.observability.context import MetricsContext

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Write metrics checkpoint")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--submission-name", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, default=None)
    args = parser.parse_args()

    ctx = MetricsContext(
        run_id=args.run_id,
        submission_name=args.submission_name,
    )

    if args.review_file and args.review_file.exists():
        try:
            review = json.loads(args.review_file.read_text())
            usage = review.get("token_usage", {})
            if usage:
                ctx.record_tokens(
                    "quality_review",
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("model"),
                )
                logger.info("Recorded quality review tokens: %s", usage)
            else:
                logger.info("No token_usage in review file")
        except Exception as e:
            logger.warning("Could not read review tokens: %s", e)
    else:
        logger.info("No review file found, skipping token capture")

    if ctx.total_tokens > 0:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        ctx.checkpoint(args.report_dir)
        logger.info("Metrics checkpoint written to %s", args.report_dir)
    else:
        logger.info("No token data collected, skipping checkpoint")


if __name__ == "__main__":
    main()
