"""Backfill scorecards, gate_results, and certifications from MinIO artifacts.

Reads scorecard.json files from MinIO and inserts into the new normalized
tables. Idempotent — skips scorecards that already exist (by pipeline_run_id).

Usage::

    python scripts/backfill_scorecards.py \\
        --database-url postgresql+psycopg://user:pass@host:5432/abevalflow \\
        --minio-endpoint minio.ab-eval-flow.svc:9000 \\
        --minio-access-key abevalflow \\
        --minio-secret-key <secret> \\
        --bucket ab-eval-reports

    # Dry run (validate only, no writes):
    python scripts/backfill_scorecards.py --dry-run ...

    # Resume from checkpoint:
    python scripts/backfill_scorecards.py --resume ...

    # Limit to N artifacts:
    python scripts/backfill_scorecards.py --limit 100 ...

    # Filter by prefix:
    python scripts/backfill_scorecards.py --prefix certified/ ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from abevalflow.db.engine import get_engine, init_db, make_session
from abevalflow.db.models import ScorecardRow
from abevalflow.scorecard import Scorecard
from scripts.store_results import map_certifications, map_gate_results, map_scorecard_to_row

logger = logging.getLogger(__name__)


@dataclass
class BackfillState:
    last_processed_key: str = ""
    processed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    errors: list[dict] = field(default_factory=list)


def _load_checkpoint(path: Path) -> BackfillState:
    if path.exists():
        data = json.loads(path.read_text())
        return BackfillState(
            last_processed_key=data.get("last_processed_key", ""),
            processed_count=data.get("processed_count", 0),
            skipped_count=data.get("skipped_count", 0),
            error_count=data.get("error_count", 0),
            errors=data.get("errors", []),
        )
    return BackfillState()


def _save_checkpoint(path: Path, state: BackfillState, max_errors: int = 50) -> None:
    path.write_text(
        json.dumps(
            {
                "last_processed_key": state.last_processed_key,
                "processed_count": state.processed_count,
                "skipped_count": state.skipped_count,
                "error_count": state.error_count,
                "errors": state.errors[-max_errors:],
            },
            indent=2,
        )
    )


def backfill(
    database_url: str | None = None,
    minio_endpoint: str | None = None,
    minio_access_key: str | None = None,
    minio_secret_key: str | None = None,
    bucket: str = "ab-eval-reports",
    secure: bool = False,
    dry_run: bool = False,
    resume: bool = False,
    limit: int | None = None,
    prefix_filter: str | None = None,
    checkpoint_path: Path | None = None,
) -> BackfillState:
    from minio import Minio

    endpoint = minio_endpoint or os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access_key = minio_access_key or os.environ.get("MINIO_ACCESS_KEY", "")
    secret_key = minio_secret_key or os.environ.get("MINIO_SECRET_KEY", "")

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    if not client.bucket_exists(bucket):
        logger.error("Bucket %s does not exist", bucket)
        return BackfillState(error_count=1)

    ckpt_path = checkpoint_path or Path("_backfill_checkpoint.json")
    state = _load_checkpoint(ckpt_path) if resume else BackfillState()

    engine = get_engine(database_url)
    init_db(engine)
    session_factory = make_session(engine)

    start_after = state.last_processed_key if (resume and state.last_processed_key) else None
    if start_after:
        logger.info("Resuming after %s", start_after)

    objects = client.list_objects(bucket, prefix=prefix_filter, recursive=True, start_after=start_after)

    scorecard_keys: list[str] = []
    for obj in objects:
        if not obj.object_name or not obj.object_name.endswith("/scorecard.json"):
            continue
        scorecard_keys.append(obj.object_name)
        if limit and len(scorecard_keys) >= limit:
            break

    logger.info("Found %d scorecard.json artifacts to process", len(scorecard_keys))

    for key in scorecard_keys:
        try:
            response = client.get_object(bucket, key)
            raw = response.read()
            response.close()
            response.release_conn()

            scorecard = Scorecard.model_validate_json(raw)

            with session_factory() as session:
                existing = session.execute(
                    select(ScorecardRow).where(ScorecardRow.pipeline_run_id == scorecard.pipeline_run_id)
                ).scalar_one_or_none()

                if existing is not None:
                    logger.debug("Skipping %s — already exists", scorecard.pipeline_run_id)
                    state.skipped_count += 1
                    state.last_processed_key = key
                    continue

                if dry_run:
                    logger.info("[DRY RUN] Would insert scorecard for %s", scorecard.pipeline_run_id)
                    state.processed_count += 1
                    state.last_processed_key = key
                    continue

                sc_row = map_scorecard_to_row(scorecard)
                gate_rows = map_gate_results(scorecard, sc_row)
                cert_rows = map_certifications(scorecard, sc_row)

                session.add(sc_row)
                session.add_all(gate_rows)
                session.add_all(cert_rows)
                session.commit()

                state.processed_count += 1
                logger.info(
                    "Backfilled: %s (gates=%d, certs=%d)",
                    scorecard.pipeline_run_id,
                    len(gate_rows),
                    len(cert_rows),
                )

        except Exception as e:
            state.error_count += 1
            state.errors.append({"key": key, "error": str(e)[:200]})
            logger.warning("Error processing %s: %s", key, str(e)[:200])

        state.last_processed_key = key

        if state.processed_count % 50 == 0 and state.processed_count > 0:
            _save_checkpoint(ckpt_path, state)

    _save_checkpoint(ckpt_path, state)

    logger.info(
        "Backfill complete: processed=%d skipped=%d errors=%d",
        state.processed_count,
        state.skipped_count,
        state.error_count,
    )
    return state


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Backfill scorecards from MinIO artifacts")
    parser.add_argument("--database-url", type=str, default=None)
    parser.add_argument("--minio-endpoint", type=str, default=None)
    parser.add_argument("--minio-access-key", type=str, default=None)
    parser.add_argument("--minio-secret-key", type=str, default=None)
    parser.add_argument("--bucket", type=str, default="ab-eval-reports")
    parser.add_argument("--secure", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    state = backfill(
        database_url=args.database_url,
        minio_endpoint=args.minio_endpoint,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        bucket=args.bucket,
        secure=args.secure,
        dry_run=args.dry_run,
        resume=args.resume,
        limit=args.limit,
        prefix_filter=args.prefix,
        checkpoint_path=args.checkpoint,
    )

    sys.exit(0 if state.error_count == 0 else 1)


if __name__ == "__main__":
    main()
