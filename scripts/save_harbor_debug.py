"""Save Harbor job directory to MinIO for debugging (tarball).

NOTE: Evaluate-time use is disabled. Prefer ``publish.upload_aeh_debug_artifacts``
at store time — individual files under prepare's report-prefix (no tar / no
second prefix).

Usage::

    python scripts/save_harbor_debug.py \
        --jobs-dir /workspace/jobs \
        --submission-name my-skill \
        --pipeline-run-id abevalflow-xyz \
        --report-prefix YYYYMMDD_HHMMSS_name_run-id
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _build_artifact_prefix(
    submission_name: str,
    pipeline_run_id: str,
) -> str:
    """Build the MinIO object prefix: YYYYMMDD_hhmmss_<name>_<run-id>."""
    datestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{datestamp}_{submission_name}_{pipeline_run_id}"


def find_latest_job_dir(jobs_dir: Path) -> Path | None:
    """Find the most recent Harbor job directory (timestamp-based).

    Harbor creates job directories with names like: YYYY-MM-DD__HH-MM-SS
    Returns the most recent one or None if not found.
    """
    if not jobs_dir.is_dir():
        logger.warning("Jobs directory not found: %s", jobs_dir)
        return None

    # Find all timestamp-based directories (start with "20")
    job_dirs = [d for d in jobs_dir.iterdir() if d.is_dir() and d.name.startswith("20")]

    if not job_dirs:
        logger.warning("No Harbor job directories found in %s", jobs_dir)
        return None

    # Sort by name (timestamp format sorts chronologically) and return latest
    latest = sorted(job_dirs)[-1]
    logger.info("Found latest Harbor job: %s", latest)
    return latest


def create_job_tarball(job_dir: Path) -> Path:
    """Create a compressed tarball of the Harbor job directory.

    Returns the path to the temporary tarball file.
    """
    tarball = Path(tempfile.mktemp(suffix=".tar.gz", prefix="harbor-debug-"))
    logger.info("Creating tarball: %s", tarball)

    with tarfile.open(tarball, "w:gz") as tar:
        # Add the job directory with its basename as the archive root
        tar.add(job_dir, arcname=job_dir.name)

    logger.info("Tarball created: %s (%d bytes)", tarball, tarball.stat().st_size)
    return tarball


def upload_to_minio(
    tarball: Path,
    job_dir_name: str,
    report_prefix: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str = "ab-eval-reports",
    secure: bool | None = None,
) -> bool:
    """Upload tarball to MinIO following publish.py pattern.

    Target path: s3://{bucket}/{report-prefix}/debug/harbor/{job-dir-name}.tar.gz
    Returns True on success.
    """
    from minio import Minio

    parsed = urlparse(endpoint)
    host = parsed.netloc or parsed.path
    if secure is None:
        secure = parsed.scheme == "https"

    client = Minio(
        host,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("Created bucket: %s", bucket)

    object_name = f"{report_prefix}/debug/harbor/{job_dir_name}.tar.gz"

    try:
        client.fput_object(bucket, object_name, str(tarball))
        logger.info("✓ Harbor debug artifacts saved to: s3://%s/%s", bucket, object_name)
        logger.info("  Download with: mc cp %s/%s/%s .", host, bucket, object_name)
        logger.info("  Extract with: tar -xzf harbor-job.tar.gz")
        return True
    except Exception as exc:
        logger.error("Failed to upload to MinIO: %s", exc)
        return False


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Save Harbor debug artifacts to MinIO")
    parser.add_argument("--jobs-dir", type=Path, required=True, help="Harbor jobs directory")
    parser.add_argument("--submission-name", type=str, required=True, help="Submission name")
    parser.add_argument("--pipeline-run-id", type=str, required=True, help="Pipeline run ID")
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="",
        help="MinIO prefix (e.g., YYYYMMDD_HHMMSS_name_run-id). If empty, builds from submission/run-id.",
    )
    parser.add_argument(
        "--minio-endpoint",
        type=str,
        default=None,
        help="MinIO endpoint (default: MINIO_ENDPOINT env var)",
    )
    parser.add_argument("--minio-bucket", type=str, default="ab-eval-reports", help="MinIO bucket name")

    args = parser.parse_args()

    # Get MinIO credentials from environment
    minio_endpoint = args.minio_endpoint or os.environ.get("MINIO_ENDPOINT", "")
    minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "")
    minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "")

    if not minio_endpoint:
        logger.error("MINIO_ENDPOINT not set")
        return 1

    if not minio_access_key:
        logger.error("MINIO_ACCESS_KEY not set")
        return 1

    if not minio_secret_key:
        logger.error("MINIO_SECRET_KEY not set")
        return 1

    logger.info("=== Saving Harbor debug artifacts to MinIO ===")
    logger.info("Jobs dir: %s", args.jobs_dir)
    logger.info("Submission: %s", args.submission_name)
    logger.info("Pipeline run: %s", args.pipeline_run_id)

    # Find latest job directory
    job_dir = find_latest_job_dir(args.jobs_dir)
    if not job_dir:
        logger.warning("No Harbor job directory found - nothing to upload")
        return 0  # Exit successfully (not an error if no jobs exist)

    # Build report prefix if not provided
    report_prefix = args.report_prefix or _build_artifact_prefix(
        args.submission_name,
        args.pipeline_run_id,
    )
    logger.info("Report prefix: %s", report_prefix)

    # Create tarball
    tarball = None
    try:
        tarball = create_job_tarball(job_dir)

        # Upload to MinIO
        success = upload_to_minio(
            tarball=tarball,
            job_dir_name=job_dir.name,
            report_prefix=report_prefix,
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            bucket=args.minio_bucket,
        )

        return 0 if success else 1

    finally:
        # Cleanup temporary tarball
        if tarball and tarball.exists():
            tarball.unlink()
            logger.debug("Cleaned up temporary tarball: %s", tarball)


if __name__ == "__main__":
    sys.exit(main())
