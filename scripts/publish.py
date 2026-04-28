"""Publish evaluation artifacts after a pipeline run.

Uploads reports to MinIO, optionally promotes images to Quay.io,
posts a summary comment on the triggering GitHub PR, and cleans up
ephemeral resources from the OpenShift internal registry.

Usage::

    python scripts/publish.py \\
        --report-dir /path/to/reports/my-submission \\
        --submission-name my-submission \\
        --pipeline-run-id abevalflow-xyz \\
        --recommendation pass \\
        --uplift-threshold -1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _build_artifact_prefix(
    submission_name: str,
    pipeline_run_id: str,
) -> str:
    """Build the MinIO object prefix: YYYYMMDD_hhmmss_<name>_<run-id>."""
    datestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{datestamp}_{submission_name}_{pipeline_run_id}"


def upload_reports(
    report_dir: Path,
    submission_name: str,
    pipeline_run_id: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str = "ab-eval-reports",
    secure: bool | None = None,
) -> str | None:
    """Upload report.json and report.md to MinIO.

    Returns the artifact prefix on success, None on failure.
    ``secure`` is auto-detected from the endpoint scheme when not provided.
    """
    from minio import Minio

    prefix = _build_artifact_prefix(submission_name, pipeline_run_id)

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

    uploaded = 0
    for filename in ("report.json", "report.md"):
        filepath = report_dir / filename
        if not filepath.exists():
            logger.warning("File not found, skipping: %s", filepath)
            continue

        object_name = f"{prefix}/{filename}"
        client.fput_object(bucket, object_name, str(filepath))
        logger.info("Uploaded %s -> s3://%s/%s", filepath, bucket, object_name)
        uploaded += 1

    if uploaded == 0:
        logger.error("No report files found in %s", report_dir)
        return None

    return prefix


def promote_to_quay(
    treatment_image_ref: str,
    submission_name: str,
    commit_sha: str,
    quay_repo: str,
    ttl_days: int = 7,
) -> bool:
    """Copy the treatment image from internal registry to Quay.io.

    Uses skopeo to copy the image. Returns True on success.
    """
    quay_tag = f"{quay_repo}/{submission_name}:treatment-{commit_sha[:12]}"

    cmd = [
        "skopeo", "copy",
        "--src-tls-verify=false",
        f"docker://{treatment_image_ref}",
        f"docker://{quay_tag}",
    ]

    logger.info("Promoting to Quay: %s -> %s (ttl=%dd)", treatment_image_ref, quay_tag, ttl_days)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error("skopeo copy failed: %s", result.stderr)
            return False
        logger.info("Image promoted to %s", quay_tag)
        return True
    except FileNotFoundError:
        logger.error("skopeo not found — cannot promote image")
        return False
    except subprocess.TimeoutExpired:
        logger.error("skopeo copy timed out after 300s")
        return False


def post_pr_comment(
    repo_name: str,
    pr_number: str,
    submission_name: str,
    report_dir: Path,
    github_token: str | None = None,
) -> bool:
    """Post evaluation summary as a GitHub PR comment.

    Uses the GitHub REST API via ``urllib`` (no ``gh`` CLI required).
    ``github_token`` falls back to the ``GITHUB_TOKEN`` environment variable.
    """
    import os

    token = github_token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.error("GITHUB_TOKEN not set — cannot post PR comment")
        return False

    report_path = report_dir / "report.json"
    if not report_path.exists():
        logger.warning("No report.json found, skipping PR comment")
        return False

    try:
        report = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read report.json: %s", exc)
        return False

    summary = report.get("summary", {})
    treatment = summary.get("treatment", {})
    control = summary.get("control", {})
    recommendation = summary.get("recommendation", "unknown").upper()
    uplift = summary.get("uplift", 0.0)
    provenance = report.get("provenance", {})

    comment_body = (
        f"## AB Evaluation Results: `{submission_name}`\n\n"
        f"| Metric | Treatment | Control |\n"
        f"|--------|-----------|--------|\n"
        f"| Trials | {treatment.get('n_trials', '?')} | {control.get('n_trials', '?')} |\n"
        f"| Passed | {treatment.get('n_passed', '?')} | {control.get('n_passed', '?')} |\n"
        f"| Pass Rate | {treatment.get('pass_rate', 0):.2%} | {control.get('pass_rate', 0):.2%} |\n"
        f"| Mean Reward | {treatment.get('mean_reward', 'N/A')} | {control.get('mean_reward', 'N/A')} |\n\n"
        f"**Uplift:** {uplift:+.4f}  \n"
        f"**Recommendation:** {recommendation}  \n"
        f"**Pipeline Run:** `{provenance.get('pipeline_run_id', 'N/A')}`  \n"
        f"**Commit:** `{provenance.get('commit_sha', 'N/A')[:12] if provenance.get('commit_sha') else 'N/A'}`\n"
    )

    url = f"https://api.github.com/repos/{repo_name}/issues/{pr_number}/comments"
    payload = json.dumps({"body": comment_body}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if 200 <= resp.status < 300:
                logger.info("Posted PR comment to %s#%s", repo_name, pr_number)
                return True
            logger.error("GitHub API returned %d", resp.status)
            return False
    except Exception as exc:
        logger.error("Failed to post PR comment: %s", exc)
        return False


def cleanup_images(
    image_refs: list[str],
    registry_url: str = "image-registry.openshift-image-registry.svc:5000",
) -> int:
    """Delete images from the OpenShift internal registry.

    For digest refs (``@sha256:…``), deletes the parent ImageStream so all
    tags and layers are removed.  For tag refs, deletes the specific
    ImageStreamTag.  Returns the number of successfully deleted resources.
    """
    deleted = 0
    seen_imagestreams: set[str] = set()

    for ref in image_refs:
        stripped = ref.replace(f"{registry_url}/", "")
        parts = stripped.split("/", 1)
        if len(parts) != 2:
            logger.warning("Cannot parse image ref for cleanup: %s", ref)
            continue

        namespace = parts[0]
        name_part = parts[1]

        if "@" in name_part:
            is_name = name_part.split("@")[0]
            key = f"{namespace}/{is_name}"
            if key in seen_imagestreams:
                continue
            seen_imagestreams.add(key)
            cmd = ["oc", "delete", "imagestream", is_name, "-n", namespace,
                   "--ignore-not-found"]
        else:
            cmd = ["oc", "delete", "istag", name_part, "-n", namespace]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info("Deleted %s in %s", name_part, namespace)
                deleted += 1
            else:
                logger.warning("Failed to delete %s: %s", name_part, result.stderr.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("Cleanup failed for %s: %s", ref, exc)

    return deleted


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Publish evaluation artifacts")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--submission-name", type=str, required=True)
    parser.add_argument("--pipeline-run-id", type=str, required=True)
    parser.add_argument("--recommendation", type=str, required=True,
                        help="'pass' or 'fail' from the analyze step")
    parser.add_argument("--treatment-image-ref", type=str, default="")
    parser.add_argument("--control-image-ref", type=str, default="")
    parser.add_argument("--commit-sha", type=str, default="")
    parser.add_argument("--uplift-threshold", type=float, default=-1.0,
                        help="Minimum uplift for Quay promotion. -1.0 disables promotion.")
    parser.add_argument("--quay-repo", type=str, default="",
                        help="Quay.io repo for image promotion (e.g. quay.io/myorg)")
    parser.add_argument("--quay-ttl-days", type=int, default=7)
    parser.add_argument("--repo-name", type=str, default="",
                        help="GitHub repo (org/name) for PR comment")
    parser.add_argument("--pr-number", type=str, default="",
                        help="PR number for GitHub comment")
    parser.add_argument("--minio-endpoint", type=str, default=None,
                        help="MinIO endpoint (default: MINIO_ENDPOINT env var)")
    parser.add_argument("--minio-bucket", type=str, default="ab-eval-reports")
    args = parser.parse_args()

    import os
    minio_endpoint = args.minio_endpoint or os.environ.get("MINIO_ENDPOINT", "")
    minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "")
    minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "")

    upload_ok = True
    success = True

    # --- 1. Upload reports to MinIO ---
    if minio_endpoint and minio_access_key:
        prefix = upload_reports(
            report_dir=args.report_dir,
            submission_name=args.submission_name,
            pipeline_run_id=args.pipeline_run_id,
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            bucket=args.minio_bucket,
        )
        if prefix:
            logger.info("Reports uploaded to s3://%s/%s/", args.minio_bucket, prefix)
        else:
            logger.error("Failed to upload reports")
            upload_ok = False
            success = False
    else:
        logger.warning("MinIO not configured — skipping report upload")

    # --- 2. Quay.io promotion (if enabled) ---
    if (
        args.uplift_threshold > -1.0
        and args.recommendation == "pass"
        and args.treatment_image_ref
        and args.quay_repo
    ):
        report_path = args.report_dir / "report.json"
        uplift = 0.0
        if report_path.exists():
            try:
                data = json.loads(report_path.read_text())
                uplift = data.get("summary", {}).get("uplift", 0.0)
            except (json.JSONDecodeError, OSError):
                pass

        if uplift >= args.uplift_threshold:
            ok = promote_to_quay(
                treatment_image_ref=args.treatment_image_ref,
                submission_name=args.submission_name,
                commit_sha=args.commit_sha,
                quay_repo=args.quay_repo,
                ttl_days=args.quay_ttl_days,
            )
            if not ok:
                success = False
        else:
            logger.info(
                "Uplift %.4f below threshold %.4f — skipping Quay promotion",
                uplift, args.uplift_threshold,
            )
    else:
        logger.info("Quay promotion disabled or not applicable")

    # --- 3. GitHub PR comment ---
    if args.repo_name and args.pr_number:
        post_pr_comment(
            repo_name=args.repo_name,
            pr_number=args.pr_number,
            submission_name=args.submission_name,
            report_dir=args.report_dir,
        )
    else:
        logger.info("No repo-name/pr-number — skipping PR comment")

    # --- 4. Cleanup internal registry images (only after successful upload) ---
    image_refs = [r for r in [args.treatment_image_ref, args.control_image_ref] if r]
    if image_refs and upload_ok:
        cleaned = cleanup_images(image_refs)
        logger.info("Cleaned up %d/%d images", cleaned, len(image_refs))
    elif image_refs and not upload_ok:
        logger.warning("Skipping cleanup — reports were not successfully uploaded")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
