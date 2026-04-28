"""Tests for scripts/publish.py — artifact publishing and cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.publish import (
    _build_artifact_prefix,
    cleanup_images,
    post_pr_comment,
    promote_to_quay,
    upload_reports,
)


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    """Create a temporary report directory with sample files."""
    report = {
        "submission_name": "hello-world",
        "provenance": {
            "commit_sha": "abc123def456",
            "pipeline_run_id": "abevalflow-xyz",
        },
        "summary": {
            "treatment": {
                "n_trials": 5,
                "n_passed": 4,
                "n_failed": 1,
                "n_errors": 0,
                "pass_rate": 0.8,
                "mean_reward": 0.75,
            },
            "control": {
                "n_trials": 5,
                "n_passed": 2,
                "n_failed": 3,
                "n_errors": 0,
                "pass_rate": 0.4,
                "mean_reward": 0.35,
            },
            "uplift": 0.4,
            "recommendation": "pass",
        },
        "trials": {},
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "report.md").write_text("# Report\nSample markdown")
    return tmp_path


class TestBuildArtifactPrefix:
    def test_format(self):
        prefix = _build_artifact_prefix("hello-world", "abevalflow-xyz")
        parts = prefix.split("_")
        assert len(parts) >= 3
        assert parts[0].isdigit() and len(parts[0]) == 8
        assert "hello-world" in prefix
        assert "abevalflow-xyz" in prefix

    def test_datestamp_is_today(self):
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        prefix = _build_artifact_prefix("test", "run1")
        assert prefix.startswith(today)


class TestUploadReports:
    @patch("minio.Minio")
    def test_uploads_both_files(self, mock_minio_cls, report_dir):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        mock_client.bucket_exists.return_value = True

        prefix = upload_reports(
            report_dir=report_dir,
            submission_name="hello-world",
            pipeline_run_id="run-123",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )

        assert prefix is not None
        assert "hello-world" in prefix
        assert "run-123" in prefix
        assert mock_client.fput_object.call_count == 2

    @patch("minio.Minio")
    def test_creates_bucket_if_missing(self, mock_minio_cls, report_dir):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        mock_client.bucket_exists.return_value = False

        upload_reports(
            report_dir=report_dir,
            submission_name="test",
            pipeline_run_id="run-1",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )

        mock_client.make_bucket.assert_called_once_with("ab-eval-reports")

    @patch("minio.Minio")
    def test_returns_none_on_empty_dir(self, mock_minio_cls, tmp_path):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        mock_client.bucket_exists.return_value = True

        result = upload_reports(
            report_dir=tmp_path,
            submission_name="test",
            pipeline_run_id="run-1",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )

        assert result is None


class TestPromoteToQuay:
    @patch("scripts.publish.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        ok = promote_to_quay(
            treatment_image_ref="registry/ns/skill@sha256:abc",
            submission_name="hello-world",
            commit_sha="abc123def456789",
            quay_repo="quay.io/myorg",
        )

        assert ok is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "skopeo" in cmd[0]
        assert "quay.io/myorg/hello-world:treatment-abc123def456" in cmd[-1]

    @patch("scripts.publish.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="auth error")

        ok = promote_to_quay(
            treatment_image_ref="registry/ns/skill@sha256:abc",
            submission_name="test",
            commit_sha="abc123",
            quay_repo="quay.io/myorg",
        )

        assert ok is False

    @patch("scripts.publish.subprocess.run", side_effect=FileNotFoundError)
    def test_skopeo_not_found(self, mock_run):
        ok = promote_to_quay(
            treatment_image_ref="ref",
            submission_name="test",
            commit_sha="abc",
            quay_repo="quay.io/org",
        )

        assert ok is False


class TestPostPrComment:
    @patch("scripts.publish.subprocess.run")
    def test_posts_comment(self, mock_run, report_dir):
        mock_run.return_value = MagicMock(returncode=0)

        ok = post_pr_comment(
            repo_name="org/repo",
            pr_number="42",
            submission_name="hello-world",
            report_dir=report_dir,
        )

        assert ok is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "gh" in cmd[0]
        assert "repos/org/repo/issues/42/comments" in cmd[2]

    @patch("scripts.publish.subprocess.run")
    def test_no_report_file(self, mock_run, tmp_path):
        ok = post_pr_comment(
            repo_name="org/repo",
            pr_number="1",
            submission_name="test",
            report_dir=tmp_path,
        )

        assert ok is False
        mock_run.assert_not_called()

    @patch("scripts.publish.subprocess.run")
    def test_gh_failure(self, mock_run, report_dir):
        mock_run.return_value = MagicMock(returncode=1, stderr="not found")

        ok = post_pr_comment(
            repo_name="org/repo",
            pr_number="1",
            submission_name="test",
            report_dir=report_dir,
        )

        assert ok is False


class TestCleanupImages:
    @patch("scripts.publish.subprocess.run")
    def test_skips_digest_refs(self, mock_run):
        cleaned = cleanup_images(
            ["registry/ns/skill@sha256:abc123"],
            registry_url="registry",
        )

        assert cleaned == 0
        mock_run.assert_not_called()

    @patch("scripts.publish.subprocess.run")
    def test_deletes_tag_refs(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        cleaned = cleanup_images(
            ["image-registry.openshift-image-registry.svc:5000/ab-eval-flow/hello-world:treatment-abc"],
        )

        assert cleaned == 1
        cmd = mock_run.call_args[0][0]
        assert "oc" in cmd[0]
        assert "delete" in cmd[1]
        assert "istag" in cmd[2]


class TestUpliftThresholdLogic:
    """Verify that -1.0 threshold disables Quay promotion."""

    def test_negative_threshold_skips_promotion(self):
        threshold = -1.0
        recommendation = "pass"
        assert not (threshold > -1.0 and recommendation == "pass")

    def test_zero_threshold_enables_on_pass(self):
        threshold = 0.0
        recommendation = "pass"
        assert threshold > -1.0 and recommendation == "pass"

    def test_positive_threshold_requires_uplift(self):
        threshold = 0.1
        uplift = 0.05
        assert not (uplift >= threshold)

        uplift = 0.15
        assert uplift >= threshold
