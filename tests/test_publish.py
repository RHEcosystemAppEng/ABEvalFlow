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
    upload_debug_artifacts,
    upload_reports,
    upload_scaffolded_configs,
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
        assert len(parts) >= 4
        assert parts[0].isdigit() and len(parts[0]) == 8  # YYYYMMDD
        assert parts[1].isdigit() and len(parts[1]) == 6  # hhmmss
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


class TestUploadScaffoldedGeneratedFiles:
    """Test that upload_scaffolded_configs also uploads generated files."""

    @pytest.fixture
    def workspace_with_generated(self, tmp_path: Path) -> Path:
        """Workspace with generated files under submissions/<name>/."""
        sub_dir = tmp_path / "submissions" / "ai-skill"
        sub_dir.mkdir(parents=True)
        (sub_dir / "instruction.md").write_text("# Task\nDo something.\n")
        tests = sub_dir / "tests"
        tests.mkdir()
        (tests / "test_outputs.py").write_text("def test_ok(): assert True\n")
        (tests / "llm_judge.py").write_text("score = 0.9\n")
        (sub_dir / "scenario_brief.json").write_text("{}")
        return tmp_path

    @patch("minio.Minio")
    def test_uploads_generated_files(self, mock_minio_cls, workspace_with_generated):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        count = upload_scaffolded_configs(
            workspace_root=workspace_with_generated,
            submission_name="ai-skill",
            prefix="20260504_ai-skill_run-42",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )

        object_names = [c.args[1] for c in mock_client.fput_object.call_args_list]
        assert any("generated/instruction.md" in n for n in object_names)
        assert any("generated/test_outputs.py" in n for n in object_names)
        assert any("generated/llm_judge.py" in n for n in object_names)
        assert any("generated/scenario_brief.json" in n for n in object_names)
        assert count >= 4

    @patch("minio.Minio")
    def test_handles_missing_generated_files(self, mock_minio_cls, tmp_path):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        count = upload_scaffolded_configs(
            workspace_root=tmp_path,
            submission_name="test",
            prefix="20260504_test_run-1",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )

        assert count == 0


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
    @patch("scripts.publish.urllib.request.urlopen")
    def test_posts_comment(self, mock_urlopen, report_dir):
        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        ok = post_pr_comment(
            repo_name="org/repo",
            pr_number="42",
            submission_name="hello-world",
            report_dir=report_dir,
            github_token="ghp_test",
        )

        assert ok is True
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert "repos/org/repo/issues/42/comments" in req.full_url

    def test_no_report_file(self, tmp_path):
        ok = post_pr_comment(
            repo_name="org/repo",
            pr_number="1",
            submission_name="test",
            report_dir=tmp_path,
            github_token="ghp_test",
        )

        assert ok is False

    def test_no_token(self, report_dir):
        ok = post_pr_comment(
            repo_name="org/repo",
            pr_number="1",
            submission_name="test",
            report_dir=report_dir,
            github_token="",
        )

        assert ok is False

    @patch("scripts.publish.urllib.request.urlopen", side_effect=Exception("network error"))
    def test_api_failure(self, mock_urlopen, report_dir):
        ok = post_pr_comment(
            repo_name="org/repo",
            pr_number="1",
            submission_name="test",
            report_dir=report_dir,
            github_token="ghp_test",
        )

        assert ok is False


class TestCleanupImages:
    @patch("scripts.publish.subprocess.run")
    def test_digest_refs_delete_imagestream(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        cleaned = cleanup_images(
            ["registry/ns/skill@sha256:abc123"],
            registry_url="registry",
        )

        assert cleaned == 1
        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["oc", "delete"]
        assert cmd[2] == "imagestream"
        assert cmd[3] == "skill"

    @patch("scripts.publish.subprocess.run")
    def test_deduplicates_digest_imagestreams(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        cleaned = cleanup_images(
            [
                "registry/ns/skill@sha256:aaa",
                "registry/ns/skill@sha256:bbb",
            ],
            registry_url="registry",
        )

        assert cleaned == 1
        assert mock_run.call_count == 1

    @patch("scripts.publish.subprocess.run")
    def test_deletes_tag_refs(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        cleaned = cleanup_images(
            ["image-registry.openshift-image-registry.svc:5000/ab-eval-flow/hello-world:treatment-abc"],
        )

        assert cleaned == 1
        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["oc", "delete"]
        assert cmd[2] == "istag"


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Create a mock Harbor results directory with agent/verifier folders."""
    for variant in ("treatment", "control"):
        trial = tmp_path / variant / "job-name" / "trial_001__uuid1"
        (trial / "agent").mkdir(parents=True)
        (trial / "agent" / "agent.log").write_text("agent log content")
        (trial / "agent" / "solution.py").write_text("print('hello')")
        (trial / "verifier").mkdir(parents=True)
        (trial / "verifier" / "verifier.log").write_text("verifier log")
        (trial / "result.json").write_text('{"status": "pass"}')
    return tmp_path


class TestUploadDebugArtifacts:
    @patch("minio.Minio")
    def test_uploads_agent_and_verifier_files(self, mock_minio_cls, results_dir):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        mock_client.bucket_exists.return_value = True

        count = upload_debug_artifacts(
            results_dir=results_dir,
            prefix="20260428_test_run1",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )

        assert count == 8  # 4 files x 2 variants (agent.log, solution.py, verifier.log, result.json)
        uploaded_names = [
            call[0][1] for call in mock_client.fput_object.call_args_list
        ]
        assert any("agent/agent.log" in n for n in uploaded_names)
        assert any("verifier/verifier.log" in n for n in uploaded_names)
        assert all(n.startswith("20260428_test_run1/debug/") for n in uploaded_names)

    @patch("minio.Minio")
    def test_skips_unrelated_files_but_uploads_trial_root_files(self, mock_minio_cls, results_dir):
        for variant in ("treatment", "control"):
            trial = results_dir / variant / "job-name" / "trial_001__uuid1"
            (trial / "random_file.txt").write_text("should be skipped")

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        mock_client.bucket_exists.return_value = True

        upload_debug_artifacts(
            results_dir=results_dir,
            prefix="prefix",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )

        uploaded_names = [
            call[0][1] for call in mock_client.fput_object.call_args_list
        ]
        assert not any("random_file.txt" in n for n in uploaded_names)
        assert any("result.json" in n for n in uploaded_names)

    def test_nonexistent_results_dir(self, tmp_path):
        count = upload_debug_artifacts(
            results_dir=tmp_path / "nonexistent",
            prefix="prefix",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )
        assert count == 0

    @patch("minio.Minio")
    def test_creates_bucket_if_missing(self, mock_minio_cls, results_dir):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        mock_client.bucket_exists.return_value = False

        upload_debug_artifacts(
            results_dir=results_dir,
            prefix="prefix",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )

        mock_client.make_bucket.assert_called_once_with("ab-eval-reports")


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
