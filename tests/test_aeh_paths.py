"""Tests for AEH harbor_job_dir path rewriting."""

from __future__ import annotations

import json
from pathlib import Path

from abevalflow.harbor_extensions.aeh_paths import rewrite_harbor_job_dir


def test_rewrite_absolute_harbor_job_dir_to_relative(tmp_path: Path):
    source = tmp_path / "source"
    job = source / "_eval_tmp" / "aeh-jobs" / "2026-07-21__10-18-16"
    job.mkdir(parents=True)
    run_dir = source / "reports" / "aeh-hello-world" / "run-1"
    run_dir.mkdir(parents=True)
    rr = run_dir / "run_result.json"
    rr.write_text(json.dumps({"harbor_job_dir": str(job.resolve()), "mean_reward": 0.0}))

    rewritten = rewrite_harbor_job_dir(rr)
    assert rewritten == "_eval_tmp/aeh-jobs/2026-07-21__10-18-16"

    data = json.loads(rr.read_text())
    assert data["harbor_job_dir"] == rewritten
    assert not Path(data["harbor_job_dir"]).is_absolute()


def test_rewrite_leaves_relative_path_alone(tmp_path: Path):
    run_dir = tmp_path / "reports" / "skill" / "run-1"
    run_dir.mkdir(parents=True)
    rr = run_dir / "run_result.json"
    rr.write_text(json.dumps({"harbor_job_dir": "_eval_tmp/aeh-jobs/job-1"}))

    assert rewrite_harbor_job_dir(rr) is None
    assert json.loads(rr.read_text())["harbor_job_dir"] == "_eval_tmp/aeh-jobs/job-1"


def test_rewrite_missing_file_returns_none(tmp_path: Path):
    assert rewrite_harbor_job_dir(tmp_path / "missing.json") is None
