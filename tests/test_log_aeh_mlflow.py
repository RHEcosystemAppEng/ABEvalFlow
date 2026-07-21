"""Tests for scripts/log_aeh_mlflow.py."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.log_aeh_mlflow import main


def _write_run(tmp_path: Path, *, skill: str = "demo-skill", run_id: str = "run-1") -> tuple[Path, Path]:
    runs = tmp_path / "reports"
    run_dir = runs / skill / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_result.json").write_text(
        json.dumps(
            {
                "execution_mode": "harbor",
                "mean_reward": 0.8,
                "cost_usd": 0.12,
                "token_usage": {"input": 100, "output": 50},
                "model": "claude-sonnet",
            }
        )
    )
    (run_dir / "summary.yaml").write_text(yaml.safe_dump({"mean_reward": 0.8}))
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "skill": skill,
                "mlflow": {"experiment": "aeh-demo"},
                "models": {"skill": "claude-sonnet"},
                "judges": [{"name": "ok", "check": "return True, 'ok'"}],
            }
        )
    )
    return runs, config


def test_skips_when_disabled(tmp_path: Path) -> None:
    runs, config = _write_run(tmp_path)
    rc = main(
        [
            "--run-id",
            "run-1",
            "--config",
            str(config),
            "--runs-dir",
            str(runs),
            "--tracking-uri",
            "http://mlflow.example:5000",
            "--enabled",
            "false",
        ]
    )
    assert rc == 0


def test_skips_when_uri_empty(tmp_path: Path) -> None:
    runs, config = _write_run(tmp_path)
    rc = main(
        [
            "--run-id",
            "run-1",
            "--config",
            str(config),
            "--runs-dir",
            str(runs),
            "--tracking-uri",
            "",
            "--enabled",
            "true",
        ]
    )
    assert rc == 0


def test_minimal_logger_with_mock_mlflow(tmp_path: Path, monkeypatch) -> None:
    runs, config = _write_run(tmp_path)

    class _FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    calls: dict[str, list] = {"params": [], "metrics": [], "artifacts": []}

    class _FakeMlflow:
        def set_tracking_uri(self, uri):
            calls["uri"] = uri

        def set_experiment(self, name):
            calls["experiment"] = name

        def start_run(self, run_name=None):
            calls["run_name"] = run_name
            return _FakeRun()

        def log_param(self, k, v):
            calls["params"].append((k, v))

        def log_metric(self, k, v):
            calls["metrics"].append((k, v))

        def log_artifact(self, path):
            calls["artifacts"].append(path)

    import sys

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow())

    rc = main(
        [
            "--run-id",
            "run-1",
            "--config",
            str(config),
            "--runs-dir",
            str(runs),
            "--tracking-uri",
            "http://mlflow.example:5000",
            "--enabled",
            "true",
        ]
    )
    assert rc == 0
    assert calls["uri"] == "http://mlflow.example:5000"
    assert calls["experiment"] == "aeh-demo"
    assert ("mean_reward", 0.8) in calls["metrics"]
    assert ("tokens_input", 100.0) in calls["metrics"]
