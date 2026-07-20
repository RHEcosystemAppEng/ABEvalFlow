"""Tests for AEH Harbor task enrichment (skills + annotations)."""

from __future__ import annotations

from pathlib import Path

import yaml

from abevalflow.harbor_extensions.aeh_task_enrichment import enrich_harbor_tasks


def _write_generated_task(task_dir: Path, *, workdir: str = "/workspace") -> None:
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text(f'[environment]\ndocker_image = "img:latest"\nworkdir = "{workdir}"\n')
    (task_dir / "tests" / "eval.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": "", "schema": "x"},
                "judges": [{"name": "file_created", "check": "return True, 'ok'"}],
            },
            sort_keys=False,
        )
    )
    (task_dir / "tests" / "test.sh").write_text(
        f"""#!/bin/bash
set -o pipefail
mkdir -p /logs/verifier
python3 -m agent_eval.harbor.reward \\
  --config /tests/eval.yaml \\
  --case-dir "{workdir}" \\
  --out-dir /logs/verifier
"""
    )
    (task_dir / "tests" / "test.sh").chmod(0o755)


def test_enrich_copies_annotations_and_restores_dataset_path(tmp_path: Path):
    submission = tmp_path / "submission"
    cases = submission / "cases" / "case-001"
    cases.mkdir(parents=True)
    (cases / "annotations.yaml").write_text(
        yaml.safe_dump({"expected_file": "greeting.txt", "expected_content": "Hello"})
    )
    (submission / "eval.yaml").write_text(
        yaml.safe_dump(
            {
                "skill": "demo",
                "dataset": {"path": "cases"},
                "runner": {"type": "claude-code"},
                "judges": [{"name": "file_created"}],
            }
        )
    )

    tasks = tmp_path / "tasks"
    task = tasks / "case-001"
    _write_generated_task(task)

    n = enrich_harbor_tasks(tasks, config_path=submission / "eval.yaml")
    assert n == 1

    ann = task / "environment" / "annotations.yaml"
    assert ann.is_file()
    assert yaml.safe_load(ann.read_text())["expected_file"] == "greeting.txt"

    bundled = yaml.safe_load((task / "tests" / "eval.yaml").read_text())
    assert bundled["dataset"]["path"] == "cases"

    test_sh = (task / "tests" / "test.sh").read_text()
    assert "ABEvalFlow: stage annotations" in test_sh
    assert "/tests/cases/" in test_sh


def test_enrich_copies_skills_and_sets_skills_dir(tmp_path: Path):
    submission = tmp_path / "submission"
    skill_dir = submission / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo-skill\n---\n# demo\n")
    (submission / "cases" / "case-001").mkdir(parents=True)
    (submission / "eval.yaml").write_text(
        yaml.safe_dump(
            {
                "skill": "demo-skill",
                "dataset": {"path": "cases"},
                "runner": {"type": "claude-code", "plugin_dirs": ["skills"]},
                "judges": [{"name": "exit_success"}],
            }
        )
    )

    tasks = tmp_path / "tasks"
    task = tasks / "case-001"
    _write_generated_task(task)

    enrich_harbor_tasks(tasks, config_path=submission / "eval.yaml")

    assert (task / "environment" / "skills" / "demo-skill" / "SKILL.md").is_file()
    toml = (task / "task.toml").read_text()
    assert 'skills_dir = "/workspace/skills"' in toml
