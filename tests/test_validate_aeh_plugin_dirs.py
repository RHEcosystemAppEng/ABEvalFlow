"""AEH validation requires plugin_dirs skills when declared."""

from pathlib import Path

from abevalflow.schemas import EvalEngine
from scripts.validate import validate_submission
from tests.test_validate_aeh import VALID_EVAL_YAML, create_aeh_submission


def test_aeh_plugin_dirs_missing_skills_fails(tmp_path: Path):
    eval_yaml = {
        **VALID_EVAL_YAML,
        "skill": "demo-skill",
        "runner": {"type": "claude-code", "plugin_dirs": ["skills"]},
    }
    create_aeh_submission(tmp_path, eval_yaml_content=eval_yaml)
    errors = validate_submission(tmp_path, eval_engine=EvalEngine.AEH)
    assert any("plugin_dirs" in e and "missing" in e for e in errors)


def test_aeh_plugin_dirs_with_named_skill_passes(tmp_path: Path):
    eval_yaml = {
        **VALID_EVAL_YAML,
        "skill": "demo-skill",
        "runner": {"type": "claude-code", "plugin_dirs": ["skills"]},
    }
    create_aeh_submission(tmp_path, eval_yaml_content=eval_yaml)
    skill = tmp_path / "skills" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo-skill\n---\n")
    errors = validate_submission(tmp_path, eval_engine=EvalEngine.AEH)
    assert not any("plugin_dirs" in e or "SKILL.md" in e for e in errors)
