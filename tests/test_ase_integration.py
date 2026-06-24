"""Tests for agent-skills-eval (ASE) integration.

Covers:
- EvalEngine enum in schemas.py
- ASE-specific validation in validate.py
- ASE results aggregation in aggregate_ase.py
"""

import json
from pathlib import Path

import pytest
import yaml

from abevalflow.report import AnalysisResult, Provenance
from abevalflow.schemas import EvalEngine, SubmissionMetadata
from scripts.aggregate_ase import build_ase_analysis, render_markdown
from scripts.validate import main as validate_main
from scripts.validate import validate_submission

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_METADATA = {
    "name": "my-skill",
    "description": "A test skill",
    "generation_mode": "manual",
}

SKILL_MD_WITH_FRONTMATTER = """\
---
name: my-skill
description: A test skill for evaluation
---

# My Skill

This is the skill content.
"""

SKILL_MD_NO_FRONTMATTER = """\
# My Skill

This is the skill content without frontmatter.
"""

VALID_EVALS_JSON = {
    "skill_name": "my-skill",
    "evals": [
        {
            "id": "test-eval",
            "name": "Test Eval",
            "prompt": "Explain how to do X.",
            "expected_output": "A thorough explanation of X.",
            "assertions": [
                "The output explains X in detail.",
                "The output provides concrete examples.",
            ],
        }
    ],
}


@pytest.fixture()
def harbor_submission(tmp_path: Path) -> Path:
    """Create a valid Harbor-only submission."""
    sub = tmp_path / "my-skill"
    sub.mkdir()
    (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
    (sub / "instruction.md").write_text("Do the thing.")
    skills = sub / "skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text(SKILL_MD_WITH_FRONTMATTER)
    tests = sub / "tests"
    tests.mkdir()
    (tests / "test_outputs.py").write_text("def test_x():\n    assert True\n")
    return sub


@pytest.fixture()
def ase_submission(tmp_path: Path) -> Path:
    """Create a valid ASE-only submission."""
    sub = tmp_path / "my-skill"
    sub.mkdir()
    (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
    skills = sub / "skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text(SKILL_MD_WITH_FRONTMATTER)
    evals_dir = sub / "evals"
    evals_dir.mkdir()
    (evals_dir / "evals.json").write_text(json.dumps(VALID_EVALS_JSON, indent=2))
    return sub


@pytest.fixture()
def both_submission(harbor_submission: Path) -> Path:
    """Create a submission valid for both engines."""
    evals_dir = harbor_submission / "evals"
    evals_dir.mkdir()
    (evals_dir / "evals.json").write_text(json.dumps(VALID_EVALS_JSON, indent=2))
    return harbor_submission


@pytest.fixture()
def ase_results_dir(tmp_path: Path) -> Path:
    """Create mock ASE results across 3 iterations."""
    results = tmp_path / "ase-results"
    for i in range(1, 4):
        iter_dir = results / f"iteration-{i}"
        skill_dir = iter_dir / "eval-my-skill"

        ws = skill_dir / "with_skill"
        ws.mkdir(parents=True)
        pass_rate = 0.8 + (i * 0.04)
        (ws / "grading.json").write_text(
            json.dumps(
                {
                    "assertion_results": [
                        {"text": "A1", "passed": True, "evidence": "ok"},
                        {"text": "A2", "passed": True, "evidence": "ok"},
                        {"text": "A3", "passed": True, "evidence": "ok"},
                        {"text": "A4", "passed": i != 2, "evidence": "ok"},
                        {"text": "A5", "passed": i == 3, "evidence": "ok"},
                    ],
                    "summary": {
                        "passed": 3 + (1 if i != 2 else 0) + (1 if i == 3 else 0),
                        "failed": 5 - (3 + (1 if i != 2 else 0) + (1 if i == 3 else 0)),
                        "total": 5,
                        "pass_rate": pass_rate,
                    },
                }
            )
        )

        wos = skill_dir / "without_skill"
        wos.mkdir(parents=True)
        (wos / "grading.json").write_text(
            json.dumps(
                {
                    "assertion_results": [
                        {"text": "A1", "passed": True, "evidence": "ok"},
                        {"text": "A2", "passed": False, "evidence": "no"},
                        {"text": "A3", "passed": False, "evidence": "no"},
                        {"text": "A4", "passed": False, "evidence": "no"},
                        {"text": "A5", "passed": False, "evidence": "no"},
                    ],
                    "summary": {
                        "passed": 1,
                        "failed": 4,
                        "total": 5,
                        "pass_rate": 0.2,
                    },
                }
            )
        )

        (iter_dir / "benchmark.json").write_text(
            json.dumps(
                {
                    "run_summary": {
                        "with_skill": {"pass_rate": {"mean": pass_rate, "stddev": 0}},
                        "without_skill": {"pass_rate": {"mean": 0.2, "stddev": 0}},
                        "delta": {"pass_rate": pass_rate - 0.2},
                    },
                }
            )
        )

    return results


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestEvalEngineEnum:
    def test_enum_values(self) -> None:
        assert EvalEngine.HARBOR == "harbor"
        assert EvalEngine.ASE == "ase"
        assert EvalEngine.BOTH == "both"

    def test_metadata_default_harbor(self) -> None:
        model = SubmissionMetadata(**VALID_METADATA)
        assert model.eval_engine == EvalEngine.HARBOR

    def test_metadata_ase(self) -> None:
        meta = {**VALID_METADATA, "eval_engine": "ase"}
        model = SubmissionMetadata(**meta)
        assert model.eval_engine == EvalEngine.ASE

    def test_metadata_both(self) -> None:
        meta = {**VALID_METADATA, "eval_engine": "both"}
        model = SubmissionMetadata(**meta)
        assert model.eval_engine == EvalEngine.BOTH

    def test_metadata_invalid_engine(self) -> None:
        meta = {**VALID_METADATA, "eval_engine": "tessl"}
        with pytest.raises(Exception):
            SubmissionMetadata(**meta)


class TestProvenanceEvalEngine:
    def test_default_harbor(self) -> None:
        p = Provenance()
        assert p.eval_engine == "harbor"

    def test_set_ase(self) -> None:
        p = Provenance(eval_engine="ase")
        assert p.eval_engine == "ase"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestAseValidation:
    def test_harbor_mode_skips_ase_checks(self, harbor_submission: Path) -> None:
        errors = validate_submission(harbor_submission, eval_engine=EvalEngine.HARBOR)
        assert errors == []

    def test_harbor_mode_missing_instruction(self, ase_submission: Path) -> None:
        errors = validate_submission(ase_submission, eval_engine=EvalEngine.HARBOR)
        assert any("instruction.md is missing" in e for e in errors)

    def test_ase_mode_valid(self, ase_submission: Path) -> None:
        errors = validate_submission(ase_submission, eval_engine=EvalEngine.ASE)
        assert errors == []

    def test_ase_mode_skips_harbor_checks(self, ase_submission: Path) -> None:
        errors = validate_submission(ase_submission, eval_engine=EvalEngine.ASE)
        assert not any("instruction.md" in e for e in errors)
        assert not any("test_outputs.py" in e for e in errors)

    def test_ase_mode_missing_evals_json_is_ok(self, tmp_path: Path) -> None:
        """Missing evals.json is NOT an error - pipeline will generate it."""
        sub = tmp_path / "skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text(SKILL_MD_WITH_FRONTMATTER)
        errors = validate_submission(sub, eval_engine=EvalEngine.ASE)
        # Missing evals.json should NOT cause validation error
        assert not any("evals" in e.lower() for e in errors)
        assert errors == []  # Should pass validation

    def test_ase_mode_missing_frontmatter(self, tmp_path: Path) -> None:
        sub = tmp_path / "skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text(SKILL_MD_NO_FRONTMATTER)
        evals_dir = sub / "evals"
        evals_dir.mkdir()
        (evals_dir / "evals.json").write_text(json.dumps(VALID_EVALS_JSON))
        errors = validate_submission(sub, eval_engine=EvalEngine.ASE)
        assert any("missing YAML frontmatter" in e for e in errors)

    def test_ase_mode_missing_name_in_frontmatter(self, tmp_path: Path) -> None:
        sub = tmp_path / "skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text("---\ndescription: no name\n---\nContent")
        evals_dir = sub / "evals"
        evals_dir.mkdir()
        (evals_dir / "evals.json").write_text(json.dumps(VALID_EVALS_JSON))
        errors = validate_submission(sub, eval_engine=EvalEngine.ASE)
        assert any("missing required 'name'" in e for e in errors)

    def test_ase_mode_invalid_evals_json(self, tmp_path: Path) -> None:
        sub = tmp_path / "skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text(SKILL_MD_WITH_FRONTMATTER)
        evals_dir = sub / "evals"
        evals_dir.mkdir()
        (evals_dir / "evals.json").write_text("not json")
        errors = validate_submission(sub, eval_engine=EvalEngine.ASE)
        assert any("not valid JSON" in e for e in errors)

    def test_ase_mode_empty_evals_array(self, tmp_path: Path) -> None:
        sub = tmp_path / "skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text(SKILL_MD_WITH_FRONTMATTER)
        evals_dir = sub / "evals"
        evals_dir.mkdir()
        (evals_dir / "evals.json").write_text(json.dumps({"evals": []}))
        errors = validate_submission(sub, eval_engine=EvalEngine.ASE)
        assert any("non-empty 'evals' array" in e for e in errors)

    def test_ase_mode_eval_missing_prompt(self, tmp_path: Path) -> None:
        sub = tmp_path / "skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text(SKILL_MD_WITH_FRONTMATTER)
        evals_dir = sub / "evals"
        evals_dir.mkdir()
        bad_evals = {"evals": [{"assertions": ["something"]}]}
        (evals_dir / "evals.json").write_text(json.dumps(bad_evals))
        errors = validate_submission(sub, eval_engine=EvalEngine.ASE)
        assert any("missing required 'prompt'" in e for e in errors)

    def test_ase_mode_eval_no_assertions_or_expected(self, tmp_path: Path) -> None:
        sub = tmp_path / "skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text(SKILL_MD_WITH_FRONTMATTER)
        evals_dir = sub / "evals"
        evals_dir.mkdir()
        bad_evals = {"evals": [{"prompt": "test"}]}
        (evals_dir / "evals.json").write_text(json.dumps(bad_evals))
        errors = validate_submission(sub, eval_engine=EvalEngine.ASE)
        assert any("'assertions' or 'expected_output'" in e for e in errors)

    def test_both_mode_valid(self, both_submission: Path) -> None:
        errors = validate_submission(both_submission, eval_engine=EvalEngine.BOTH)
        assert errors == []

    def test_both_mode_missing_evals_is_ok(self, harbor_submission: Path) -> None:
        """Missing evals.json is NOT an error - pipeline will generate it."""
        errors = validate_submission(harbor_submission, eval_engine=EvalEngine.BOTH)
        # Missing evals.json should NOT cause validation error (will be generated)
        assert not any("evals/evals.json is missing" in e for e in errors)

    def test_cli_eval_engine_flag(
        self,
        ase_submission: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = validate_main([str(ase_submission), "--eval-engine", "ase"])
        assert rc == 0
        output = json.loads(capsys.readouterr().out)
        assert output["valid"] is True


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------


class TestAseAggregation:
    def test_build_analysis_basic(self, ase_results_dir: Path) -> None:
        result = build_ase_analysis(
            results_dir=ase_results_dir,
            submission_name="my-skill",
            n_iterations=3,
        )
        assert result.submission_name == "my-skill"
        assert result.provenance.eval_engine == "ase"
        assert result.summary.treatment.n_trials == 3
        assert result.summary.control.n_trials == 3
        assert result.summary.treatment.mean_reward is not None
        assert result.summary.control.mean_reward is not None
        assert result.summary.treatment.mean_reward > result.summary.control.mean_reward
        assert result.summary.mean_reward_gap is not None
        assert result.summary.mean_reward_gap > 0
        assert result.summary.recommendation.value == "pass"

    def test_build_analysis_with_provenance(self, ase_results_dir: Path) -> None:
        prov = Provenance(commit_sha="abc123", pipeline_run_id="run-1")
        result = build_ase_analysis(
            results_dir=ase_results_dir,
            submission_name="my-skill",
            n_iterations=3,
            provenance=prov,
        )
        assert result.provenance.commit_sha == "abc123"
        assert result.provenance.pipeline_run_id == "run-1"
        assert result.provenance.eval_engine == "ase"

    def test_build_analysis_serializable(self, ase_results_dir: Path) -> None:
        result = build_ase_analysis(
            results_dir=ase_results_dir,
            submission_name="my-skill",
            n_iterations=3,
        )
        json_str = result.model_dump_json(indent=2)
        roundtrip = AnalysisResult.model_validate_json(json_str)
        assert roundtrip.submission_name == "my-skill"
        assert roundtrip.provenance.eval_engine == "ase"

    def test_build_analysis_empty_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = build_ase_analysis(
            results_dir=empty_dir,
            submission_name="empty",
            n_iterations=3,
        )
        assert result.summary.recommendation.value == "fail"
        assert result.summary.treatment.n_trials == 0

    def test_render_markdown(self, ase_results_dir: Path) -> None:
        result = build_ase_analysis(
            results_dir=ase_results_dir,
            submission_name="my-skill",
            n_iterations=3,
        )
        md = render_markdown(result)
        assert "agent-skills-eval" in md
        assert "my-skill" in md
        assert "With Skill" in md
        assert "Without Skill" in md
        assert "Recommendation" in md

    def test_statistical_tests(self, ase_results_dir: Path) -> None:
        result = build_ase_analysis(
            results_dir=ase_results_dir,
            submission_name="my-skill",
            n_iterations=3,
        )
        assert result.summary.ttest_p_value is not None
        assert result.summary.fisher_p_value is not None
        assert result.summary.mean_reward_gap is not None
        assert result.summary.mean_reward_gap > 0

    def test_threshold_fail(self, ase_results_dir: Path) -> None:
        result = build_ase_analysis(
            results_dir=ase_results_dir,
            submission_name="my-skill",
            n_iterations=3,
            threshold=0.99,
        )
        assert result.summary.recommendation.value == "fail"
