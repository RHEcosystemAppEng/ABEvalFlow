"""Tests for abevalflow/experiment.py — experiment strategy implementations."""

from pathlib import Path

import pytest

from abevalflow.experiment import (
    ConfigDrivenStrategy,
    ModelExperimentStrategy,
    SkillExperimentStrategy,
    get_strategy,
)
from abevalflow.schemas import (
    CopySpec,
    ExperimentConfig,
    ExperimentType,
    VariantSpec,
)


@pytest.fixture()
def submission_dir(tmp_path: Path) -> Path:
    """Minimal submission with skills/, docs/, tests/, supportive/."""
    sub = tmp_path / "my-skill"
    sub.mkdir()
    (sub / "skills").mkdir()
    (sub / "docs").mkdir()
    (sub / "tests").mkdir()
    (sub / "supportive").mkdir()
    return sub


@pytest.fixture()
def submission_dir_no_docs(tmp_path: Path) -> Path:
    """Submission with skills/ but no docs/."""
    sub = tmp_path / "my-skill"
    sub.mkdir()
    (sub / "skills").mkdir()
    (sub / "tests").mkdir()
    return sub


# ──────────────────────────────────────────────
# get_strategy factory
# ──────────────────────────────────────────────


class TestGetStrategy:
    def test_skill_type_returns_skill_strategy(self) -> None:
        config = ExperimentConfig(type="skill")
        strategy = get_strategy(config)
        assert isinstance(strategy, SkillExperimentStrategy)

    def test_model_type_returns_model_strategy(self) -> None:
        config = ExperimentConfig(type="model")
        strategy = get_strategy(config)
        assert isinstance(strategy, ModelExperimentStrategy)

    def test_prompt_type_returns_skill_strategy(self) -> None:
        config = ExperimentConfig(type="prompt")
        strategy = get_strategy(config)
        assert isinstance(strategy, SkillExperimentStrategy)

    def test_custom_type_returns_config_driven(self) -> None:
        config = ExperimentConfig(type="custom")
        strategy = get_strategy(config)
        assert isinstance(strategy, ConfigDrivenStrategy)

    def test_default_config_returns_skill_strategy(self) -> None:
        config = ExperimentConfig()
        strategy = get_strategy(config)
        assert isinstance(strategy, SkillExperimentStrategy)


# ──────────────────────────────────────────────
# SkillExperimentStrategy
# ──────────────────────────────────────────────


class TestSkillExperimentStrategy:
    def test_treatment_copy_specs_include_skills_and_docs(
        self, submission_dir: Path,
    ) -> None:
        config = ExperimentConfig()
        strategy = SkillExperimentStrategy(config)
        specs = strategy.variant_copy_specs(submission_dir, "treatment")
        srcs = [s.src for s in specs]
        assert "skills" in srcs
        assert "docs" in srcs

    def test_control_copy_specs_empty_by_default(
        self, submission_dir: Path,
    ) -> None:
        config = ExperimentConfig()
        strategy = SkillExperimentStrategy(config)
        specs = strategy.variant_copy_specs(submission_dir, "control")
        assert specs == []

    def test_treatment_skips_missing_dirs(
        self, submission_dir_no_docs: Path,
    ) -> None:
        config = ExperimentConfig()
        strategy = SkillExperimentStrategy(config)
        specs = strategy.variant_copy_specs(submission_dir_no_docs, "treatment")
        srcs = [s.src for s in specs]
        assert "skills" in srcs
        assert "docs" not in srcs

    def test_treatment_context_has_skills_dir(self, submission_dir: Path) -> None:
        config = ExperimentConfig()
        strategy = SkillExperimentStrategy(config)
        ctx = strategy.customize_context({}, "treatment", submission_dir)
        assert ctx["skills_dir"] == "/skills"
        assert ("skills", "/skills") in ctx["copy_pairs"]

    def test_control_context_no_skills_dir(self, submission_dir: Path) -> None:
        config = ExperimentConfig()
        strategy = SkillExperimentStrategy(config)
        ctx = strategy.customize_context({}, "control", submission_dir)
        assert ctx["skills_dir"] is None
        assert ctx["copy_pairs"] == []

    def test_treatment_context_preserves_base(self, submission_dir: Path) -> None:
        config = ExperimentConfig()
        strategy = SkillExperimentStrategy(config)
        base = {"skill_name": "test", "persona": "sre"}
        ctx = strategy.customize_context(base, "treatment", submission_dir)
        assert ctx["skill_name"] == "test"
        assert ctx["persona"] == "sre"

    def test_base_context_not_mutated(self, submission_dir: Path) -> None:
        config = ExperimentConfig()
        strategy = SkillExperimentStrategy(config)
        base = {"key": "value"}
        strategy.customize_context(base, "treatment", submission_dir)
        assert "skills_dir" not in base

    def test_treatment_context_filters_missing_docs(
        self, submission_dir_no_docs: Path,
    ) -> None:
        """copy_pairs must not include dirs that don't exist in submission."""
        config = ExperimentConfig()
        strategy = SkillExperimentStrategy(config)
        ctx = strategy.customize_context({}, "treatment", submission_dir_no_docs)
        srcs = [src for src, _ in ctx["copy_pairs"]]
        assert "skills" in srcs
        assert "docs" not in srcs


# ──────────────────────────────────────────────
# ModelExperimentStrategy
# ──────────────────────────────────────────────


class TestModelExperimentStrategy:
    def _model_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            type="model",
            n_trials=10,
            treatment=VariantSpec(
                env_from_secrets={"MODEL_ENDPOINT": "models/endpoint-a"},
            ),
            control=VariantSpec(
                env_from_secrets={"MODEL_ENDPOINT": "models/endpoint-b"},
            ),
        )

    def test_treatment_copy_specs_empty(self, submission_dir: Path) -> None:
        strategy = ModelExperimentStrategy(self._model_config())
        specs = strategy.variant_copy_specs(submission_dir, "treatment")
        assert specs == []

    def test_control_copy_specs_empty(self, submission_dir: Path) -> None:
        strategy = ModelExperimentStrategy(self._model_config())
        specs = strategy.variant_copy_specs(submission_dir, "control")
        assert specs == []

    def test_treatment_context_has_env_from_secrets(self, submission_dir: Path) -> None:
        strategy = ModelExperimentStrategy(self._model_config())
        ctx = strategy.customize_context({}, "treatment", submission_dir)
        assert ctx["env_from_secrets"]["MODEL_ENDPOINT"] == "models/endpoint-a"

    def test_control_context_has_env_from_secrets(self, submission_dir: Path) -> None:
        strategy = ModelExperimentStrategy(self._model_config())
        ctx = strategy.customize_context({}, "control", submission_dir)
        assert ctx["env_from_secrets"]["MODEL_ENDPOINT"] == "models/endpoint-b"

    def test_no_skills_dir_for_model_experiment(self, submission_dir: Path) -> None:
        strategy = ModelExperimentStrategy(self._model_config())
        ctx = strategy.customize_context({}, "treatment", submission_dir)
        assert ctx["skills_dir"] is None

    def test_model_with_copy_specs(self, submission_dir: Path) -> None:
        config = ExperimentConfig(
            type="model",
            treatment=VariantSpec(
                copy=[CopySpec(src="skills", dest="/skills")],
                env_from_secrets={"MODEL": "models/a"},
            ),
            control=VariantSpec(
                copy=[CopySpec(src="skills", dest="/skills")],
                env_from_secrets={"MODEL": "models/b"},
            ),
        )
        strategy = ModelExperimentStrategy(config)
        specs = strategy.variant_copy_specs(submission_dir, "treatment")
        assert len(specs) == 1
        ctx = strategy.customize_context({}, "treatment", submission_dir)
        assert ctx["skills_dir"] == "/skills"

    def test_model_filters_missing_dirs_in_context(self, submission_dir: Path) -> None:
        """copy_pairs must not include dirs missing from submission."""
        config = ExperimentConfig(
            type="model",
            treatment=VariantSpec(
                copy=[CopySpec(src="nonexistent", dest="/nonexistent")],
                env_from_secrets={"MODEL": "models/a"},
            ),
        )
        strategy = ModelExperimentStrategy(config)
        ctx = strategy.customize_context({}, "treatment", submission_dir)
        assert ctx["copy_pairs"] == []
        assert ctx["skills_dir"] is None


# ──────────────────────────────────────────────
# ConfigDrivenStrategy
# ──────────────────────────────────────────────


class TestConfigDrivenStrategy:
    def test_custom_with_skills(self, submission_dir: Path) -> None:
        config = ExperimentConfig(
            type="custom",
            treatment=VariantSpec(
                copy=[CopySpec(src="skills", dest="/skills")],
            ),
            control=VariantSpec(),
        )
        strategy = ConfigDrivenStrategy(config)
        treatment_specs = strategy.variant_copy_specs(submission_dir, "treatment")
        assert len(treatment_specs) == 1
        assert treatment_specs[0].src == "skills"

        control_specs = strategy.variant_copy_specs(submission_dir, "control")
        assert control_specs == []

    def test_custom_skills_dir_set(self, submission_dir: Path) -> None:
        config = ExperimentConfig(
            type="custom",
            treatment=VariantSpec(
                copy=[CopySpec(src="skills", dest="/skills")],
            ),
        )
        strategy = ConfigDrivenStrategy(config)
        ctx = strategy.customize_context({}, "treatment", submission_dir)
        assert ctx["skills_dir"] == "/skills"

    def test_custom_no_skills_dir_when_absent(self, submission_dir: Path) -> None:
        config = ExperimentConfig(
            type="custom",
            treatment=VariantSpec(
                copy=[CopySpec(src="docs", dest="/workspace/docs")],
            ),
        )
        strategy = ConfigDrivenStrategy(config)
        ctx = strategy.customize_context({}, "treatment", submission_dir)
        assert ctx["skills_dir"] is None

    def test_custom_matches_skill_strategy_output(self, submission_dir: Path) -> None:
        """ConfigDrivenStrategy with skill-like config produces same output."""
        skill_config = ExperimentConfig()
        custom_config = ExperimentConfig(
            type="custom",
            treatment=VariantSpec(
                copy=[
                    CopySpec(src="skills", dest="/skills"),
                    CopySpec(src="docs", dest="/workspace/docs"),
                ],
            ),
            control=VariantSpec(),
        )

        skill_strategy = SkillExperimentStrategy(skill_config)
        custom_strategy = ConfigDrivenStrategy(custom_config)

        skill_ctx = skill_strategy.customize_context({}, "treatment", submission_dir)
        custom_ctx = custom_strategy.customize_context({}, "treatment", submission_dir)

        assert skill_ctx["skills_dir"] == custom_ctx["skills_dir"]
        assert skill_ctx["copy_pairs"] == custom_ctx["copy_pairs"]

    def test_custom_env_from_secrets(self, submission_dir: Path) -> None:
        config = ExperimentConfig(
            type="custom",
            treatment=VariantSpec(
                env_from_secrets={"KEY": "secret/key"},
            ),
        )
        strategy = ConfigDrivenStrategy(config)
        ctx = strategy.customize_context({}, "treatment", submission_dir)
        assert ctx["env_from_secrets"]["KEY"] == "secret/key"

    def test_custom_skips_missing_dirs(self, submission_dir: Path) -> None:
        config = ExperimentConfig(
            type="custom",
            treatment=VariantSpec(
                copy=[CopySpec(src="nonexistent", dest="/nonexistent")],
            ),
        )
        strategy = ConfigDrivenStrategy(config)
        specs = strategy.variant_copy_specs(submission_dir, "treatment")
        assert specs == []

    def test_custom_missing_dirs_not_in_context(self, submission_dir: Path) -> None:
        """copy_pairs must not reference dirs absent from submission."""
        config = ExperimentConfig(
            type="custom",
            treatment=VariantSpec(
                copy=[
                    CopySpec(src="skills", dest="/skills"),
                    CopySpec(src="nonexistent", dest="/nonexistent"),
                ],
            ),
        )
        strategy = ConfigDrivenStrategy(config)
        ctx = strategy.customize_context({}, "treatment", submission_dir)
        srcs = [src for src, _ in ctx["copy_pairs"]]
        assert "skills" in srcs
        assert "nonexistent" not in srcs
