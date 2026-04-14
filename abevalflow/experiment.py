"""Experiment strategies for A/B evaluation variants.

Each strategy determines which directories to copy and how to customize
the template context for treatment and control variants. The scaffold
module delegates to these strategies rather than hardcoding skilled/unskilled
logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from abevalflow.schemas import CopySpec, ExperimentConfig, ExperimentType, VariantSpec


class ExperimentStrategy(Protocol):
    """Protocol for experiment variant strategies."""

    def variant_copy_specs(
        self, submission_dir: Path, variant: str,
    ) -> list[CopySpec]:
        """Return copy specs for this variant ('control' or 'treatment').

        Only specs whose src directory exists in submission_dir are returned.
        """
        ...

    def customize_context(
        self, base_context: dict, variant: str, submission_dir: Path,
    ) -> dict:
        """Adjust template context per variant.

        Must set 'skills_dir' when skills/ is in the copy spec, and
        'copy_pairs' as a list of (src, dest) tuples for Dockerfile.j2.
        copy_pairs are filtered to directories that actually exist in
        submission_dir to avoid COPY instructions for missing dirs.
        """
        ...


def _skills_dir_from_specs(specs: list[CopySpec]) -> str | None:
    """Return the dest path if 'skills' is in the copy spec src list."""
    for spec in specs:
        if spec.src == "skills":
            return spec.dest
    return None


def _get_variant_spec(config: ExperimentConfig, variant: str) -> VariantSpec:
    return config.treatment if variant == "treatment" else config.control


def _filter_specs(specs: list[CopySpec], submission_dir: Path) -> list[CopySpec]:
    """Return only specs whose src directory exists in the submission."""
    return [s for s in specs if (submission_dir / s.src).is_dir()]


class SkillExperimentStrategy:
    """Default strategy: treatment includes skills/docs, control excludes them.

    Common dirs (tests, supportive, scripts) are handled by the scaffold
    module and are not part of the strategy's copy specs.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def variant_copy_specs(
        self, submission_dir: Path, variant: str,
    ) -> list[CopySpec]:
        specs = _get_variant_spec(self._config, variant).copy_dirs
        return _filter_specs(specs, submission_dir)

    def customize_context(
        self, base_context: dict, variant: str, submission_dir: Path,
    ) -> dict:
        filtered = self.variant_copy_specs(submission_dir, variant)
        ctx = {**base_context}
        ctx["skills_dir"] = _skills_dir_from_specs(filtered)
        ctx["copy_pairs"] = [(s.src, s.dest) for s in filtered]
        return ctx


class _PerVariantStrategy:
    """Shared base for strategies that read copy/env from the per-variant spec.

    Both ModelExperimentStrategy and ConfigDrivenStrategy have identical
    mechanics — they differ only in semantic intent. This base eliminates
    the duplication so changes need not be mirrored.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def variant_copy_specs(
        self, submission_dir: Path, variant: str,
    ) -> list[CopySpec]:
        specs = _get_variant_spec(self._config, variant).copy_dirs
        return _filter_specs(specs, submission_dir)

    def customize_context(
        self, base_context: dict, variant: str, submission_dir: Path,
    ) -> dict:
        variant_spec = _get_variant_spec(self._config, variant)
        filtered = _filter_specs(variant_spec.copy_dirs, submission_dir)
        ctx = {**base_context}
        ctx["skills_dir"] = _skills_dir_from_specs(filtered)
        ctx["copy_pairs"] = [(s.src, s.dest) for s in filtered]
        ctx["env_from_secrets"] = variant_spec.env_from_secrets
        return ctx


class ModelExperimentStrategy(_PerVariantStrategy):
    """Same files for both variants, different env vars.

    Both variants get identical copy specs from their respective
    VariantSpec. The difference is in env_from_secrets — e.g.,
    treatment uses model A, control uses model B. Env vars are
    injected via Harbor's persistent_env at runtime, not baked
    into the Dockerfile.
    """


class ConfigDrivenStrategy(_PerVariantStrategy):
    """Reads copy/env directly from ExperimentConfig for 'custom' type.

    Sets skills_dir when 'skills' is in the copy spec src list.
    """


_STRATEGY_MAP: dict[ExperimentType, type] = {
    ExperimentType.SKILL: SkillExperimentStrategy,
    ExperimentType.MODEL: ModelExperimentStrategy,
    # Prompt experiments differ at runtime (different system prompt), not in
    # container layout — same copy/scaffold behavior as skill experiments.
    ExperimentType.PROMPT: SkillExperimentStrategy,
    ExperimentType.CUSTOM: ConfigDrivenStrategy,
}


def get_strategy(config: ExperimentConfig) -> ExperimentStrategy:
    """Return the appropriate strategy for the given experiment config."""
    cls = _STRATEGY_MAP[config.type]
    return cls(config)
