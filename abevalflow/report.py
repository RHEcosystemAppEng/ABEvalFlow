"""Pydantic models for A/B evaluation analysis reports.

These models define the structure of the JSON report produced by
``scripts/analyze.py``. They serve as the contract between the analysis
step and any downstream consumers (DB persistence, PR comments, dashboards).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class Recommendation(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class TrialResult(BaseModel):
    """A single trial's outcome."""

    trial_name: str
    reward: float | None = Field(
        default=None,
        description="Continuous reward score (0.0-1.0). None if the trial produced no parseable result.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return self.reward is not None and self.reward > 0.0


class VariantSummary(BaseModel):
    """Aggregate statistics for one variant's trials."""

    n_trials: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_errors: int = Field(
        default=0,
        description="Trials with missing or unparseable results",
    )
    pass_rate: float = 0.0
    mean_reward: float | None = None
    median_reward: float | None = None
    std_reward: float | None = None


class Provenance(BaseModel):
    """Run provenance metadata for reproducibility."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    commit_sha: str | None = None
    pipeline_run_id: str | None = None
    treatment_image_ref: str | None = None
    control_image_ref: str | None = None
    harbor_fork_revision: str | None = None


class AnalysisSummary(BaseModel):
    """Comparison statistics between treatment and control."""

    treatment: VariantSummary
    control: VariantSummary
    uplift: float = Field(description="treatment pass_rate - control pass_rate")
    mean_reward_gap: float | None = Field(
        default=None,
        description="treatment mean_reward - control mean_reward",
    )
    ttest_p_value: float | None = Field(
        default=None,
        description="Welch's t-test p-value on continuous reward scores",
    )
    fisher_p_value: float | None = Field(
        default=None,
        description="Fisher's exact test p-value on binary pass/fail counts",
    )
    recommendation: Recommendation


class AnalysisResult(BaseModel):
    """Top-level report model written to report.json."""

    submission_name: str
    provenance: Provenance
    summary: AnalysisSummary
    trials: dict[str, list[TrialResult]] = Field(
        description="Per-variant list of trial outcomes, keyed by 'treatment'/'control'",
    )
