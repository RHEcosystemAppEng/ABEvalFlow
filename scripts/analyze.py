"""Analyze Harbor A/B evaluation results and produce JSON + Markdown reports.

Walks the result directory tree produced by ``harbor-eval`` (two separate
Harbor jobs, one per variant), computes per-variant statistics, runs
statistical significance tests, and writes a structured report.

Expected input layout::

    <results-dir>/
        treatment/
            <job-name>/
                <task>__<uuid>/result.json
                ...
        control/
            <job-name>/
                <task>__<uuid>/result.json
                ...

Usage::

    python scripts/analyze.py \\
        --results-dir /workspace/eval-results/my-submission \\
        --output-dir /workspace/reports/my-submission \\
        --submission-name my-submission \\
        --threshold 0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from statistics import median, stdev

from scipy import stats as sp_stats

from abevalflow.report import (
    AnalysisResult,
    AnalysisSummary,
    Provenance,
    Recommendation,
    TrialResult,
    VariantSummary,
)

logger = logging.getLogger(__name__)

VARIANTS = ("treatment", "control")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _extract_reward(result: dict) -> float | None:
    """Extract reward from a Harbor result.json.

    Handles two known formats:
    - Nested: ``verifier_result.rewards.reward`` (actual Harbor output)
    - Flat:   ``verifier_result.reward`` (used in harbor-eval inline parser)
    """
    vr = result.get("verifier_result")
    if not isinstance(vr, dict):
        return None
    rewards = vr.get("rewards")
    if isinstance(rewards, dict):
        r = rewards.get("reward")
        if r is not None:
            return float(r)
    r = vr.get("reward")
    if r is not None:
        return float(r)
    return None


def parse_variant_trials(variant_dir: Path) -> list[TrialResult]:
    """Scan all trial directories under a variant's results directory.

    Skips job-level ``result.json`` files (which lack ``verifier_result``)
    so only actual trial results are counted.
    """
    trials: list[TrialResult] = []
    if not variant_dir.is_dir():
        return trials

    for result_file in sorted(variant_dir.rglob("result.json")):
        trial_name = result_file.parent.name
        try:
            data = json.loads(result_file.read_text())
            if "verifier_result" not in data:
                continue
            reward = _extract_reward(data)
            trials.append(TrialResult(trial_name=trial_name, reward=reward))
        except (json.JSONDecodeError, ValueError, TypeError):
            trials.append(TrialResult(trial_name=trial_name))

    return trials


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_variant_summary(trials: list[TrialResult]) -> VariantSummary:
    """Compute aggregate stats from a list of trial results."""
    rewards = [t.reward for t in trials if t.reward is not None]
    n_errors = sum(1 for t in trials if t.reward is None)
    n_passed = sum(1 for t in trials if t.passed)
    n_total = len(trials)
    n_failed = n_total - n_passed - n_errors

    pass_rate = n_passed / n_total if n_total > 0 else 0.0
    mean_r = sum(rewards) / len(rewards) if rewards else None
    median_r = median(rewards) if rewards else None
    std_r = stdev(rewards) if len(rewards) > 1 else None

    return VariantSummary(
        n_trials=n_total,
        n_passed=n_passed,
        n_failed=n_failed,
        n_errors=n_errors,
        pass_rate=pass_rate,
        mean_reward=mean_r,
        median_reward=median_r,
        std_reward=std_r,
    )


def compute_ttest(treatment_trials: list[TrialResult],
                  control_trials: list[TrialResult]) -> float | None:
    """Welch's t-test on continuous reward scores between variants."""
    t_rewards = [t.reward for t in treatment_trials if t.reward is not None]
    c_rewards = [t.reward for t in control_trials if t.reward is not None]
    if len(t_rewards) < 2 or len(c_rewards) < 2:
        return None
    _, p = sp_stats.ttest_ind(t_rewards, c_rewards, equal_var=False)
    if math.isnan(p):
        return None
    return float(p)


def compute_fisher(treatment_summary: VariantSummary,
                   control_summary: VariantSummary) -> float | None:
    """Fisher's exact test on the 2x2 pass/fail contingency table.

    Error trials (missing/corrupt results) are excluded from the table so
    infrastructure failures don't skew significance.
    """
    t_pass = treatment_summary.n_passed
    t_fail = treatment_summary.n_failed
    c_pass = control_summary.n_passed
    c_fail = control_summary.n_failed
    if (t_pass + t_fail) == 0 or (c_pass + c_fail) == 0:
        return None
    table = [[t_pass, t_fail], [c_pass, c_fail]]
    _, p = sp_stats.fisher_exact(table)
    return float(p)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_analysis(
    results_dir: Path,
    submission_name: str,
    threshold: float = 0.0,
    provenance: Provenance | None = None,
) -> AnalysisResult:
    """Parse results, compute stats, and assemble the full analysis model."""
    treatment_trials = parse_variant_trials(results_dir / "treatment")
    control_trials = parse_variant_trials(results_dir / "control")

    t_summary = compute_variant_summary(treatment_trials)
    c_summary = compute_variant_summary(control_trials)

    _MIN_TRIALS_FOR_RELIABLE_STATS = 15
    for label, vs in [("treatment", t_summary), ("control", c_summary)]:
        if 0 < vs.n_trials < _MIN_TRIALS_FOR_RELIABLE_STATS:
            logger.warning(
                "%s has only %d trials (< %d) — statistical tests may be unreliable",
                label, vs.n_trials, _MIN_TRIALS_FOR_RELIABLE_STATS,
            )

    uplift = t_summary.pass_rate - c_summary.pass_rate
    mean_gap = None
    if t_summary.mean_reward is not None and c_summary.mean_reward is not None:
        mean_gap = t_summary.mean_reward - c_summary.mean_reward

    ttest_p = compute_ttest(treatment_trials, control_trials)
    fisher_p = compute_fisher(t_summary, c_summary)

    if t_summary.n_trials == 0 or c_summary.n_trials == 0:
        logger.warning("No trial data for one or both variants — defaulting to FAIL")
        recommendation = Recommendation.FAIL
    elif t_summary.n_passed == 0 and c_summary.n_passed == 0:
        logger.warning("Zero passes in both variants — defaulting to FAIL")
        recommendation = Recommendation.FAIL
    else:
        recommendation = Recommendation.PASS if uplift >= threshold else Recommendation.FAIL

    return AnalysisResult(
        submission_name=submission_name,
        provenance=provenance or Provenance(),
        summary=AnalysisSummary(
            treatment=t_summary,
            control=c_summary,
            uplift=uplift,
            mean_reward_gap=mean_gap,
            ttest_p_value=ttest_p,
            fisher_p_value=fisher_p,
            recommendation=recommendation,
        ),
        trials={
            "treatment": treatment_trials,
            "control": control_trials,
        },
    )


def _fmt(val: float | None, fmt: str = ".4f") -> str:
    return f"{val:{fmt}}" if val is not None else "N/A"


def _sig_marker(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def render_markdown(result: AnalysisResult) -> str:
    """Render a human-readable Markdown report from analysis results."""
    s = result.summary
    t = s.treatment
    c = s.control
    prov = result.provenance

    lines: list[str] = []
    lines.append(f"# A/B Evaluation Report: {result.submission_name}\n")

    # --- Summary table ---
    lines.append("## Summary\n")
    lines.append("| Metric | Treatment | Control |")
    lines.append("|--------|-----------|---------|")
    lines.append(f"| Trials | {t.n_trials} | {c.n_trials} |")
    lines.append(f"| Passed | {t.n_passed} | {c.n_passed} |")
    lines.append(f"| Failed | {t.n_failed} | {c.n_failed} |")
    lines.append(f"| Errors | {t.n_errors} | {c.n_errors} |")
    lines.append(f"| Pass Rate | {_fmt(t.pass_rate)} | {_fmt(c.pass_rate)} |")
    lines.append(f"| Mean Reward | {_fmt(t.mean_reward)} | {_fmt(c.mean_reward)} |")
    lines.append(f"| Median Reward | {_fmt(t.median_reward)} | {_fmt(c.median_reward)} |")
    lines.append(f"| Std Reward | {_fmt(t.std_reward)} | {_fmt(c.std_reward)} |")
    lines.append("")

    # --- Comparison ---
    lines.append("## Comparison\n")
    lines.append(f"- **Uplift (pass rate gap):** {s.uplift:+.4f}")
    if s.mean_reward_gap is not None:
        lines.append(f"- **Mean reward gap:** {s.mean_reward_gap:+.4f}")
    lines.append(
        f"- **Welch's t-test p-value:** {_fmt(s.ttest_p_value)}{_sig_marker(s.ttest_p_value)}"
    )
    lines.append(
        f"- **Fisher's exact p-value:** {_fmt(s.fisher_p_value)}{_sig_marker(s.fisher_p_value)}"
    )
    lines.append(f"- **Recommendation:** **{s.recommendation.value.upper()}**")
    lines.append("")

    # --- Provenance ---
    lines.append("## Provenance\n")
    lines.append(f"- Generated at: {prov.generated_at.isoformat()}")
    if prov.commit_sha:
        lines.append(f"- Commit SHA: `{prov.commit_sha}`")
    if prov.pipeline_run_id:
        lines.append(f"- Pipeline run: `{prov.pipeline_run_id}`")
    if prov.treatment_image_ref:
        lines.append(f"- Treatment image: `{prov.treatment_image_ref}`")
    if prov.control_image_ref:
        lines.append(f"- Control image: `{prov.control_image_ref}`")
    if prov.harbor_fork_revision:
        lines.append(f"- Harbor fork revision: `{prov.harbor_fork_revision}`")
    lines.append("")

    # --- Per-trial details ---
    lines.append("## Trial Details\n")
    for variant in VARIANTS:
        trials = result.trials.get(variant, [])
        lines.append(f"<details>\n<summary>{variant.capitalize()} ({len(trials)} trials)</summary>\n")
        lines.append("| # | Trial | Reward | Passed |")
        lines.append("|---|-------|--------|--------|")
        for i, tr in enumerate(trials, 1):
            r_str = _fmt(tr.reward) if tr.reward is not None else "ERROR"
            p_str = "PASS" if tr.passed else "FAIL"
            lines.append(f"| {i} | {tr.trial_name} | {r_str} | {p_str} |")
        lines.append("\n</details>\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Analyze A/B evaluation results and produce JSON + Markdown reports",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Path to the results directory containing treatment/ and control/ subdirs",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write report.json and report.md",
    )
    parser.add_argument(
        "--submission-name", required=True,
        help="Name of the submission being analyzed",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum uplift for a 'pass' recommendation (default: 0.0)",
    )
    parser.add_argument("--commit-sha", default=None)
    parser.add_argument("--pipeline-run-id", default=None)
    parser.add_argument("--treatment-image-ref", default=None)
    parser.add_argument("--control-image-ref", default=None)
    parser.add_argument("--harbor-fork-revision", default=None)

    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        logger.error("Results directory does not exist: %s", args.results_dir)
        return 1

    provenance = Provenance(
        commit_sha=args.commit_sha,
        pipeline_run_id=args.pipeline_run_id,
        treatment_image_ref=args.treatment_image_ref,
        control_image_ref=args.control_image_ref,
        harbor_fork_revision=args.harbor_fork_revision,
    )

    result = build_analysis(
        results_dir=args.results_dir,
        submission_name=args.submission_name,
        threshold=args.threshold,
        provenance=provenance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "report.json"
    json_path.write_text(result.model_dump_json(indent=2))
    logger.info("Wrote JSON report to %s", json_path)

    md_path = args.output_dir / "report.md"
    md_path.write_text(render_markdown(result))
    logger.info("Wrote Markdown report to %s", md_path)

    s = result.summary
    print(f"Treatment pass rate: {s.treatment.pass_rate:.4f}")
    print(f"Control pass rate:   {s.control.pass_rate:.4f}")
    print(f"Uplift:              {s.uplift:+.4f}")
    print(f"Recommendation:      {s.recommendation.value.upper()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
