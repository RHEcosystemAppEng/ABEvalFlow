"""Aggregate agent-skills-eval benchmark results into AnalysisResult format.

Reads benchmark.json and grading.json files produced by agent-skills-eval
across N iterations, maps with_skill -> treatment and without_skill -> control,
and writes a report.json + report.md compatible with the Harbor analysis
pipeline (same AnalysisResult Pydantic model).

Usage::

    python scripts/aggregate_ase.py \
        --results-dir /workspace/ase-results/my-submission \
        --output-dir /workspace/reports/my-submission \
        --submission-name my-submission \
        --threshold 0.0 \
        --iterations 5
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


def _find_grading_files(results_dir: Path, mode: str) -> list[Path]:
    """Find all grading.json files for a given mode across iterations."""
    files: list[Path] = []
    for grading in sorted(results_dir.rglob("grading.json")):
        if mode in grading.parts:
            files.append(grading)
    return files


def _parse_grading(grading_path: Path) -> tuple[float, int, int]:
    """Parse a grading.json and return (pass_rate, n_passed, n_total)."""
    data = json.loads(grading_path.read_text())
    summary = data.get("summary", {})
    passed = summary.get("passed", 0)
    total = summary.get("total", 0)
    pass_rate = summary.get("pass_rate", 0.0)
    return pass_rate, passed, total


def _parse_benchmarks(results_dir: Path, n_iterations: int) -> list[dict]:
    """Find and parse all benchmark.json files across iterations."""
    benchmarks = []
    for i in range(1, n_iterations + 1):
        bench_path = results_dir / f"iteration-{i}" / "benchmark.json"
        if bench_path.is_file():
            benchmarks.append(json.loads(bench_path.read_text()))
    if not benchmarks:
        for bench_path in sorted(results_dir.rglob("benchmark.json")):
            benchmarks.append(json.loads(bench_path.read_text()))
    return benchmarks


def _collect_trials(
    results_dir: Path, n_iterations: int
) -> tuple[list[TrialResult], list[TrialResult]]:
    """Collect per-iteration trial results for treatment (with_skill) and control (without_skill)."""
    treatment_trials: list[TrialResult] = []
    control_trials: list[TrialResult] = []

    for i in range(1, n_iterations + 1):
        iter_dir = results_dir / f"iteration-{i}"
        if not iter_dir.is_dir():
            continue

        for grading_path in sorted(iter_dir.rglob("grading.json")):
            parts = grading_path.parts
            if "with_skill" in parts:
                variant = "treatment"
                trials_list = treatment_trials
            elif "without_skill" in parts:
                variant = "control"
                trials_list = control_trials
            else:
                continue

            pass_rate, n_passed, n_total = _parse_grading(grading_path)
            trial_name = f"iteration-{i}-{variant}"
            reward = pass_rate
            trials_list.append(TrialResult(trial_name=trial_name, reward=reward))

    return treatment_trials, control_trials


def compute_variant_summary(trials: list[TrialResult]) -> VariantSummary:
    """Compute aggregate stats from trial results."""
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


def compute_ttest(
    treatment_trials: list[TrialResult],
    control_trials: list[TrialResult],
) -> float | None:
    """Welch's t-test on continuous reward scores between variants."""
    t_rewards = [t.reward for t in treatment_trials if t.reward is not None]
    c_rewards = [t.reward for t in control_trials if t.reward is not None]
    if len(t_rewards) < 2 or len(c_rewards) < 2:
        return None
    _, p = sp_stats.ttest_ind(t_rewards, c_rewards, equal_var=False)
    if math.isnan(p):
        return None
    return float(p)


def compute_fisher(
    treatment_summary: VariantSummary,
    control_summary: VariantSummary,
) -> float | None:
    """Fisher's exact test on the 2x2 pass/fail contingency table."""
    t_pass = treatment_summary.n_passed
    t_fail = treatment_summary.n_failed
    c_pass = control_summary.n_passed
    c_fail = control_summary.n_failed
    if (t_pass + t_fail) == 0 or (c_pass + c_fail) == 0:
        return None
    table = [[t_pass, t_fail], [c_pass, c_fail]]
    _, p = sp_stats.fisher_exact(table)
    return float(p)


def build_ase_analysis(
    results_dir: Path,
    submission_name: str,
    n_iterations: int = 5,
    threshold: float = 0.0,
    provenance: Provenance | None = None,
) -> AnalysisResult:
    """Parse ASE results across iterations and build an AnalysisResult."""
    treatment_trials, control_trials = _collect_trials(results_dir, n_iterations)

    if not treatment_trials:
        logger.warning("No with_skill (treatment) results found")
    if not control_trials:
        logger.warning("No without_skill (control) results found")

    t_summary = compute_variant_summary(treatment_trials)
    c_summary = compute_variant_summary(control_trials)

    uplift = t_summary.pass_rate - c_summary.pass_rate
    mean_gap = None
    if t_summary.mean_reward is not None and c_summary.mean_reward is not None:
        mean_gap = t_summary.mean_reward - c_summary.mean_reward

    ttest_p = compute_ttest(treatment_trials, control_trials)
    fisher_p = compute_fisher(t_summary, c_summary)

    primary_gap = mean_gap if mean_gap is not None else uplift

    if t_summary.n_trials == 0 or c_summary.n_trials == 0:
        logger.warning("No trial data for one or both variants — defaulting to FAIL")
        recommendation = Recommendation.FAIL
    elif t_summary.n_passed == 0 and c_summary.n_passed == 0:
        logger.warning("Zero passes in both variants — defaulting to FAIL")
        recommendation = Recommendation.FAIL
    else:
        recommendation = (
            Recommendation.PASS if primary_gap >= threshold else Recommendation.FAIL
        )

    prov = provenance or Provenance()
    prov.eval_engine = "ase"

    return AnalysisResult(
        submission_name=submission_name,
        provenance=prov,
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
    """Render a human-readable Markdown report from ASE analysis results."""
    s = result.summary
    t = s.treatment
    c = s.control
    prov = result.provenance

    lines: list[str] = []
    lines.append(f"# A/B Evaluation Report (agent-skills-eval): {result.submission_name}\n")

    lines.append("## Summary\n")
    lines.append("| Metric | With Skill (Treatment) | Without Skill (Control) |")
    lines.append("|--------|------------------------|-------------------------|")
    lines.append(f"| Iterations | {t.n_trials} | {c.n_trials} |")
    lines.append(f"| Passed | {t.n_passed} | {c.n_passed} |")
    lines.append(f"| Failed | {t.n_failed} | {c.n_failed} |")
    lines.append(f"| Errors | {t.n_errors} | {c.n_errors} |")
    lines.append(f"| Pass Rate | {_fmt(t.pass_rate)} | {_fmt(c.pass_rate)} |")
    lines.append(f"| Mean Reward | {_fmt(t.mean_reward)} | {_fmt(c.mean_reward)} |")
    lines.append(f"| Median Reward | {_fmt(t.median_reward)} | {_fmt(c.median_reward)} |")
    lines.append(f"| Std Reward | {_fmt(t.std_reward)} | {_fmt(c.std_reward)} |")
    lines.append("")

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
    lines.append(f"- **Evaluation engine:** {prov.eval_engine}")
    lines.append("")

    lines.append("## Provenance\n")
    lines.append(f"- Generated at: {prov.generated_at.isoformat()}")
    lines.append(f"- Evaluation engine: {prov.eval_engine}")
    if prov.commit_sha:
        lines.append(f"- Commit SHA: `{prov.commit_sha}`")
    if prov.pipeline_run_id:
        lines.append(f"- Pipeline run: `{prov.pipeline_run_id}`")
    lines.append("")

    lines.append("## Iteration Details\n")
    for variant_key, variant_label in [("treatment", "With Skill"), ("control", "Without Skill")]:
        trials = result.trials.get(variant_key, [])
        lines.append(f"<details>\n<summary>{variant_label} ({len(trials)} iterations)</summary>\n")
        lines.append("| # | Iteration | Reward (Pass Rate) | Passed |")
        lines.append("|---|-----------|-------------------|--------|")
        for i, tr in enumerate(trials, 1):
            r_str = _fmt(tr.reward) if tr.reward is not None else "ERROR"
            p_str = "PASS" if tr.passed else "FAIL"
            lines.append(f"| {i} | {tr.trial_name} | {r_str} | {p_str} |")
        lines.append("\n</details>\n")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Aggregate agent-skills-eval results into AnalysisResult format",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Path to ASE results directory containing iteration-N/ subdirs",
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
        "--iterations", type=int, default=5,
        help="Number of iterations to aggregate (default: 5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum uplift for a 'pass' recommendation (default: 0.0)",
    )
    parser.add_argument("--commit-sha", default=None)
    parser.add_argument("--pipeline-run-id", default=None)

    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        logger.error("Results directory does not exist: %s", args.results_dir)
        return 1

    provenance = Provenance(
        commit_sha=args.commit_sha,
        pipeline_run_id=args.pipeline_run_id,
        eval_engine="ase",
    )

    result = build_ase_analysis(
        results_dir=args.results_dir,
        submission_name=args.submission_name,
        n_iterations=args.iterations,
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
    print(f"Treatment mean reward: {_fmt(s.treatment.mean_reward)}")
    print(f"Control mean reward:   {_fmt(s.control.mean_reward)}")
    print(f"Mean reward gap:       {_fmt(s.mean_reward_gap, '+.4f')}")
    print(f"Uplift (pass rate):    {s.uplift:+.4f}")
    print(f"Recommendation:        {s.recommendation.value.upper()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
