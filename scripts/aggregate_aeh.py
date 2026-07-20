#!/usr/bin/env python3
"""Map AEH output to ABEvalFlow report format.

Reads agent-eval-harness output files (summary.yaml, run_result.json) and
produces a unified report.json compatible with ABEvalFlow's scorecard logic.

Supports both single-run and pairwise modes:
  - Single: One run directory
  - Pairwise: Treatment and control directories with pairwise comparison results

Usage:
    # Single mode (default)
    python scripts/aggregate_aeh.py <run_dir> [--output <path>]

    # Pairwise mode
    python scripts/aggregate_aeh.py <treatment_dir> --mode pairwise --control-dir <control_dir>

Where <run_dir> is the AEH output directory containing:
    - summary.yaml: Per-judge means, per-case results, run metadata, pairwise results
    - run_result.json: Execution metadata (duration, cost, tokens)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from abevalflow.aeh_scoring import (
    DEFAULT_AEH_THRESHOLD,
    numeric_judge_passes,
    pairwise_outcome,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_execution_metadata(run_dir: Path) -> dict[str, Any]:
    """Load execution metadata from run_result.json."""
    run_result_path = run_dir / "run_result.json"
    if not run_result_path.exists():
        return {}

    try:
        run_result = json.loads(run_result_path.read_text())
        return {
            "duration_s": run_result.get("duration_s"),
            "cost_usd": run_result.get("cost_usd"),
            "tokens": run_result.get("token_usage"),
            "harbor_job_dir": run_result.get("harbor_job_dir"),
            "num_turns": run_result.get("num_turns"),
            "n_infra_errors": run_result.get("n_infra_errors"),
            "n_trial_errors": run_result.get("n_trial_errors"),
        }
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load run_result.json: %s", e)
        return {}


def _extract_mean_reward(run_dir: Path) -> float:
    """Extract mean_reward from run_result.json or summary.yaml."""
    run_result_path = run_dir / "run_result.json"
    summary_path = run_dir / "summary.yaml"

    mean_reward = 0.0

    if summary_path.exists():
        try:
            summary = yaml.safe_load(summary_path.read_text())
            mean_reward = summary.get("mean_reward", 0.0)
        except yaml.YAMLError:
            pass

    if run_result_path.exists():
        try:
            run_result = json.loads(run_result_path.read_text())
            if "mean_reward" in run_result:
                mean_reward = run_result["mean_reward"]
        except json.JSONDecodeError:
            pass

    return mean_reward


def _case_reward(case_data: Any) -> float | None:
    """Derive a single reward from an AEH per_case entry.

    Prefers the mean of numeric judge values; falls back to 1.0/0.0 from
    boolean judge passes. Returns None when no parseable judge value exists.
    """
    if not isinstance(case_data, dict):
        return None

    if isinstance(case_data.get("reward"), (int, float)):
        return float(case_data["reward"])

    numeric: list[float] = []
    bools: list[bool] = []
    for key, result in case_data.items():
        if key == "reward":
            continue
        if isinstance(result, dict) and "value" in result:
            value = result.get("value")
        else:
            value = result
        if isinstance(value, bool):
            bools.append(value)
        elif isinstance(value, (int, float)):
            numeric.append(float(value))

    if numeric:
        return sum(numeric) / len(numeric)
    if bools:
        return 1.0 if any(bools) else 0.0
    return None


def _trials_from_per_case(per_case: Any) -> list[dict[str, Any]]:
    """Map AEH per_case dict → TrialResult-shaped list for report.json."""
    if not isinstance(per_case, dict):
        return []
    trials: list[dict[str, Any]] = []
    for case_id, case_data in per_case.items():
        trials.append({"trial_name": str(case_id), "reward": _case_reward(case_data)})
    return trials


def aggregate_single_run(
    run_dir: Path,
    *,
    submission_name: str | None = None,
    threshold: float = DEFAULT_AEH_THRESHOLD,
) -> dict[str, Any]:
    """Read one AEH harness run directory and produce report dict.

    AEH output layout (from harbor.run):
        <run_dir>/summary.yaml      # run_id, judges, per_case, run_metrics
        <run_dir>/run_result.json   # duration, cost, tokens, mean_reward
        <run_dir>/cases/...         # per-case artifacts

    Args:
        run_dir: Path to the AEH run output directory
        submission_name: Override for report submission_name (Tekton param)
        threshold: Pass/fail threshold for mean_reward (matches GatePolicy default)

    Returns:
        Dict in ABEvalFlow report format with full judge metadata
    """
    summary_path = run_dir / "summary.yaml"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.yaml not found in {run_dir}")

    summary = yaml.safe_load(summary_path.read_text())
    mean_reward = _extract_mean_reward(run_dir)

    # Preserve full judge structure (not just means)
    judges_full = summary.get("judges", {})
    per_case_full = summary.get("per_case", {})
    run_metrics = summary.get("run_metrics")

    # Calculate pass rate from per_case data
    total_cases = len(per_case_full)
    passed_cases = 0
    for case_id, case_data in per_case_full.items():
        if isinstance(case_data, dict):
            # Check if any judge reported a passing value
            case_passed = False
            for judge_name, judge_result in case_data.items():
                if isinstance(judge_result, dict):
                    value = judge_result.get("value")
                    if isinstance(value, bool) and value:
                        case_passed = True
                        break
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if numeric_judge_passes(value):
                            case_passed = True
                            break
            if case_passed:
                passed_cases += 1

    pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0
    recommendation = "pass" if (mean_reward is not None and mean_reward >= threshold) else "fail"
    resolved_name = submission_name or (run_dir.parent.name if run_dir.parent != run_dir else run_dir.name)
    mean_for_gap = mean_reward if mean_reward is not None else 0.0

    # AnalysisResult-compatible shape (analyze task + scorecard) plus AEH extras.
    return {
        "submission_name": resolved_name,
        "provenance": {
            "eval_engine": "aeh",
            "pipeline_run_id": summary.get("run_id", run_dir.name),
        },
        "summary": {
            "treatment": {
                "n_trials": total_cases,
                "n_passed": passed_cases,
                "n_failed": max(total_cases - passed_cases, 0),
                "pass_rate": pass_rate,
                "mean_reward": mean_reward,
            },
            "control": {
                "n_trials": 0,
                "n_passed": 0,
                "n_failed": 0,
                "pass_rate": 0.0,
                "mean_reward": 0.0,
            },
            "uplift": pass_rate,
            "mean_reward_gap": mean_for_gap,
            "recommendation": recommendation,
        },
        "trials": {
            "treatment": _trials_from_per_case(per_case_full),
            "control": [],
        },
        "eval_engine": "aeh",
        "mode": "single",
        "run_id": summary.get("run_id", run_dir.name),
        "mean_reward": mean_reward,
        "pass_rate": pass_rate,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "judges": judges_full,
        "per_case": per_case_full,
        "run_metrics": run_metrics,
        "execution": _load_execution_metadata(run_dir),
        "aeh_warnings": [],
        "recommendation": recommendation,
    }


def aggregate_pairwise_run(
    treatment_dir: Path,
    control_dir: Path,
    *,
    submission_name: str | None = None,
    threshold: float = DEFAULT_AEH_THRESHOLD,
) -> dict[str, Any]:
    """Read pairwise AEH run directories and produce report dict.

    Pairwise mode expects:
        - Two run directories (treatment and control)
        - Treatment summary.yaml contains `pairwise` section from score.py pairwise

    Args:
        treatment_dir: Path to the treatment (A) run directory
        control_dir: Path to the control (B) run directory
        submission_name: Override for report submission_name (Tekton param)
        threshold: Pass/fail win-rate threshold (matches GatePolicy / engine)

    Returns:
        Dict in ABEvalFlow report format with pairwise results
    """
    treatment_summary_path = treatment_dir / "summary.yaml"
    control_summary_path = control_dir / "summary.yaml"

    if not treatment_summary_path.exists():
        raise FileNotFoundError(f"summary.yaml not found in {treatment_dir}")

    treatment_summary = yaml.safe_load(treatment_summary_path.read_text())
    treatment_mean_reward = _extract_mean_reward(treatment_dir)

    control_summary = {}
    control_mean_reward = 0.0
    if control_summary_path.exists():
        control_summary = yaml.safe_load(control_summary_path.read_text())
        control_mean_reward = _extract_mean_reward(control_dir)

    # Extract pairwise results from treatment summary
    pairwise = treatment_summary.get("pairwise", {})
    outcome = pairwise_outcome(
        pairwise.get("wins_a", 0),
        pairwise.get("wins_b", 0),
        pairwise.get("ties", 0),
        pairwise.get("errors", 0),
        threshold=threshold,
    )
    wins_a = outcome["wins_a"]
    wins_b = outcome["wins_b"]
    ties = outcome["ties"]
    errors = outcome["errors"]
    total = outcome["total"]
    win_rate = outcome["win_rate"]
    recommendation = outcome["recommendation"]
    cases_compared = pairwise.get("cases_compared", total)

    # Preserve full judge and per_case structures
    treatment_judges = treatment_summary.get("judges", {})
    treatment_per_case = treatment_summary.get("per_case", {})
    control_judges = control_summary.get("judges", {})
    control_per_case = control_summary.get("per_case", {})

    resolved_name = submission_name or (
        treatment_dir.parent.name if treatment_dir.parent != treatment_dir else treatment_dir.name
    )
    t_mean = treatment_mean_reward if treatment_mean_reward is not None else 0.0
    c_mean = control_mean_reward if control_mean_reward is not None else 0.0
    treatment_cases = len(treatment_per_case) if isinstance(treatment_per_case, dict) else 0
    control_cases = len(control_per_case) if isinstance(control_per_case, dict) else 0

    return {
        "submission_name": resolved_name,
        "provenance": {
            "eval_engine": "aeh",
            "pipeline_run_id": treatment_summary.get("run_id", treatment_dir.name),
        },
        "summary": {
            "treatment": {
                "n_trials": treatment_cases,
                "n_passed": wins_a,
                "n_failed": max(treatment_cases - wins_a, 0),
                "pass_rate": win_rate,
                "mean_reward": treatment_mean_reward,
            },
            "control": {
                "n_trials": control_cases,
                "n_passed": wins_b,
                "n_failed": max(control_cases - wins_b, 0),
                "pass_rate": (wins_b / total) if total > 0 else 0.0,
                "mean_reward": control_mean_reward,
            },
            "uplift": win_rate - ((wins_b / total) if total > 0 else 0.0),
            "mean_reward_gap": t_mean - c_mean,
            "recommendation": recommendation,
        },
        "trials": {
            "treatment": _trials_from_per_case(treatment_per_case),
            "control": _trials_from_per_case(control_per_case),
        },
        "eval_engine": "aeh",
        "mode": "pairwise",
        "treatment": {
            "run_id": treatment_summary.get("run_id", treatment_dir.name),
            "mean_reward": treatment_mean_reward,
            "judges": treatment_judges,
            "per_case": treatment_per_case,
            "run_metrics": treatment_summary.get("run_metrics"),
            "execution": _load_execution_metadata(treatment_dir),
        },
        "control": {
            "run_id": control_summary.get("run_id", control_dir.name),
            "mean_reward": control_mean_reward,
            "judges": control_judges,
            "per_case": control_per_case,
            "run_metrics": control_summary.get("run_metrics"),
            "execution": _load_execution_metadata(control_dir),
        },
        "pairwise": {
            "run_a": pairwise.get("run_a", treatment_dir.name),
            "run_b": pairwise.get("run_b", control_dir.name),
            "cases_compared": cases_compared,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "errors": errors,
            "win_rate": win_rate,
            "per_case": pairwise.get("per_case", []),
            "stability": pairwise.get("stability"),
        },
        "mean_reward": treatment_mean_reward,
        "aeh_warnings": (["pairwise section missing from treatment summary.yaml"] if not pairwise else []),
        "recommendation": recommendation,
    }


def aggregate_aeh_results(
    run_dir: Path,
    mode: str = "single",
    control_dir: Path | None = None,
    *,
    submission_name: str | None = None,
    threshold: float = DEFAULT_AEH_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate AEH results into ABEvalFlow report format.

    Args:
        run_dir: Path to the AEH run output directory (treatment in pairwise mode)
        mode: Either "single" or "pairwise"
        control_dir: Path to control directory (required for pairwise mode)
        submission_name: Override for report submission_name
        threshold: Pass/fail threshold aligned with AEHEngine / GatePolicy default

    Returns:
        Dict in ABEvalFlow report format
    """
    if mode == "pairwise":
        if control_dir is None:
            raise ValueError("control_dir is required for pairwise mode")
        return aggregate_pairwise_run(
            run_dir,
            control_dir,
            submission_name=submission_name,
            threshold=threshold,
        )
    return aggregate_single_run(
        run_dir,
        submission_name=submission_name,
        threshold=threshold,
    )


def find_latest_run_dir(reports_dir: Path, submission_name: str) -> Path | None:
    """Find the most recent run directory for a submission.

    Layout: reports/<submission_name>/<run_id>/summary.yaml

    Args:
        reports_dir: Root reports directory
        submission_name: Name of the submission

    Returns:
        Path to the latest run dir, or None if not found
    """
    submission_dir = reports_dir / submission_name
    if not submission_dir.exists():
        return None

    run_dirs = [d for d in submission_dir.iterdir() if d.is_dir() and (d / "summary.yaml").exists()]
    if not run_dirs:
        return None

    return sorted(run_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate AEH results into ABEvalFlow report format")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to AEH run directory containing summary.yaml (treatment dir in pairwise mode)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for report.json (default: <run_dir>/report.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "pairwise"],
        default="single",
        help="Aggregation mode (single or pairwise)",
    )
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=None,
        help="Path to control run directory (required for pairwise mode)",
    )
    parser.add_argument(
        "--submission-name",
        type=str,
        default=None,
        help="Override submission_name in report.json (Tekton submission-name)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_AEH_THRESHOLD,
        help=f"Pass/fail threshold (default: {DEFAULT_AEH_THRESHOLD})",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        logger.error("Not a directory: %s", run_dir)
        return 1

    if args.mode == "pairwise" and args.control_dir is None:
        logger.error("--control-dir is required for pairwise mode")
        return 1

    if args.control_dir and not args.control_dir.is_dir():
        logger.error("Control directory not found: %s", args.control_dir)
        return 1

    try:
        report = aggregate_aeh_results(
            run_dir,
            mode=args.mode,
            control_dir=args.control_dir,
            submission_name=args.submission_name,
            threshold=args.threshold,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    output_path = args.output or (run_dir / "report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote report to: %s", output_path)

    if args.mode == "pairwise":
        pairwise = report.get("pairwise", {})
        logger.info(
            "Pairwise: treatment wins %d/%d (%.0f%%), ties=%d, errors=%d",
            pairwise.get("wins_a", 0),
            pairwise.get("cases_compared", 0),
            pairwise.get("win_rate", 0) * 100,
            pairwise.get("ties", 0),
            pairwise.get("errors", 0),
        )
    else:
        mean_reward = report["mean_reward"]
        mean_reward_str = f"{mean_reward:.3f}" if mean_reward is not None else "None"
        logger.info(
            "Summary: mean_reward=%s, pass_rate=%.2f (%d/%d cases)",
            mean_reward_str,
            report.get("pass_rate", 0),
            report.get("passed_cases", 0),
            report.get("total_cases", 0),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
