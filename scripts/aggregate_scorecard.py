#!/usr/bin/env python3
"""Aggregate all gate results into a unified scorecard.

Collects results from the evaluation engine, security gates, and quality gates,
then applies the configured policy to produce a single recommendation.

Note:
    Currently, MCPChecker submissions do not produce a scorecard because the
    analyze task (which runs this script) is skipped for MCPChecker. This is
    a known limitation. To fix it, the scorecard step would need to be
    extracted into a standalone task that runs after evaluate for all engines.

Usage::

    python scripts/aggregate_scorecard.py \\
        --submission-dir /workspace/submissions/my-submission \\
        --results-dir /workspace/eval-results/my-submission \\
        --reports-dir /workspace/reports/my-submission \\
        --workspace-root /workspace \\
        --eval-engine harbor \\
        --pipeline-run-id abc123
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from abevalflow.certification import compute_certification
from abevalflow.compass_facts import (
    CertificationFactPushResult,
    FactPushResult,
    UnresolvedEnvVarError,
    push_certification_facts,
    push_gate_fact_from_config,
    validate_push_facts_config,
)
from abevalflow.engines import get_engine
from abevalflow.gates.base import GateResult
from abevalflow.gates.quality import get_all_quality_gates
from abevalflow.gates.security import get_all_security_gates
from abevalflow.schemas import CertificationPolicy, GatePolicy, SubmissionMetadata
from abevalflow.scorecard import Recommendation, Scorecard, apply_combination_logic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_gate_policy(submission_dir: Path) -> GatePolicy:
    """Load gate policy from metadata.yaml or return defaults."""
    metadata_path = submission_dir / "metadata.yaml"
    if not metadata_path.exists():
        logger.info("No metadata.yaml found, using default policy")
        return GatePolicy()

    try:
        data = yaml.safe_load(metadata_path.read_text())
        if data is None:
            return GatePolicy()

        metadata = SubmissionMetadata(**data)
        if metadata.gate_policy:
            logger.info("Loaded gate policy from metadata.yaml")
            return metadata.gate_policy
        else:
            logger.info("No gate_policy in metadata.yaml, using defaults")
            return GatePolicy()

    except Exception as e:
        logger.warning("Failed to parse metadata.yaml, using defaults: %s", e)
        return GatePolicy()


def load_certification_policy(submission_dir: Path) -> CertificationPolicy | None:
    """Load certification policy from metadata.yaml or return None for defaults."""
    metadata_path = submission_dir / "metadata.yaml"
    if not metadata_path.exists():
        logger.info("No metadata.yaml found, using default certification policy")
        return None

    try:
        data = yaml.safe_load(metadata_path.read_text())
        if data is None:
            return None

        metadata = SubmissionMetadata(**data)
        if metadata.certification_policy:
            logger.info("Loaded certification policy from metadata.yaml")
            return metadata.certification_policy
        else:
            logger.info("No certification_policy in metadata.yaml, using defaults")
            return None

    except Exception as e:
        logger.warning("Failed to parse metadata.yaml for certification policy: %s", e)
        return None


def load_provenance(reports_dir: Path) -> dict:
    """Load provenance from existing report.json if available."""
    report_path = reports_dir / "report.json"
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text())
            return data.get("provenance", {})
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _maybe_push_fact(
    gate_result: GateResult, policy: GatePolicy
) -> FactPushResult | None:
    """Push gate fact to Compass if configured.

    Args:
        gate_result: The gate result to potentially push
        policy: Policy containing push_facts configuration

    Returns:
        FactPushResult if push was attempted, None if skipped
    """
    # Policy is keyed by category (gate_name), not implementation (policy_key)
    if not policy.should_push_fact(gate_result.gate_name):
        return None

    if policy.push_facts is None:
        return None

    # Compute intended fact_ref before the call (for use in error case)
    policy_key = gate_result.get_policy_key()
    if policy_key and policy_key != gate_result.gate_name:
        intended_fact_ref = f"{policy.push_facts.fact_ref_prefix}{gate_result.gate_name}_{policy_key}"
    else:
        intended_fact_ref = f"{policy.push_facts.fact_ref_prefix}{gate_result.gate_name}"

    try:
        result = push_gate_fact_from_config(gate_result, policy.push_facts)
        if result.success:
            logger.info("Pushed fact for gate %s to Compass", gate_result.gate_name)
        else:
            logger.warning(
                "Failed to push fact for gate %s: %s",
                gate_result.gate_name,
                result.error,
            )
        return result
    except UnresolvedEnvVarError as e:
        logger.error(
            "Cannot push fact for gate %s: unresolved env var - %s",
            gate_result.gate_name,
            e,
        )
        return FactPushResult(
            gate_name=gate_result.gate_name,
            fact_ref=intended_fact_ref,
            success=False,
            error=str(e),
        )


def aggregate_scorecard(
    submission_dir: Path,
    results_dir: Path,
    reports_dir: Path,
    workspace_root: Path,
    eval_engine: str,
    pipeline_run_id: str,
) -> Scorecard:
    """Aggregate all gates into a unified scorecard.

    Args:
        submission_dir: Path to submissions/{submission-name}/
        results_dir: Path to eval-results/{submission-name}/
        reports_dir: Path to reports/{submission-name}/
        workspace_root: Path to workspace root (for _ai_review.json)
        eval_engine: Primary evaluation engine name (or "both" for Harbor+ASE)
        pipeline_run_id: Tekton PipelineRun ID

    Returns:
        Populated Scorecard instance
    """
    policy = load_gate_policy(submission_dir)
    certification_policy = load_certification_policy(submission_dir)
    provenance = load_provenance(reports_dir)

    # Validate push_facts configuration
    validate_push_facts_config(policy.push_facts, policy.get_gates_with_push_fact())

    submission_name = submission_dir.name
    gates: list[GateResult] = []
    fact_push_results: list[FactPushResult] = []

    # Determine which engines to process and their report directories
    # In 'both' mode, Harbor reports are in reports_dir/harbor/, ASE in reports_dir/
    if eval_engine == "both":
        engine_configs = [
            ("harbor", reports_dir / "harbor"),
            ("ase", reports_dir),
        ]
        logger.info("'both' engine mode: processing Harbor and ASE")
    else:
        engine_configs = [(eval_engine, reports_dir)]

    # Process all requested engines
    for engine_name, engine_reports_dir in engine_configs:
        engine = get_engine(engine_name)
        logger.info("Processing engine gate: %s (reports: %s)", engine.name, engine_reports_dir)

        raw_result = engine.read_result(engine_reports_dir)
        if raw_result:
            engine_gate = engine.to_gate_result(raw_result, policy)
            gates.append(engine_gate)
            push_result = _maybe_push_fact(engine_gate, policy)
            if push_result:
                fact_push_results.append(push_result)
            logger.info(
                "Engine %s: passed=%s, score=%.3f",
                engine.name, engine_gate.passed, engine_gate.score
            )
        else:
            logger.warning("No result found for engine %s at %s", engine.name, engine_reports_dir)

    for security_gate in get_all_security_gates():
        if not policy.is_enabled(security_gate.name):
            logger.info("Security gate %s is disabled, skipping", security_gate.name)
            continue

        logger.info("Processing security gate: %s", security_gate.name)
        gate_result = security_gate.evaluate(reports_dir, policy)
        gates.append(gate_result)
        push_result = _maybe_push_fact(gate_result, policy)
        if push_result:
            fact_push_results.append(push_result)
        logger.info(
            "Security %s: passed=%s, score=%.3f, findings=%d",
            security_gate.name, gate_result.passed, gate_result.score, len(gate_result.findings)
        )

    for quality_gate in get_all_quality_gates():
        if not policy.is_enabled(quality_gate.name):
            logger.info("Quality gate %s is disabled, skipping", quality_gate.name)
            continue

        logger.info("Processing quality gate: %s", quality_gate.name)
        gate_result = quality_gate.evaluate(workspace_root, policy)
        gates.append(gate_result)
        push_result = _maybe_push_fact(gate_result, policy)
        if push_result:
            fact_push_results.append(push_result)
        logger.info(
            "Quality %s: passed=%s, score=%.3f",
            quality_gate.name, gate_result.passed, gate_result.score
        )

    recommendation, reason = apply_combination_logic(gates, policy)
    logger.info("Final recommendation: %s (%s)", recommendation, reason)

    # Read validation results from validate task output
    validation_path = reports_dir / "validation.json"
    validation_passed = False
    metadata_valid = False
    validation_errors: list[str] = []

    if validation_path.exists():
        try:
            validation_data = json.loads(validation_path.read_text())
            validation_passed = validation_data.get("valid", False)
            validation_errors = validation_data.get("errors", [])
            # Metadata is valid if validation passed (schema check is part of validation)
            metadata_valid = validation_passed
            logger.info(
                "Read validation results: valid=%s, errors=%d",
                validation_passed,
                len(validation_errors),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read validation.json: %s", e)
            validation_passed = False
            metadata_valid = False
    else:
        logger.warning(
            "validation.json not found at %s; validation_passed defaults to False",
            validation_path,
        )
        validation_passed = False
        metadata_valid = False

    has_eval_assets = (
        (submission_dir / "evals" / "evals.json").exists()
        or (submission_dir / "tests").exists()
    )

    certification = compute_certification(
        gates=gates,
        validation_passed=validation_passed,
        metadata_valid=metadata_valid,
        has_eval_assets=has_eval_assets,
        policy=certification_policy,
    )
    logger.info(
        "Certification levels: foundational=%s, trusted=%s, certified=%s (highest=%s)",
        certification.foundational.passed,
        certification.trusted.passed,
        certification.certified.passed,
        certification.highest_level.value,
    )

    cert_push_results: list[CertificationFactPushResult] = []
    if policy.push_facts is not None:
        try:
            cert_push_results = push_certification_facts(certification, policy.push_facts)
            success_count = sum(1 for r in cert_push_results if r.success)
            logger.info(
                "Certification facts: %d/%d pushed successfully",
                success_count,
                len(cert_push_results),
            )
        except UnresolvedEnvVarError as e:
            logger.error("Cannot push certification facts: unresolved env var - %s", e)

    if fact_push_results:
        success_count = sum(1 for r in fact_push_results if r.success)
        logger.info(
            "Gate facts: %d/%d pushed successfully",
            success_count,
            len(fact_push_results),
        )

    return Scorecard(
        submission_name=submission_name,
        pipeline_run_id=pipeline_run_id,
        eval_engine=eval_engine,
        gates=gates,
        policy=policy,
        recommendation=recommendation,
        recommendation_reason=reason,
        created_at=datetime.now(timezone.utc),
        provenance=provenance,
        fact_push_results=fact_push_results,
        certification=certification,
        certification_fact_push_results=cert_push_results,
    )


def write_scorecard(scorecard: Scorecard, reports_dir: Path) -> Path:
    """Write scorecard to JSON file."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "scorecard.json"

    with open(output_path, "w") as f:
        f.write(scorecard.model_dump_json(indent=2))

    logger.info("Wrote scorecard to %s", output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate gate results into unified scorecard"
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        required=True,
        help="Path to submissions/{submission-name}/",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to eval-results/{submission-name}/",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Path to reports/{submission-name}/",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        required=True,
        help="Path to workspace root (for _ai_review.json)",
    )
    parser.add_argument(
        "--eval-engine",
        type=str,
        required=True,
        choices=["harbor", "ase", "a2a", "mcpchecker", "both"],
        help="Primary evaluation engine",
    )
    parser.add_argument(
        "--pipeline-run-id",
        type=str,
        default="unknown",
        help="Tekton PipelineRun ID",
    )
    parser.add_argument(
        "--output-tekton-results",
        type=Path,
        default=None,
        help="Directory to write Tekton result files",
    )

    args = parser.parse_args()

    try:
        scorecard = aggregate_scorecard(
            submission_dir=args.submission_dir,
            results_dir=args.results_dir,
            reports_dir=args.reports_dir,
            workspace_root=args.workspace_root,
            eval_engine=args.eval_engine,
            pipeline_run_id=args.pipeline_run_id,
        )

        write_scorecard(scorecard, args.reports_dir)

        if args.output_tekton_results:
            results_dir = args.output_tekton_results
            results_dir.mkdir(parents=True, exist_ok=True)

            (results_dir / "scorecard-recommendation").write_text(scorecard.recommendation.value)
            (results_dir / "scorecard-gates-passed").write_text(str(scorecard.gates_passed))
            (results_dir / "scorecard-gates-failed").write_text(str(scorecard.gates_failed))
            (results_dir / "scorecard-blocking-passed").write_text(str(scorecard.blocking_gates_passed))
            (results_dir / "scorecard-blocking-failed").write_text(str(scorecard.blocking_gates_failed))
            (results_dir / "highest-certification").write_text(scorecard.highest_certification.value)

            logger.info("Wrote Tekton results to %s", results_dir)

        print(f"Scorecard recommendation: {scorecard.recommendation.value}")
        print(f"Reason: {scorecard.recommendation_reason}")
        print(f"Gates: {scorecard.gates_passed} passed, {scorecard.gates_failed} failed")
        print(f"Highest certification: {scorecard.highest_certification.value}")

        return 0

    except Exception as e:
        logger.exception("Failed to aggregate scorecard: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
