"""MLflow observer for logging evaluation results as MLflow experiment runs.

One experiment per submission, one run per pipeline execution. Logs metrics,
parameters, tags, and artifacts. Auto-activates when MLFLOW_TRACKING_URI is set.

Observer failures are isolated and never fail the pipeline.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import mlflow

from abevalflow.report import AnalysisResult

logger = logging.getLogger(__name__)


class MLflowObserver:
    """Logs evaluation results to MLflow as experiment runs."""

    def __init__(self, tracking_uri: str, experiment_prefix: str = "abevalflow") -> None:
        self.tracking_uri = tracking_uri
        self.experiment_prefix = experiment_prefix

    def on_evaluation_stored(
        self,
        result: AnalysisResult,
        run_id: uuid.UUID,
        report_dir: Path | None = None,
        pipeline_run_id: str | None = None,
    ) -> None:

        mlflow.set_tracking_uri(self.tracking_uri)

        experiment_name = f"{self.experiment_prefix}/{result.submission_name}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        s = result.summary
        p = result.provenance
        effective_run_id = pipeline_run_id or p.pipeline_run_id or str(run_id)
        effective_commit = p.commit_sha or ""

        with mlflow.start_run(experiment_id=experiment_id, run_name=effective_run_id):
            mlflow.log_params(
                {
                    "eval_engine": p.eval_engine,
                    "commit_sha": effective_commit,
                    "pipeline_run_id": effective_run_id,
                }
            )

            mlflow.log_metrics(
                {
                    "uplift": s.uplift,
                    "treatment_pass_rate": s.treatment.pass_rate,
                    "control_pass_rate": s.control.pass_rate,
                    "treatment_n_trials": s.treatment.n_trials,
                    "control_n_trials": s.control.n_trials,
                }
            )

            if s.mean_reward_gap is not None:
                mlflow.log_metric("mean_reward_gap", s.mean_reward_gap)
            if s.ttest_p_value is not None:
                mlflow.log_metric("ttest_p_value", s.ttest_p_value)
            if s.fisher_p_value is not None:
                mlflow.log_metric("fisher_p_value", s.fisher_p_value)
            if s.treatment.mean_reward is not None:
                mlflow.log_metric("treatment_mean_reward", s.treatment.mean_reward)
            if s.control.mean_reward is not None:
                mlflow.log_metric("control_mean_reward", s.control.mean_reward)

            mlflow.set_tags(
                {
                    "recommendation": s.recommendation.value,
                    "eval_engine": p.eval_engine,
                    "submission_name": result.submission_name,
                    "db_run_id": str(run_id),
                }
            )

            if report_dir and report_dir.is_dir():
                for artifact_name in ("scorecard.json", "report.md", "report.json"):
                    artifact_path = report_dir / artifact_name
                    if artifact_path.is_file():
                        mlflow.log_artifact(str(artifact_path))

                _log_scorecard_metrics(report_dir)
                _log_observability_metrics(report_dir)

        logger.info(
            "MLflow: logged run for %s (experiment=%s)",
            result.submission_name,
            experiment_name,
        )


def _log_scorecard_metrics(report_dir: Path) -> None:
    """Log gate scores and certification from scorecard.json if present."""

    scorecard_path = report_dir / "scorecard.json"
    if not scorecard_path.is_file():
        return

    try:
        scorecard = json.loads(scorecard_path.read_text())

        if "highest_certification" in scorecard:
            mlflow.set_tag("highest_certification", scorecard["highest_certification"])

        if "gates_passed" in scorecard:
            mlflow.log_metric("gates_passed", scorecard["gates_passed"])
        if "gates_failed" in scorecard:
            mlflow.log_metric("gates_failed", scorecard["gates_failed"])

        for gate in scorecard.get("gates", []):
            key = gate.get("policy_key") or gate.get("gate_name", "unknown")
            if "score" in gate:
                mlflow.log_metric(f"gate_score_{key}", gate["score"])
            if "findings_count" in gate or "findings" in gate:
                count = gate.get("findings_count", len(gate.get("findings", [])))
                mlflow.log_metric(f"gate_findings_{key}", count)

        cert = scorecard.get("certification")
        if cert:
            for level_name in ("foundational", "trusted", "certified"):
                level_data = cert.get(level_name)
                if level_data:
                    mlflow.log_metric(
                        f"certification_{level_name}_passed",
                        1.0 if level_data.get("passed") else 0.0,
                    )
    except Exception as e:
        logger.warning("Failed to log scorecard metrics: %s", e)


def _log_observability_metrics(report_dir: Path) -> None:
    """Log token usage from metrics checkpoint if present."""

    checkpoint_path = report_dir / "_metrics_checkpoint.json"
    if not checkpoint_path.is_file():
        return

    try:
        checkpoint = json.loads(checkpoint_path.read_text())

        token_usage = checkpoint.get("token_usage", {})
        total_prompt = 0
        total_completion = 0
        total_calls = 0

        for phase_name, usage in token_usage.items():
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)
            total_calls += usage.get("call_count", 0)

        if total_prompt > 0 or total_completion > 0:
            metrics = {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "llm_calls_count": total_calls,
            }

            model_name = checkpoint.get("model_name")
            if model_name:
                from abevalflow.observability.cost import calculate_cost

                cost = calculate_cost(total_prompt, total_completion, model_name)
                if cost is not None:
                    metrics["estimated_cost_usd"] = cost

            mlflow.log_metrics(metrics)

        model_name = checkpoint.get("model_name")
        if model_name:
            mlflow.set_tag("model_name", model_name)

    except Exception as e:
        logger.warning("Failed to log observability metrics: %s", e)
