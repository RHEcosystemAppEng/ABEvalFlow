"""Log AEH Harbor run results to MLflow (post-evaluate, non-blocking caller).

Prefers upstream AEH ``skills/eval-mlflow/scripts/log_results.py`` when present
(under ``/opt/agent-eval-harness`` or ``AGENT_EVAL_HARNESS_ROOT``). Falls back
to a minimal metrics/params logger from ``run_result.json`` + ``summary.yaml``.

Usage::

    python scripts/log_aeh_mlflow.py \\
        --run-id <pipeline-run-id or treatment-<id>> \\
        --config /path/to/eval.yaml \\
        --runs-dir /workspace/source/reports \\
        --tracking-uri http://mlflow.example:5000

Skip (exit 0) when ``--enabled false`` or tracking URI is empty.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_AEH_LOG_RESULTS_CANDIDATES = (Path("/opt/agent-eval-harness/skills/eval-mlflow/scripts/log_results.py"),)


def _resolve_log_results_script() -> Path | None:
    root = os.environ.get("AGENT_EVAL_HARNESS_ROOT", "").strip()
    candidates: list[Path] = []
    if root:
        candidates.append(Path(root) / "skills/eval-mlflow/scripts/log_results.py")
    candidates.extend(_AEH_LOG_RESULTS_CANDIDATES)
    for path in candidates:
        if path.is_file():
            return path
    return None


def _mlflow_importable() -> bool:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def _run_aeh_log_results(*, script: Path, run_id: str, config: Path, runs_dir: Path) -> int:
    """Run upstream log_results.py.

    Returns 0 on real success. Returns non-zero when the script no-ops because
    ``mlflow`` is missing (upstream exits 0 with a message — treat as failure
    so we can fall back to the minimal logger).
    """
    env = os.environ.copy()
    env["AGENT_EVAL_RUNS_DIR"] = str(runs_dir)
    cmd = [
        sys.executable,
        str(script),
        "--run-id",
        run_id,
        "--config",
        str(config),
    ]
    logger.info("Running AEH MLflow logger: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    combined = f"{result.stdout}\n{result.stderr}"
    if "MLflow not installed" in combined:
        logger.warning("AEH log_results.py skipped (mlflow package missing)")
        return 2
    return result.returncode


def _minimal_mlflow_log(*, run_id: str, config: Path, runs_dir: Path, tracking_uri: str) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("mlflow is not installed; pip install 'mlflow[genai]' or use AEH image with mlflow") from exc

    raw = yaml.safe_load(config.read_text()) or {}
    skill = str(raw.get("skill") or config.parent.name)
    mlflow_cfg = raw.get("mlflow") if isinstance(raw.get("mlflow"), dict) else {}
    experiment = str(mlflow_cfg.get("experiment") or skill or "abevalflow-aeh")

    run_dir = runs_dir / skill / run_id
    if not run_dir.is_dir():
        # Pairwise control/treatment dirs use control-<id> / treatment-<id>
        # when --run-id already includes the prefix.
        run_dir = runs_dir / skill / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"AEH run dir not found: {run_dir}")

    run_result: dict = {}
    rr_path = run_dir / "run_result.json"
    if rr_path.is_file():
        run_result = json.loads(rr_path.read_text()) or {}

    summary: dict = {}
    summary_path = run_dir / "summary.yaml"
    if summary_path.is_file():
        loaded = yaml.safe_load(summary_path.read_text()) or {}
        if isinstance(loaded, dict):
            summary = loaded

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("skill", skill)
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("execution_mode", run_result.get("execution_mode", "harbor"))
        if run_result.get("model"):
            mlflow.log_param("model", run_result["model"])

        tokens = run_result.get("token_usage") or {}
        if isinstance(tokens, dict):
            for key in ("input", "output", "prompt", "completion", "total"):
                if key in tokens and tokens[key] is not None:
                    mlflow.log_metric(f"tokens_{key}", float(tokens[key]))
            # Harbor-style nested usage
            for key in ("input_tokens", "output_tokens", "cache_read", "cache_create"):
                if key in tokens and tokens[key] is not None:
                    mlflow.log_metric(f"tokens_{key}", float(tokens[key]))

        if run_result.get("cost_usd") is not None:
            mlflow.log_metric("cost_usd", float(run_result["cost_usd"]))
        if run_result.get("mean_reward") is not None:
            mlflow.log_metric("mean_reward", float(run_result["mean_reward"]))
        elif summary.get("mean_reward") is not None:
            mlflow.log_metric("mean_reward", float(summary["mean_reward"]))

        if summary_path.is_file():
            mlflow.log_artifact(str(summary_path))
        if rr_path.is_file():
            mlflow.log_artifact(str(rr_path))
        report_html = run_dir / "report.html"
        if report_html.is_file():
            mlflow.log_artifact(str(report_html))

    logger.info("Minimal MLflow log complete: experiment=%s run_id=%s", experiment, run_id)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Log AEH results to MLflow")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", ""))
    parser.add_argument(
        "--enabled",
        default="true",
        help="Set to false/0/no to skip (default: true when URI set by caller)",
    )
    args = parser.parse_args(argv)

    enabled = str(args.enabled).strip().lower() in {"1", "true", "yes", "on"}
    tracking_uri = (args.tracking_uri or "").strip()
    if not enabled:
        logger.info("MLflow logging disabled (--enabled=%s)", args.enabled)
        return 0
    if not tracking_uri:
        logger.info("MLflow logging skipped (empty tracking URI)")
        return 0

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    config = args.config.resolve()
    runs_dir = args.runs_dir.resolve()
    if not config.is_file():
        logger.error("Config not found: %s", config)
        return 1

    if not _mlflow_importable():
        logger.error(
            "Python package 'mlflow' is not installed in this environment; install with: python -m pip install mlflow"
        )
        return 1

    script = _resolve_log_results_script()
    if script is not None:
        rc = _run_aeh_log_results(
            script=script,
            run_id=args.run_id,
            config=config,
            runs_dir=runs_dir,
        )
        if rc == 0:
            return 0
        logger.warning("AEH log_results.py exited %s; trying minimal logger", rc)

    try:
        _minimal_mlflow_log(
            run_id=args.run_id,
            config=config,
            runs_dir=runs_dir,
            tracking_uri=tracking_uri,
        )
    except Exception:
        logger.exception("MLflow logging failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
