"""Generate per-variant Harbor job configs for A/B evaluation.

Reads metadata.yaml from the submission directory and produces two config
files (one per variant) that can be passed to ``harbor run -c <config.yaml>``.

Each variant runs as a separate Harbor job with its own jobs directory,
producing a clean result layout::

    <results-base-dir>/
        treatment/
            <job-name>/
                <trial-1>__<uuid>/result.json
                ...
        control/
            <job-name>/
                <trial-1>__<uuid>/result.json
                ...

Usage:
    python scripts/generate_eval_config.py \\
        --submission-dir /workspace/submissions/my-submission \\
        --treatment-task-dir /workspace/tasks-treatment/my-submission \\
        --control-task-dir /workspace/tasks-control/my-submission \\
        --eval-mode prebuilt \\
        --treatment-image-ref registry/ns/img@sha256:abc \\
        --control-image-ref registry/ns/img@sha256:def \\
        --output-dir /workspace/eval-configs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from abevalflow.schemas import SubmissionMetadata

logger = logging.getLogger(__name__)

EVAL_MODES = ("prebuilt", "local-build")
VARIANTS = ("treatment", "control")

# Harbor's default agent/verifier/setup timeouts used as reference baselines
# for computing timeout multipliers from the absolute seconds in metadata.yaml.
_HARBOR_DEFAULT_AGENT_TIMEOUT = 600.0
_HARBOR_DEFAULT_VERIFIER_TIMEOUT = 120.0
_HARBOR_DEFAULT_SETUP_TIMEOUT = 600.0
_HARBOR_DEFAULT_BUILD_TIMEOUT = 600.0


def load_metadata(submission_dir: Path) -> SubmissionMetadata:
    """Load and validate metadata.yaml from a submission directory."""
    meta_path = submission_dir / "metadata.yaml"
    with meta_path.open() as f:
        raw = yaml.safe_load(f)
    return SubmissionMetadata(**raw)


def _timeout_multiplier(actual: float, default: float) -> float:
    """Compute a timeout multiplier relative to Harbor's default.

    Returns 1.0 when actual equals the default, >1.0 for longer timeouts.
    """
    if default <= 0:
        return 1.0
    return actual / default


def _resolve_llm_params(
    metadata: SubmissionMetadata,
    llm_model: str,
    llm_api_base: str,
    llm_api_key: str,
    llm_agent_wrapper: str,
) -> tuple[str, str, str, str]:
    """Apply per-submission LLM overrides from metadata.yaml over pipeline defaults."""
    if metadata.llm:
        if metadata.llm.model is not None:
            llm_model = metadata.llm.model
        if metadata.llm.api_base is not None:
            llm_api_base = metadata.llm.api_base
        if metadata.llm.api_key is not None:
            llm_api_key = metadata.llm.api_key
        if metadata.llm.agent_wrapper is not None:
            llm_agent_wrapper = metadata.llm.agent_wrapper
    return llm_model, llm_api_base, llm_api_key, llm_agent_wrapper


def build_variant_config(
    metadata: SubmissionMetadata,
    variant: str,
    task_dir: str,
    eval_mode: str,
    jobs_dir: str,
    image_ref: str = "",
    llm_model: str = "",
    llm_api_base: str = "",
    llm_api_key: str = "",
    llm_agent_wrapper: str = "",
) -> dict[str, Any]:
    """Build a Harbor JobConfig dict for a single variant.

    Each variant gets its own job so results land in a variant-specific
    directory and trial classification is unambiguous.
    """
    if eval_mode == "prebuilt" and not image_ref:
        raise ValueError(
            f"image_ref is required for variant '{variant}' in prebuilt mode"
        )

    llm_model, llm_api_base, llm_api_key, llm_agent_wrapper = _resolve_llm_params(
        metadata, llm_model, llm_api_base, llm_api_key, llm_agent_wrapper,
    )

    task: dict[str, Any] = {"path": task_dir}

    env_block: dict[str, Any] = {
        "type": "openshift",
        "delete": True,
        "override_memory_mb": metadata.memory_mb,
        "override_storage_mb": metadata.storage_mb,
    }

    kwargs: dict[str, Any] = {
        "cpu_request": "100m",
        "memory_limit_multiplier": 1.5,
    }

    if eval_mode == "prebuilt":
        kwargs["image_ref"] = image_ref
    env_block["kwargs"] = kwargs

    if eval_mode != "prebuilt":
        env_block["force_build"] = True

    agent_mult = _timeout_multiplier(
        metadata.agent_timeout_sec, _HARBOR_DEFAULT_AGENT_TIMEOUT
    )
    verifier_mult = _timeout_multiplier(
        metadata.verifier_timeout_sec, _HARBOR_DEFAULT_VERIFIER_TIMEOUT
    )
    setup_mult = _timeout_multiplier(
        metadata.agent_setup_timeout_sec, _HARBOR_DEFAULT_SETUP_TIMEOUT
    )
    build_mult = _timeout_multiplier(
        metadata.build_timeout_sec, _HARBOR_DEFAULT_BUILD_TIMEOUT
    )

    agent_config: dict[str, Any] = {}
    if llm_agent_wrapper:
        agent_config["name"] = llm_agent_wrapper
        if llm_model:
            agent_config["model_name"] = llm_model
        if llm_api_base:
            agent_config.setdefault("env", {})["OPENAI_BASE_URL"] = llm_api_base
    elif llm_model:
        agent_config["name"] = "claude-code"
        agent_config["model_name"] = llm_model
        env: dict[str, str] = {}
        if llm_api_key:
            env["ANTHROPIC_API_KEY"] = llm_api_key
        if llm_api_base:
            env["ANTHROPIC_BASE_URL"] = llm_api_base
        if env:
            agent_config["env"] = env

    config: dict[str, Any] = {
        "job_name": f"{metadata.name}-{variant}",
        "jobs_dir": jobs_dir,
        "n_attempts": int(os.environ.get("OVERRIDE_N_TRIALS", metadata.experiment.n_trials)),
        "timeout_multiplier": 1.0,
        "agent_timeout_multiplier": agent_mult,
        "verifier_timeout_multiplier": verifier_mult,
        "agent_setup_timeout_multiplier": setup_mult,
        "environment_build_timeout_multiplier": build_mult,
        "n_concurrent_trials": 1,
        "environment": env_block,
        "agents": [agent_config],
        "tasks": [task],
    }

    return config


def generate_eval_configs(
    submission_dir: Path,
    treatment_task_dir: str,
    control_task_dir: str,
    output_dir: Path,
    eval_mode: str,
    results_base_dir: str,
    treatment_image_ref: str = "",
    control_image_ref: str = "",
    llm_model: str = "",
    llm_api_base: str = "",
    llm_api_key: str = "",
    llm_agent_wrapper: str = "",
) -> dict[str, dict[str, Any]]:
    """Generate per-variant Harbor configs, write YAML files, return both."""
    metadata = load_metadata(submission_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_args = dict(zip(
        VARIANTS,
        ((treatment_task_dir, treatment_image_ref),
         (control_task_dir, control_image_ref)),
    ))

    configs: dict[str, dict[str, Any]] = {}
    for variant, (task_dir, img_ref) in variant_args.items():
        jobs_dir = f"{results_base_dir}/{variant}"
        config = build_variant_config(
            metadata=metadata,
            variant=variant,
            task_dir=task_dir,
            eval_mode=eval_mode,
            jobs_dir=jobs_dir,
            image_ref=img_ref,
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            llm_api_key=llm_api_key,
            llm_agent_wrapper=llm_agent_wrapper,
        )
        out_path = output_dir / f"{variant}-config.yaml"
        with out_path.open("w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info("Wrote %s config to %s", variant, out_path)
        configs[variant] = config

    return configs


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate per-variant Harbor eval configs from submission metadata",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        required=True,
        help="Path to the submission directory containing metadata.yaml",
    )
    parser.add_argument(
        "--treatment-task-dir",
        required=True,
        help="Path to the scaffolded treatment task directory",
    )
    parser.add_argument(
        "--control-task-dir",
        required=True,
        help="Path to the scaffolded control task directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the per-variant config YAML files",
    )
    parser.add_argument(
        "--eval-mode",
        choices=EVAL_MODES,
        required=True,
        help="'prebuilt' uses digest image refs; 'local-build' lets Harbor build from Dockerfiles",
    )
    parser.add_argument(
        "--treatment-image-ref",
        default="",
        help="Digest-based image ref for treatment variant (required for prebuilt mode)",
    )
    parser.add_argument(
        "--control-image-ref",
        default="",
        help="Digest-based image ref for control variant (required for prebuilt mode)",
    )
    parser.add_argument(
        "--results-base-dir",
        default="eval-results",
        help="Base directory for Harbor job results (default: eval-results)",
    )
    parser.add_argument(
        "--llm-model",
        default="",
        help="LLM model name (e.g. claude-sonnet). Empty falls back to oracle.",
    )
    parser.add_argument(
        "--llm-api-base",
        default="",
        help="Base URL of the LLM API proxy (e.g. http://litellm:4000)",
    )
    parser.add_argument(
        "--llm-api-key",
        default="",
        help="API key for the LLM provider (mock for LiteLLM proxy)",
    )
    parser.add_argument(
        "--llm-agent-wrapper",
        default="",
        help="Agent wrapper for non-Claude models (e.g. opencode). Empty uses claude-code.",
    )

    args = parser.parse_args(argv)

    if args.eval_mode == "prebuilt":
        if not args.treatment_image_ref or not args.control_image_ref:
            parser.error(
                "--treatment-image-ref and --control-image-ref are required "
                "when --eval-mode is 'prebuilt'"
            )

    if not args.submission_dir.is_dir():
        logger.error("Submission directory does not exist: %s", args.submission_dir)
        return 1

    generate_eval_configs(
        submission_dir=args.submission_dir,
        treatment_task_dir=args.treatment_task_dir,
        control_task_dir=args.control_task_dir,
        output_dir=args.output_dir,
        eval_mode=args.eval_mode,
        results_base_dir=args.results_base_dir,
        treatment_image_ref=args.treatment_image_ref,
        control_image_ref=args.control_image_ref,
        llm_model=args.llm_model,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        llm_agent_wrapper=args.llm_agent_wrapper,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
