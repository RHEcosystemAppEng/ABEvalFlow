"""Generate treatment and control task directories from a validated submission.

Usage:
    python scripts/scaffold.py <submission-dir> <output-dir>

Produces two directories under <output-dir>:
    tasks-treatment/<skill-name>/  -- treatment variant
    tasks-control/<skill-name>/    -- control variant (baseline)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import stat
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

from abevalflow.experiment import ExperimentStrategy, get_strategy
from abevalflow.schemas import SubmissionMetadata

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

COMMON_COPY_DIRS = ("tests", "supportive", "scripts")


def _load_metadata(submission_dir: Path) -> SubmissionMetadata:
    meta_path = submission_dir / "metadata.yaml"
    with meta_path.open() as f:
        raw = yaml.safe_load(f)
    return SubmissionMetadata(**raw)


def _build_template_context(
    metadata: SubmissionMetadata,
    submission_dir: Path,
    variant: str,
    strategy: ExperimentStrategy,
) -> dict:
    """Build the Jinja2 template context from metadata and directory inspection."""
    tags = metadata.tags or []
    has_llm_judge = (submission_dir / "tests" / "llm_judge.py").is_file()

    base_context = {
        "skill_name": metadata.name,
        "persona": metadata.persona or "general",
        "description": metadata.description or "",
        "version": metadata.version,
        "author": metadata.author or "",
        "tags": tags,
        "has_supportive": (submission_dir / "supportive").is_dir(),
        "has_scripts": (submission_dir / "scripts").is_dir(),
        "has_llm_judge": has_llm_judge,
        # These were formerly ad-hoc dict reads from raw metadata; they are
        # not SubmissionMetadata fields (extra="forbid" rejects them), so the
        # hardcoded defaults are the only values they ever had in practice.
        "llm_env_key": "LLM_API_KEY",
        "model_name": "",
        "agent_timeout": metadata.agent_timeout_sec,
        "agent_setup_timeout": metadata.agent_setup_timeout_sec,
        "verifier_timeout": metadata.verifier_timeout_sec,
        "build_timeout": metadata.build_timeout_sec,
        "cpus": metadata.cpus,
        "memory_mb": metadata.memory_mb,
        "storage_mb": metadata.storage_mb,
    }

    return strategy.customize_context(base_context, variant, submission_dir)


def _render_templates(
    jinja_env: Environment,
    context: dict,
) -> dict[str, str]:
    """Render all templates for a variant, returning {filename: content}."""
    return {
        "Dockerfile": jinja_env.get_template("Dockerfile.j2").render(context),
        "test.sh": jinja_env.get_template("test.sh.j2").render(context),
        "task.toml": jinja_env.get_template("task.toml.j2").render(context),
    }


def _copy_submission_files(
    submission_dir: Path,
    build_context_dir: Path,
    strategy_copy_srcs: list[str],
) -> None:
    """Copy instruction.md and relevant directories into the build context.

    The build context (environment/) is the Docker build root. instruction.md
    is copied here so the Dockerfile can COPY it into the image.
    """
    shutil.copy2(submission_dir / "instruction.md", build_context_dir / "instruction.md")

    # Preserve insertion order, deduplicate (strategy dirs may overlap with common dirs)
    all_dirs = list(dict.fromkeys(strategy_copy_srcs + list(COMMON_COPY_DIRS)))
    for dirname in all_dirs:
        src = submission_dir / dirname
        if src.is_dir():
            shutil.copytree(src, build_context_dir / dirname, dirs_exist_ok=True)


def scaffold_submission(
    submission_dir: Path,
    output_dir: Path,
    templates_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Generate treatment and control task directories.

    Returns the paths to (treatment_dir, control_dir).
    """
    templates_dir = templates_dir or TEMPLATES_DIR
    jinja_env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )

    metadata = _load_metadata(submission_dir)
    strategy = get_strategy(metadata.experiment)

    treatment_dir = output_dir / "tasks-treatment" / metadata.name
    control_dir = output_dir / "tasks-control" / metadata.name

    for variant, target_dir in (
        ("treatment", treatment_dir),
        ("control", control_dir),
    ):
        context = _build_template_context(
            metadata, submission_dir, variant, strategy,
        )
        rendered = _render_templates(jinja_env, context)

        target_dir.mkdir(parents=True, exist_ok=True)

        environment_dir = target_dir / "environment"
        environment_dir.mkdir(exist_ok=True)

        for filename, content in rendered.items():
            if filename == "Dockerfile":
                dest = environment_dir / filename
            else:
                dest = target_dir / filename
            dest.write_text(content)
            if filename == "test.sh":
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)

        strategy_srcs = [src for src, _ in context.get("copy_pairs", [])]
        _copy_submission_files(submission_dir, environment_dir, strategy_srcs)

        # Second copy at task root: Harbor reads instruction.md from the task
        # directory (outside the build context) for display/metadata purposes.
        shutil.copy2(submission_dir / "instruction.md", target_dir / "instruction.md")

        logger.info("Scaffolded %s variant at %s", variant, target_dir)

    return treatment_dir, control_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scaffold submission into Harbor task dirs")
    parser.add_argument("submission_dir", type=Path, help="Path to validated submission directory")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for tasks-treatment/ and tasks-control/",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=None,
        help="Override templates directory (default: templates/ in repo root)",
    )
    args = parser.parse_args()

    if not args.submission_dir.is_dir():
        logger.error("Submission directory does not exist: %s", args.submission_dir)
        return 1

    treatment_dir, control_dir = scaffold_submission(
        args.submission_dir, args.output_dir, args.templates_dir
    )
    logger.info("Treatment: %s", treatment_dir)
    logger.info("Control:   %s", control_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
