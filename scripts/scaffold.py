"""Generate skilled and unskilled task directories from a validated submission.

Usage:
    python scripts/scaffold.py <submission-dir> <output-dir>

Produces two directories under <output-dir>:
    tasks/<skill-name>/          -- skilled variant (with skills/ and docs/)
    tasks-no-skills/<skill-name>/ -- unskilled variant (without skills/ and docs/)
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

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

SKILLED_COPY_DIRS = ("skills", "docs", "tests", "supportive", "scripts")
UNSKILLED_COPY_DIRS = ("tests", "supportive", "scripts")


def _load_metadata(submission_dir: Path) -> dict:
    meta_path = submission_dir / "metadata.yaml"
    with meta_path.open() as f:
        return yaml.safe_load(f)


def _build_template_context(metadata: dict, submission_dir: Path, variant: str) -> dict:
    """Build the Jinja2 template context from metadata and directory inspection."""
    tags = metadata.get("tags") or []
    has_llm_judge = (submission_dir / "tests" / "llm_judge.py").is_file()
    return {
        "skill_name": metadata["name"],
        "persona": metadata.get("persona") or "general",
        "description": metadata.get("description") or "",
        "version": metadata.get("version", "0.1.0"),
        "author": metadata.get("author") or "",
        "tags": tags,
        "variant": variant,
        "has_supportive": (submission_dir / "supportive").is_dir(),
        "has_scripts": (submission_dir / "scripts").is_dir(),
        "has_docs": (submission_dir / "docs").is_dir(),
        "has_llm_judge": has_llm_judge,
        "llm_env_key": metadata.get("llm_judge_env_key", "LLM_API_KEY"),
        "model_name": metadata.get("llm_judge_model", ""),
        "agent_timeout": metadata.get("agent_timeout_sec", 600.0),
        "agent_setup_timeout": metadata.get("agent_setup_timeout_sec", 600.0),
        "verifier_timeout": metadata.get("verifier_timeout_sec", 120.0),
        "build_timeout": metadata.get("build_timeout_sec", 600.0),
        "cpus": metadata.get("cpus", 1),
        "memory_mb": metadata.get("memory_mb", 2048),
        "storage_mb": metadata.get("storage_mb", 10240),
    }


def _render_templates(
    jinja_env: Environment,
    context: dict,
    variant: str,
) -> dict[str, str]:
    """Render all templates for a variant, returning {filename: content}."""
    dockerfile_template = f"Dockerfile.{variant}.j2"
    return {
        "Dockerfile": jinja_env.get_template(dockerfile_template).render(context),
        "test.sh": jinja_env.get_template("test.sh.j2").render(context),
        "task.toml": jinja_env.get_template("task.toml.j2").render(context),
    }


def _copy_submission_files(
    submission_dir: Path,
    target_dir: Path,
    dirs_to_copy: tuple[str, ...],
) -> None:
    """Copy instruction.md and relevant directories into the target task directory."""
    shutil.copy2(submission_dir / "instruction.md", target_dir / "instruction.md")

    for dirname in dirs_to_copy:
        src = submission_dir / dirname
        if src.is_dir():
            shutil.copytree(src, target_dir / dirname, dirs_exist_ok=True)


def scaffold_submission(
    submission_dir: Path,
    output_dir: Path,
    templates_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Generate skilled and unskilled task directories.

    Returns the paths to (skilled_dir, unskilled_dir).
    """
    templates_dir = templates_dir or TEMPLATES_DIR
    jinja_env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        keep_trailing_newline=True,
    )

    metadata = _load_metadata(submission_dir)
    skill_name = metadata["name"]

    skilled_dir = output_dir / "tasks" / skill_name
    unskilled_dir = output_dir / "tasks-no-skills" / skill_name

    for variant, target_dir, copy_dirs in (
        ("skilled", skilled_dir, SKILLED_COPY_DIRS),
        ("unskilled", unskilled_dir, UNSKILLED_COPY_DIRS),
    ):
        context = _build_template_context(metadata, submission_dir, variant)
        rendered = _render_templates(jinja_env, context, variant)

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

        _copy_submission_files(submission_dir, environment_dir, copy_dirs)
        # instruction.md is placed at both levels: environment/ for the Docker
        # build context (COPY in Dockerfile) and task root for Harbor metadata.
        shutil.copy2(submission_dir / "instruction.md", target_dir / "instruction.md")

        logger.info("Scaffolded %s variant at %s", variant, target_dir)

    return skilled_dir, unskilled_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scaffold skill submission into Harbor task dirs")
    parser.add_argument("submission_dir", type=Path, help="Path to validated submission directory")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for tasks/ and tasks-no-skills/",
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

    skilled_dir, unskilled_dir = scaffold_submission(
        args.submission_dir, args.output_dir, args.templates_dir
    )
    logger.info("Skilled:   %s", skilled_dir)
    logger.info("Unskilled: %s", unskilled_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
