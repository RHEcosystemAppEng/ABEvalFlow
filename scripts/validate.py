"""Validate a submission directory against the submission contract.

Checks:
  1. instruction.md exists and is non-empty
  2. skills/ contains at least one SKILL.md (flat or nested layout)
  3. tests/test_outputs.py compiles
  4. tests/llm_judge.py compiles if present
  5. metadata.yaml passes Pydantic schema validation
  6. supportive/ total size < 50 MB

Exit codes: 0 = pass, 1 = validation failure (structured JSON on stdout).
"""

import argparse
import ast
import json
import logging
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from abevalflow.schemas import SubmissionMetadata

logger = logging.getLogger(__name__)

MAX_SUPPORTIVE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def _check_instruction_md(submission_dir: Path) -> list[str]:
    instruction = submission_dir / "instruction.md"
    if not instruction.is_file():
        return ["instruction.md is missing"]
    if not instruction.read_text().strip():
        return ["instruction.md is empty"]
    return []


def _check_skills_dir(submission_dir: Path) -> list[str]:
    """Validate skills/ contains at least one SKILL.md.

    Supports two layouts:
      - Flat:   skills/SKILL.md  (single skill at top level)
      - Nested: skills/<name>/SKILL.md  (Claude discovers each subdirectory as a skill)

    Both are valid — Claude Code scans the skills_dir for subdirectories
    containing SKILL.md and registers each as a /skill-name command.
    """
    skills_dir = submission_dir / "skills"
    if not skills_dir.is_dir():
        return ["skills/ directory is missing"]

    skill_files: list[Path] = []
    # Flat layout: skills/SKILL.md
    top_level = skills_dir / "SKILL.md"
    if top_level.is_file():
        skill_files.append(top_level)
    # Nested layout: skills/<name>/SKILL.md
    for child in sorted(skills_dir.iterdir()):
        if child.is_dir():
            nested = child / "SKILL.md"
            if nested.is_file():
                skill_files.append(nested)

    if not skill_files:
        return ["skills/ must contain at least one SKILL.md (either skills/SKILL.md or skills/<name>/SKILL.md)"]

    errors: list[str] = []
    for sf in skill_files:
        rel = sf.relative_to(submission_dir)
        if not sf.read_text().strip():
            errors.append(f"{rel} is empty")

    return errors


def _check_py_compiles(file_path: Path) -> list[str]:
    if not file_path.is_file():
        return [f"{file_path.name} is missing"]
    try:
        source = file_path.read_text()
        ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        return [f"{file_path.name} does not compile: {exc}"]
    return []


def _check_metadata_yaml(submission_dir: Path) -> tuple[list[str], SubmissionMetadata | None]:
    metadata_path = submission_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return ["metadata.yaml is missing"], None
    try:
        raw = yaml.safe_load(metadata_path.read_text())
    except yaml.YAMLError as exc:
        return [f"metadata.yaml is not valid YAML: {exc}"], None
    if not isinstance(raw, dict):
        return ["metadata.yaml must contain a YAML mapping"], None
    try:
        model = SubmissionMetadata(**raw)
    except ValidationError as exc:
        errors = [
            f"metadata.yaml validation: {e['msg']} ({'.'.join(str(loc) for loc in e['loc'])})"
            for e in exc.errors()
        ]
        return errors, None
    return [], model


def _check_supportive_size(submission_dir: Path) -> list[str]:
    supportive_dir = submission_dir / "supportive"
    if not supportive_dir.is_dir():
        return []
    total_size = sum(f.stat().st_size for f in supportive_dir.rglob("*") if f.is_file())
    if total_size > MAX_SUPPORTIVE_SIZE_BYTES:
        size_mb = total_size / (1024 * 1024)
        return [f"supportive/ exceeds 50 MB limit ({size_mb:.1f} MB)"]
    return []


def validate_submission(submission_dir: Path) -> list[str]:
    """Run all validation checks and return a list of error strings (empty = valid)."""
    logger.info("Validating submission: %s", submission_dir)
    errors: list[str] = []

    # Parse metadata first — generation_mode determines which files are required.
    metadata_errors, metadata = _check_metadata_yaml(submission_dir)
    errors.extend(metadata_errors)

    errors.extend(_check_skills_dir(submission_dir))
    errors.extend(_check_instruction_md(submission_dir))
    errors.extend(_check_py_compiles(submission_dir / "tests" / "test_outputs.py"))

    llm_judge = submission_dir / "tests" / "llm_judge.py"
    if llm_judge.is_file():
        errors.extend(_check_py_compiles(llm_judge))

    errors.extend(_check_supportive_size(submission_dir))

    if errors:
        logger.warning("Validation failed with %d error(s)", len(errors))
    else:
        logger.info("Validation passed")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a submission directory")
    parser.add_argument("submission_dir", type=Path, help="Path to the submission directory")
    args = parser.parse_args(argv)

    submission_dir: Path = args.submission_dir
    if not submission_dir.is_dir():
        result = {"valid": False, "errors": [f"Not a directory: {submission_dir}"]}
        print(json.dumps(result, indent=2))
        return 1

    errors = validate_submission(submission_dir)
    result = {"valid": len(errors) == 0, "errors": errors}
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
