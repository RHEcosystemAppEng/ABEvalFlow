"""Validate a submission directory against the submission contract.

Checks vary by eval-engine mode:

Harbor (default):
  1. instruction.md exists and is non-empty
  2. skills/ contains at least one SKILL.md (flat or nested layout)
  3. tests/test_outputs.py compiles
  4. tests/llm_judge.py compiles if present
  5. metadata.yaml passes Pydantic schema validation
  6. supportive/ total size < 50 MB

ASE (agent-skills-eval):
  1. SKILL.md exists with valid YAML frontmatter containing 'name'
  2. evals/evals.json exists and is valid
  3. metadata.yaml passes Pydantic schema validation

Both: all checks from both engines apply.

Exit codes: 0 = pass, 1 = validation failure (structured JSON on stdout).
"""

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from abevalflow.schemas import EvalEngine, SubmissionMetadata

logger = logging.getLogger(__name__)

MAX_SUPPORTIVE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

_YAML_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


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


# ---------------------------------------------------------------------------
# ASE-specific checks
# ---------------------------------------------------------------------------

def _check_skill_md_frontmatter(submission_dir: Path) -> list[str]:
    """Validate that at least one SKILL.md has YAML frontmatter with 'name'."""
    skill_files: list[Path] = []

    skills_dir = submission_dir / "skills"
    if skills_dir.is_dir():
        top_level = skills_dir / "SKILL.md"
        if top_level.is_file():
            skill_files.append(top_level)
        for child in sorted(skills_dir.iterdir()):
            if child.is_dir():
                nested = child / "SKILL.md"
                if nested.is_file():
                    skill_files.append(nested)

    if not skill_files:
        return ["No SKILL.md found under skills/ for ASE validation"]

    errors: list[str] = []
    for sf in skill_files:
        rel = sf.relative_to(submission_dir)
        content = sf.read_text()
        match = _YAML_FRONTMATTER_RE.match(content)
        if not match:
            errors.append(f"{rel} is missing YAML frontmatter (---...--- block)")
            continue
        try:
            fm = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            errors.append(f"{rel} has invalid YAML frontmatter: {exc}")
            continue
        if not isinstance(fm, dict):
            errors.append(f"{rel} frontmatter must be a YAML mapping")
            continue
        if not fm.get("name"):
            errors.append(f"{rel} frontmatter is missing required 'name' field")

    return errors


def _check_evals_json(submission_dir: Path) -> list[str]:
    """Validate evals/evals.json for agent-skills-eval format.
    
    Note: Missing evals.json is NOT an error - the pipeline will generate it
    from SKILL.md. This function only validates format when the file exists.
    """
    evals_path = submission_dir / "evals" / "evals.json"
    if not evals_path.is_file():
        logger.info("evals/evals.json not present - will be generated from SKILL.md")
        return []  # Not an error - pipeline will generate it

    try:
        data = json.loads(evals_path.read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        return [f"evals/evals.json is not valid JSON: {exc}"]

    if not isinstance(data, dict):
        return ["evals/evals.json must be a JSON object"]

    evals = data.get("evals")
    if not isinstance(evals, list) or len(evals) == 0:
        return ["evals/evals.json must contain a non-empty 'evals' array"]

    errors: list[str] = []
    for i, ev in enumerate(evals):
        prefix = f"evals/evals.json evals[{i}]"
        if not isinstance(ev, dict):
            errors.append(f"{prefix}: must be a JSON object")
            continue
        if not ev.get("prompt"):
            errors.append(f"{prefix}: missing required 'prompt' field")
        if not ev.get("assertions") and not ev.get("expected_output"):
            errors.append(f"{prefix}: must have 'assertions' or 'expected_output'")

    return errors


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_submission(
    submission_dir: Path,
    eval_engine: EvalEngine = EvalEngine.HARBOR,
) -> list[str]:
    """Run validation checks based on the eval engine and return error strings."""
    logger.info("Validating submission: %s (eval_engine=%s)", submission_dir, eval_engine)
    errors: list[str] = []

    run_harbor = eval_engine in (EvalEngine.HARBOR, EvalEngine.BOTH)
    run_ase = eval_engine in (EvalEngine.ASE, EvalEngine.BOTH)

    metadata_errors, metadata = _check_metadata_yaml(submission_dir)
    errors.extend(metadata_errors)

    errors.extend(_check_skills_dir(submission_dir))

    if run_harbor:
        errors.extend(_check_instruction_md(submission_dir))
        errors.extend(_check_py_compiles(submission_dir / "tests" / "test_outputs.py"))
        llm_judge = submission_dir / "tests" / "llm_judge.py"
        if llm_judge.is_file():
            errors.extend(_check_py_compiles(llm_judge))

    if run_ase:
        errors.extend(_check_skill_md_frontmatter(submission_dir))
        errors.extend(_check_evals_json(submission_dir))

    errors.extend(_check_supportive_size(submission_dir))

    if errors:
        logger.warning("Validation failed with %d error(s)", len(errors))
    else:
        logger.info("Validation passed")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a submission directory")
    parser.add_argument("submission_dir", type=Path, help="Path to the submission directory")
    parser.add_argument(
        "--eval-engine",
        type=str,
        choices=["harbor", "ase", "both"],
        default="harbor",
        help="Evaluation engine (controls which checks run)",
    )
    args = parser.parse_args(argv)

    submission_dir: Path = args.submission_dir
    if not submission_dir.is_dir():
        result = {"valid": False, "errors": [f"Not a directory: {submission_dir}"]}
        print(json.dumps(result, indent=2))
        return 1

    engine = EvalEngine(args.eval_engine)
    errors = validate_submission(submission_dir, eval_engine=engine)
    result = {"valid": len(errors) == 0, "errors": errors}
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
