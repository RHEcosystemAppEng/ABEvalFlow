"""Operational policy compliance checks for skill submissions.

Deterministic checks that validate submissions meet operational standards:
resource limits, timeout compliance, error handling patterns, and logging
suppression. No LLM calls required.

These checks complement (not duplicate) Foundational-level schema validation.
Schema validation (Pydantic) enforces field types and ``gt=0`` constraints;
these checks enforce *policy limits* — e.g., ``cpus <= MAX_CPUS``.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from abevalflow.certification import CheckId, CheckResult
from abevalflow.schemas import SubmissionMetadata

logger = logging.getLogger(__name__)

MAX_CPUS = 4
MAX_MEMORY_MB = 8192
MAX_AGENT_TIMEOUT_SEC = 3600.0

LOGGING_SUPPRESSION_PATTERNS = [
    re.compile(r"\bdo\s+not\s+log\b", re.IGNORECASE),
    re.compile(r"\bdisable\s+logging\b", re.IGNORECASE),
    re.compile(r"\bsuppress\s+output\b", re.IGNORECASE),
    re.compile(r"\bno\s+logging\b", re.IGNORECASE),
    re.compile(r"\bturn\s+off\s+log", re.IGNORECASE),
]


def _check_resource_limits(metadata: SubmissionMetadata) -> list[str]:
    """Check that resource requests are within policy limits."""
    issues = []
    if metadata.cpus > MAX_CPUS:
        issues.append(f"CPU request ({metadata.cpus}) exceeds max allowed ({MAX_CPUS})")
    if metadata.memory_mb > MAX_MEMORY_MB:
        issues.append(f"Memory request ({metadata.memory_mb}MB) exceeds max allowed ({MAX_MEMORY_MB}MB)")
    return issues


def _check_timeout_compliance(metadata: SubmissionMetadata) -> list[str]:
    """Check that timeouts are within reasonable bounds."""
    issues = []
    if metadata.agent_timeout_sec > MAX_AGENT_TIMEOUT_SEC:
        issues.append(
            f"Agent timeout ({metadata.agent_timeout_sec}s) exceeds max allowed ({MAX_AGENT_TIMEOUT_SEC}s)"
        )
    return issues


def _check_error_handling(test_file: Path) -> list[str]:
    """Check for bare except patterns that hide failures."""
    issues = []
    if not test_file.exists():
        return issues

    try:
        tree = ast.parse(test_file.read_text())
    except SyntaxError:
        return issues

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        handler_body = node.body
        if len(handler_body) != 1:
            continue
        stmt = handler_body[0]
        is_pass = isinstance(stmt, ast.Pass)
        is_ellipsis = isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...
        if not (is_pass or is_ellipsis):
            continue
        if node.type is None:
            issues.append(f"Bare 'except: pass' at line {node.lineno} — hides all errors")
        elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
            issues.append(f"'except Exception: pass' at line {node.lineno} — hides all errors")

    return issues


def _check_logging_suppression(skill_path: Path) -> list[str]:
    """Check SKILL.md for patterns that suppress agent logging."""
    issues = []
    if not skill_path.exists():
        return issues

    content = skill_path.read_text()
    for pattern in LOGGING_SUPPRESSION_PATTERNS:
        match = pattern.search(content)
        if match:
            issues.append(f"Logging suppression pattern found: '{match.group()}'")

    return issues


def check_operational_policy(submission_dir: Path) -> CheckResult:
    """Run all operational policy checks on a submission.

    Args:
        submission_dir: Path to the submission directory containing
            metadata.yaml, skills/SKILL.md, and tests/test_outputs.py.

    Returns:
        CheckResult with aggregated pass/fail, score, and message.
    """
    all_issues: list[str] = []

    metadata_path = submission_dir / "metadata.yaml"
    if metadata_path.exists():
        try:
            data = yaml.safe_load(metadata_path.read_text())
            metadata = SubmissionMetadata(**(data or {}))
            all_issues.extend(_check_resource_limits(metadata))
            all_issues.extend(_check_timeout_compliance(metadata))
        except (ValidationError, yaml.YAMLError) as e:
            logger.warning("Failed to parse metadata.yaml for policy check: %s", e)
    else:
        logger.info("No metadata.yaml found, skipping resource/timeout checks")

    test_file = submission_dir / "tests" / "test_outputs.py"
    all_issues.extend(_check_error_handling(test_file))

    skills_dir = submission_dir / "skills"
    if skills_dir.exists():
        for skill_file in skills_dir.glob("*.md"):
            all_issues.extend(_check_logging_suppression(skill_file))

    passed = len(all_issues) == 0
    score = 1.0 if passed else max(0.0, 1.0 - (len(all_issues) * 0.25))

    if passed:
        message = "All operational policy checks passed"
    else:
        message = f"Operational policy issues ({len(all_issues)}): " + "; ".join(all_issues)

    return CheckResult(
        check_id=CheckId.OPERATIONAL_POLICY_COMPLIANCE,
        name="Operational Policy Compliance",
        passed=passed,
        score=score,
        message=message,
    )
