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
from abevalflow.schemas import OperationalLimits, SubmissionMetadata

logger = logging.getLogger(__name__)

LOGGING_SUPPRESSION_PATTERNS = [
    re.compile(r"\bdo\s+not\s+log\b", re.IGNORECASE),
    re.compile(r"\bdisable\s+logging\b", re.IGNORECASE),
    re.compile(r"\bsuppress\s+output\b", re.IGNORECASE),
    re.compile(r"\bno\s+logging\b", re.IGNORECASE),
    re.compile(r"\bturn\s+off\s+log", re.IGNORECASE),
]

SECURITY_QUALIFIERS = re.compile(
    r"\b(passwords?|credentials?|secrets?|pii|tokens?|sensitive|personal|api[_-]?keys?|auth|bearer|private)\b",
    re.IGNORECASE,
)


EXPECTED_RESOURCE_FIELDS = ["cpus", "memory_mb", "agent_timeout_sec"]


def _check_resource_declaration(raw_data: dict) -> list[str]:
    """Check that resource fields are explicitly declared, not just using defaults."""
    missing = [f for f in EXPECTED_RESOURCE_FIELDS if f not in raw_data]
    if missing:
        return [f"Resource fields not explicitly declared: {', '.join(missing)}"]
    return []


def _check_resource_limits(metadata: SubmissionMetadata, limits: OperationalLimits) -> list[str]:
    """Check that resource requests are within policy limits."""
    issues = []
    if metadata.cpus > limits.max_cpus:
        issues.append(f"CPU request ({metadata.cpus}) exceeds max allowed ({limits.max_cpus})")
    if metadata.memory_mb > limits.max_memory_mb:
        issues.append(f"Memory request ({metadata.memory_mb}MB) exceeds max allowed ({limits.max_memory_mb}MB)")
    return issues


def _check_timeout_compliance(metadata: SubmissionMetadata, limits: OperationalLimits) -> list[str]:
    """Check that timeouts are within reasonable bounds."""
    issues = []
    if metadata.agent_timeout_sec > limits.max_agent_timeout_sec:
        issues.append(
            f"Agent timeout ({metadata.agent_timeout_sec}s) exceeds max allowed ({limits.max_agent_timeout_sec}s)"
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
        elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
            issues.append(f"'except {node.type.id}: pass' at line {node.lineno} — hides all errors")

    return issues


def _check_logging_suppression(skill_path: Path) -> list[str]:
    """Check SKILL.md for patterns that suppress agent logging.

    Skips lines that contain security qualifiers (e.g., "Do not log passwords")
    since those are legitimate security instructions, not logging suppression.
    """
    issues = []
    if not skill_path.exists():
        return issues

    content = skill_path.read_text()
    in_code_fence = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence or stripped.startswith(">"):
            continue
        for pattern in LOGGING_SUPPRESSION_PATTERNS:
            match = pattern.search(line)
            if match and not SECURITY_QUALIFIERS.search(line):
                issues.append(f"Logging suppression pattern found: '{match.group()}'")

    return issues


SKIPPED_CHECKS = [
    "resource declarations",
    "resource limits",
    "timeout compliance",
    "error handling",
    "logging suppression",
]


def check_operational_policy(
    submission_dir: Path,
    limits: OperationalLimits | None = None,
) -> CheckResult:
    """Run all operational policy checks on a submission.

    Args:
        submission_dir: Path to the submission directory containing
            metadata.yaml, skills/SKILL.md, and tests/test_outputs.py.
        limits: Optional configurable limits from CertificationPolicy.
            Uses OperationalLimits defaults if not provided.

    Returns:
        CheckResult with aggregated pass/fail, score, and message.
    """
    if limits is None:
        limits = OperationalLimits()

    if not limits.enabled:
        skipped = ", ".join(SKIPPED_CHECKS)
        return CheckResult(
            check_id=CheckId.OPERATIONAL_POLICY_COMPLIANCE,
            name="Operational Policy Compliance",
            passed=True,
            score=1.0,
            message=f"Operational policy check disabled — not checking: {skipped}",
        )

    all_issues: list[tuple[str, str]] = []

    metadata_path = submission_dir / "metadata.yaml"
    if metadata_path.exists():
        try:
            data = yaml.safe_load(metadata_path.read_text()) or {}
            for msg in _check_resource_declaration(data):
                all_issues.append(("low", msg))
            metadata = SubmissionMetadata(**data)
            for msg in _check_resource_limits(metadata, limits):
                all_issues.append(("high", msg))
            for msg in _check_timeout_compliance(metadata, limits):
                all_issues.append(("high", msg))
        except (ValidationError, yaml.YAMLError) as e:
            logger.warning("Failed to parse metadata.yaml for policy check: %s", e)
    else:
        logger.info("No metadata.yaml found, skipping resource/timeout checks")

    tests_dir = submission_dir / "tests"
    if tests_dir.exists():
        for test_file in tests_dir.glob("*.py"):
            for msg in _check_error_handling(test_file):
                all_issues.append(("high", msg))

    root_skill = submission_dir / "SKILL.md"
    if root_skill.exists():
        for msg in _check_logging_suppression(root_skill):
            all_issues.append(("low", msg))

    skills_dir = submission_dir / "skills"
    if skills_dir.exists():
        for skill_file in skills_dir.glob("*.md"):
            for msg in _check_logging_suppression(skill_file):
                all_issues.append(("low", msg))

    high_issues = [(sev, msg) for sev, msg in all_issues if sev == "high"]
    passed = len(high_issues) == 0
    if not all_issues:
        score = 1.0
        message = "All operational policy checks passed"
    else:
        penalty = sum(0.35 if sev == "high" else 0.15 for sev, _ in all_issues)
        score = max(0.0, 1.0 - penalty)
        messages = [f"[{sev}] {msg}" for sev, msg in all_issues]
        message = f"Operational policy issues ({len(all_issues)}): " + "; ".join(messages)

    return CheckResult(
        check_id=CheckId.OPERATIONAL_POLICY_COMPLIANCE,
        name="Operational Policy Compliance",
        passed=passed,
        score=score,
        message=message,
    )
