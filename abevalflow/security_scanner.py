"""Security scanner for AI skill submissions.

Scans SKILL.md files for prompt injection patterns (context-aware severity)
and credential access patterns (sensitive paths, environment variables, and
dangerous commands).  All checks are deterministic regex matching with no
LLM calls.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class FindingCategory(StrEnum):
    PROMPT_INJECTION = "prompt_injection"
    CREDENTIAL_ACCESS = "credential_access"


class SecurityFinding(BaseModel):
    """A single security finding from the scanner."""

    severity: Severity
    category: FindingCategory
    file: str
    line: int
    message: str


class SecurityScanResult(BaseModel):
    """Aggregated result of scanning a submission for security issues."""

    passed: bool
    findings: list[SecurityFinding]
    summary: str


# Prompt injection patterns

_I = re.I

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore previous instructions", re.compile(
        r"ignore\s+(all\s+)?previous\s+instructions", _I)),
    ("disregard prior", re.compile(
        r"disregard\s+(all\s+)?(prior|previous|above)", _I)),
    ("you are now", re.compile(
        r"you\s+are\s+now\s+(?:a|an|the)\s+", _I)),
    ("system prompt override", re.compile(
        r"system\s*prompt\s*(override|injection|change)", _I)),
    ("override instructions", re.compile(
        r"override\s+(all\s+)?(instructions|rules|guidelines)", _I)),
    ("new instructions", re.compile(
        r"new\s+instructions?\s*:", _I)),
    ("jailbreak attempt", re.compile(
        r"(\bDAN\b|do\s+anything\s+now|developer\s+mode)", _I)),
    ("prompt leak", re.compile(
        r"(reveal|show|print|output)\s+(your|the)\s+(system\s+)?prompt",
        _I)),
    ("role hijack", re.compile(
        r"forget\s+(everything|all|your)\s+(you|instructions|rules)", _I)),
    ("hidden instruction", re.compile(
        r"<\s*(?:system|instruction|hidden)\s*>", _I)),
    ("role play", re.compile(
        r"pretend\s+(?:to\s+be|you\s+are)\s+(?:a|an|the)\s+", _I)),
    ("encoding evasion", re.compile(
        r"(?:in\s+base64|encode\s+(?:as|in|to)\s+base64|base64\s+encod)",
        _I)),
    ("repeat after me", re.compile(
        r"repeat\s+after\s+me", _I)),
    ("bypass safety", re.compile(
        r"(?:ignore\s+safety|bypass\s+(?:filter|safety|restriction))", _I)),
    ("output control", re.compile(
        r"output\s+the\s+following\s+exactly", _I)),
    ("markdown image exfiltration", re.compile(
        r"!\[.*?\]\(https?://", _I)),
    ("translate evasion", re.compile(
        r"translate\s+(?:this|the\s+following)\s+(?:to|into)\s+", _I)),
    ("act as", re.compile(
        r"act\s+as\s+(?:a|an|the|if)\s+", _I)),
    ("simulate mode", re.compile(
        r"(?:enter|enable|activate)\s+(?:\w+\s+)?mode", _I)),
    ("data exfiltration via url", re.compile(
        r"(?:curl|wget|fetch)\s+https?://", _I)),
]

# Credential access patterns

_SENSITIVE_PATHS: list[re.Pattern[str]] = [
    re.compile(r"~/\.ssh/", re.I),
    re.compile(r"~/\.aws/credentials", re.I),
    re.compile(r"~/\.aws/config", re.I),
    re.compile(r"~/\.config/gcloud", re.I),
    re.compile(r"~/\.kube/config", re.I),
    re.compile(r"/etc/shadow", re.I),
    re.compile(r"/etc/passwd", re.I),
    re.compile(r"~/\.netrc", re.I),
    re.compile(r"~/\.env\b"),
    re.compile(r"~/\.docker/config\.json", re.I),
    re.compile(r"~/\.npmrc\b"),
    re.compile(r"~/\.pypirc\b"),
    re.compile(r"~/\.gitconfig", re.I),
    re.compile(r"~/\.git-credentials", re.I),
    re.compile(r"~/\.gnupg/", re.I),
    re.compile(r"~/\.config/gh/", re.I),
]

_SENSITIVE_ENV_VARS: list[re.Pattern[str]] = [
    re.compile(r"\$(?:ANTHROPIC|OPENAI|GEMINI|GOOGLE)_API_KEY"),
    re.compile(r"\$(?:AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)"),
    re.compile(r"\$AWS_ACCESS_KEY_ID"),
    re.compile(r"\$(?:DATABASE_URL|DB_PASSWORD)"),
    re.compile(r"\$(?:GITHUB_TOKEN|GH_TOKEN)"),
    re.compile(r"\$(?:SECRET_KEY|PRIVATE_KEY)"),
    re.compile(r"\$SLACK_TOKEN"),
    re.compile(r"\$STRIPE_SECRET_KEY"),
    re.compile(r"\$JWT_SECRET"),
    re.compile(r"\$ENCRYPTION_KEY"),
    re.compile(r"\$REDIS_PASSWORD"),
    re.compile(r"\$(?:AZURE|HUGGINGFACE)_API_KEY"),
]

_DANGEROUS_COMMANDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsudo\s+"), "sudo"),
    (re.compile(r"\bchmod\s+777\b"), "chmod 777"),
    (re.compile(r"\bchown\s+root\b"), "chown root"),
    (re.compile(r"\brm\s+-rf\s+/"), "rm -rf /"),
    (re.compile(r"\bcurl\s+.*\|\s*(?:ba)?sh\b"), "curl | sh"),
]

_EXAMPLE_MARKERS = ("for example", "e.g.", "such as", "like:")


def scan_file_for_injections(
    file_path: str, content: str,
) -> list[SecurityFinding]:
    """Scan file content for prompt injection patterns.

    Context-aware: findings inside code fences or example/quote contexts
    are reported as WARNING instead of ERROR.
    """
    findings: list[SecurityFinding] = []
    lines = content.split("\n")
    in_code_fence = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        for label, pattern in _INJECTION_PATTERNS:
            if pattern.search(line):
                is_quoted = (
                    stripped.startswith(">") or stripped.startswith('"')
                )
                is_example = any(
                    w in line.lower() for w in _EXAMPLE_MARKERS
                )

                if in_code_fence:
                    severity = Severity.WARNING
                    msg = (
                        f"Line {i + 1} contains '{label}' inside a"
                        " code block - likely safe"
                        " (documentation or example)."
                    )
                elif is_quoted or is_example:
                    severity = Severity.WARNING
                    msg = (
                        f"Line {i + 1} contains '{label}' in a quote"
                        " or example - likely safe."
                    )
                else:
                    severity = Severity.ERROR
                    msg = (
                        f"Line {i + 1} contains a word pattern"
                        f" ('{label}') that could be used to"
                        " manipulate the agent. Check if this is"
                        " intentional content or an actual risk."
                    )

                logger.debug(
                    "%s: %s in %s (line %d)",
                    severity, label, file_path, i + 1,
                )
                findings.append(
                    SecurityFinding(
                        severity=severity,
                        category=FindingCategory.PROMPT_INJECTION,
                        file=file_path,
                        line=i + 1,
                        message=msg,
                    )
                )
                break

    return findings


def scan_file_for_credentials(
    file_path: str, content: str,
) -> list[SecurityFinding]:
    """Scan file content for credential access patterns.

    All findings are ERROR severity since credential access in skill
    definitions is never acceptable.
    """
    findings: list[SecurityFinding] = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        for pattern in _SENSITIVE_PATHS:
            match = pattern.search(line)
            if match:
                findings.append(
                    SecurityFinding(
                        severity=Severity.ERROR,
                        category=FindingCategory.CREDENTIAL_ACCESS,
                        file=file_path,
                        line=i + 1,
                        message=(
                            f"References sensitive path"
                            f" '{match.group(0)}'"
                            f" at line {i + 1}"
                        ),
                    )
                )
                break

        for pattern in _SENSITIVE_ENV_VARS:
            match = pattern.search(line)
            if match:
                findings.append(
                    SecurityFinding(
                        severity=Severity.ERROR,
                        category=FindingCategory.CREDENTIAL_ACCESS,
                        file=file_path,
                        line=i + 1,
                        message=(
                            f"References sensitive environment variable"
                            f" '{match.group(0)}' at line {i + 1}"
                        ),
                    )
                )
                break

        for pattern, label in _DANGEROUS_COMMANDS:
            if pattern.search(line):
                findings.append(
                    SecurityFinding(
                        severity=Severity.ERROR,
                        category=FindingCategory.CREDENTIAL_ACCESS,
                        file=file_path,
                        line=i + 1,
                        message=(
                            f"Contains dangerous command '{label}'"
                            f" at line {i + 1}"
                        ),
                    )
                )
                break

    return findings


def _discover_skill_files(submission_dir: Path) -> list[Path]:
    """Find all SKILL.md files in a submission (flat and nested layouts)."""
    skills_dir = submission_dir / "skills"
    if not skills_dir.is_dir():
        return []

    skill_files: list[Path] = []

    top_level = skills_dir / "SKILL.md"
    if top_level.is_file():
        skill_files.append(top_level)

    for child in sorted(skills_dir.iterdir()):
        if child.is_dir():
            nested = child / "SKILL.md"
            if nested.is_file():
                skill_files.append(nested)

    return skill_files


def scan_submission(submission_dir: Path) -> SecurityScanResult:
    """Scan all SKILL.md files in a submission for security issues."""
    logger.info("Security scanning submission: %s", submission_dir)
    skill_files = _discover_skill_files(submission_dir)
    all_findings: list[SecurityFinding] = []

    for skill_path in skill_files:
        content = skill_path.read_text()
        rel_path = str(skill_path.relative_to(submission_dir))
        all_findings.extend(scan_file_for_injections(rel_path, content))
        all_findings.extend(scan_file_for_credentials(rel_path, content))

    error_count = sum(
        1 for f in all_findings if f.severity == Severity.ERROR
    )
    warning_count = sum(
        1 for f in all_findings if f.severity == Severity.WARNING
    )
    files_scanned = len(skill_files)

    if error_count:
        logger.warning(
            "Security scan found %d error(s) in %d file(s)",
            error_count, files_scanned,
        )
    else:
        logger.info(
            "Security scan passed: %d file(s) scanned, %d warning(s)",
            files_scanned, warning_count,
        )

    return SecurityScanResult(
        passed=error_count == 0,
        findings=all_findings,
        summary=(
            f"{error_count} error(s), {warning_count} warning(s)"
            f" in {files_scanned} file(s) scanned"
        ),
    )
