"""Security scanner for AI skill submissions.

Scans SKILL.md files for security risks across six categories:

  1. Prompt injection (20 patterns, WARNING only, non-blocking)
  2. Credential access (sensitive paths, env vars, dangerous commands)
  3. Data exfiltration (8 patterns, context-aware severity)
  4. Reverse shells (10 patterns, context-aware severity)
  5. Code obfuscation (6 patterns, context-aware severity)
  6. Hidden content (zero-width chars, RTL overrides, homoglyphs)

All checks are deterministic regex matching with no LLM calls.
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
    DATA_EXFILTRATION = "data_exfiltration"
    REVERSE_SHELL = "reverse_shell"
    OBFUSCATION = "obfuscation"
    HIDDEN_CONTENT = "hidden_content"


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
    ("ignore previous instructions", re.compile(r"ignore\s+(all\s+)?previous\s+instructions", _I)),
    ("disregard prior", re.compile(r"disregard\s+(all\s+)?(prior|previous|above)", _I)),
    ("you are now", re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+", _I)),
    ("system prompt override", re.compile(r"system\s*prompt\s*(override|injection|change)", _I)),
    (
        "override instructions",
        re.compile(r"override\s+(all\s+)?(instructions|rules|guidelines)", _I),
    ),
    ("new instructions", re.compile(r"new\s+instructions?\s*:", _I)),
    ("jailbreak attempt", re.compile(r"(\bDAN\b|do\s+anything\s+now|developer\s+mode)", _I)),
    (
        "prompt leak",
        re.compile(r"(reveal|show|print|output)\s+(your|the)\s+(system\s+)?prompt", _I),
    ),
    ("role hijack", re.compile(r"forget\s+(everything|all|your)\s+(you|instructions|rules)", _I)),
    ("hidden instruction", re.compile(r"<\s*(?:system|instruction|hidden)\s*>", _I)),
    ("role play", re.compile(r"pretend\s+(?:to\s+be|you\s+are)\s+(?:a|an|the)\s+", _I)),
    (
        "encoding evasion",
        re.compile(r"(?:in\s+base64|encode\s+(?:as|in|to)\s+base64|base64\s+encod)", _I),
    ),
    ("repeat after me", re.compile(r"repeat\s+after\s+me", _I)),
    (
        "bypass safety",
        re.compile(r"(?:ignore\s+safety|bypass\s+(?:filter|safety|restriction))", _I),
    ),
    ("output control", re.compile(r"output\s+the\s+following\s+exactly", _I)),
    ("markdown image exfiltration", re.compile(r"!\[.*?\]\(https?://", _I)),
    ("translate evasion", re.compile(r"translate\s+(?:this|the\s+following)\s+(?:to|into)\s+", _I)),
    ("act as", re.compile(r"act\s+as\s+(?:a|an|the|if)\s+", _I)),
    ("simulate mode", re.compile(r"(?:enter|enable|activate)\s+(?:\w+\s+)?mode", _I)),
    ("data exfiltration via url", re.compile(r"(?:curl|wget|fetch)\s+https?://", _I)),
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

# Data exfiltration patterns

_EXFIL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("curl post file contents", re.compile(r"curl\s+.*-d\s+\"\$\(cat\b", _I)),
    ("curl with command substitution", re.compile(r"curl\s+.*--data.*\$\(", _I)),
    ("wget post data", re.compile(r"wget\s+--post-data", _I)),
    ("dns tunneling dig", re.compile(r"\bdig\s+.*\bTXT\b", _I)),
    ("dns tunneling nslookup", re.compile(r"\bnslookup\s+.*-type=TXT", _I)),
    (
        "webhook exfiltration",
        re.compile(
            r"(?:curl|wget|fetch)\s+.*"
            r"(?:webhook|hooks\.|pipedream|requestbin|ngrok)",
            _I,
        ),
    ),
    ("base64 pipe to network", re.compile(r"base64\s+.*\|\s*(?:curl|wget|nc)\b", _I)),
    ("archive pipe to network", re.compile(r"tar\s+.*\|\s*(?:curl|wget|nc|ssh)\b", _I)),
]

# Reverse shell patterns

_REVERSE_SHELL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("bash reverse shell", re.compile(r"bash\s+-i\s+>&\s*/dev/tcp/", _I)),
    ("netcat exec", re.compile(r"\bnc\s+.*-e\s+/bin/", _I)),
    ("ncat exec", re.compile(r"\bncat\s+.*--exec", _I)),
    ("python socket shell", re.compile(r"python[23]?\s+-c\s+.*(?:socket|subprocess)", _I)),
    ("perl socket shell", re.compile(r"perl\s+-e\s+.*(?:socket|Socket)", _I)),
    ("ruby socket shell", re.compile(r"ruby\s+-rsocket\s+-e", _I)),
    ("php socket shell", re.compile(r"php\s+-r\s+.*fsockopen", _I)),
    ("socat exec", re.compile(r"\bsocat\s+.*exec:", _I)),
    ("named pipe shell", re.compile(r"\bmknod\s+.*\bp\b.*(?:/bin/sh|bash)", _I)),
    ("powershell reverse shell", re.compile(r"\bpowershell\s+.*(?:Net\.Sockets|TCPClient)", _I)),
]

# Code obfuscation patterns

_OBFUSCATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "eval with decode",
        re.compile(r"eval\s*\(\s*(?:atob|Buffer\.from|base64\.b64decode)\s*\(", _I),
    ),
    ("char code construction", re.compile(r"String\.fromCharCode\s*\(", _I)),
    ("hex escape sequence", re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}")),
    ("unicode escape sequence", re.compile(r"(?:\\u[0-9a-fA-F]{4}){4,}")),
    ("python dynamic exec", re.compile(r"exec\s*\(\s*(?:compile|__import__)\s*\(", _I)),
    ("char code round-trip", re.compile(r"charCodeAt\b.*\bfromCharCode\b", _I)),
]

# Hidden content patterns (unicode deception)

_ZERO_WIDTH_CHARS: dict[str, str] = {
    "​": "zero-width space",
    "‌": "zero-width non-joiner",
    "‍": "zero-width joiner",
    "⁠": "word joiner",
    "﻿": "BOM / zero-width no-break space",
    "­": "soft hyphen",
}

_RTL_OVERRIDE_CHARS: dict[str, str] = {
    "‪": "LRE",
    "‫": "RLE",
    "‬": "PDF",
    "‭": "LRO",
    "‮": "RLO",
    "⁦": "LRI",
    "⁧": "RLI",
    "⁨": "FSI",
    "⁩": "PDI",
}

_HOMOGLYPH_MAP: dict[str, str] = {
    "А": "A (Cyrillic)",
    "В": "B (Cyrillic)",
    "С": "C (Cyrillic)",
    "Е": "E (Cyrillic)",
    "Н": "H (Cyrillic)",
    "К": "K (Cyrillic)",
    "М": "M (Cyrillic)",
    "О": "O (Cyrillic)",
    "Р": "P (Cyrillic)",
    "Т": "T (Cyrillic)",
    "Х": "X (Cyrillic)",
    "а": "a (Cyrillic)",
    "е": "e (Cyrillic)",
    "о": "o (Cyrillic)",
    "р": "p (Cyrillic)",
    "с": "c (Cyrillic)",
    "у": "y (Cyrillic)",
    "х": "x (Cyrillic)",
    "Α": "A (Greek)",
    "Β": "B (Greek)",
    "Ε": "E (Greek)",
    "Η": "H (Greek)",
    "Κ": "K (Greek)",
    "Μ": "M (Greek)",
    "Ο": "O (Greek)",
    "Ρ": "P (Greek)",
    "Τ": "T (Greek)",
    "Χ": "X (Greek)",
    "ο": "o (Greek)",
}


def scan_file_for_injections(
    file_path: str,
    content: str,
) -> list[SecurityFinding]:
    """Scan file content for prompt injection patterns.

    All findings are WARNING severity (non-blocking) because some
    patterns are broad enough to match legitimate skill instructions.
    Context is noted in the message for reviewer awareness.
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
                is_quoted = stripped.startswith(">") or stripped.startswith('"')
                is_example = any(w in line.lower() for w in _EXAMPLE_MARKERS)

                if in_code_fence:
                    msg = (
                        f"Line {i + 1} contains '{label}' inside a"
                        " code block - likely safe"
                        " (documentation or example)."
                    )
                elif is_quoted or is_example:
                    msg = f"Line {i + 1} contains '{label}' in a quote or example - likely safe."
                else:
                    msg = (
                        f"Line {i + 1} contains a word pattern"
                        f" ('{label}') that could indicate prompt"
                        " injection. Review whether this is"
                        " intentional content or an actual risk."
                    )

                logger.debug(
                    "WARNING: %s in %s (line %d)",
                    label,
                    file_path,
                    i + 1,
                )
                findings.append(
                    SecurityFinding(
                        severity=Severity.WARNING,
                        category=FindingCategory.PROMPT_INJECTION,
                        file=file_path,
                        line=i + 1,
                        message=msg,
                    )
                )
                break

    return findings


def scan_file_for_credentials(
    file_path: str,
    content: str,
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
                        message=(f"References sensitive path '{match.group(0)}' at line {i + 1}"),
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
                        message=(f"Contains dangerous command '{label}' at line {i + 1}"),
                    )
                )
                break

    return findings


def scan_file_for_exfiltration(
    file_path: str,
    content: str,
) -> list[SecurityFinding]:
    """Scan file content for data exfiltration patterns.

    Context-aware: findings inside code fences are reported as WARNING
    instead of ERROR.
    """
    findings: list[SecurityFinding] = []
    lines = content.split("\n")
    in_code_fence = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        for label, pattern in _EXFIL_PATTERNS:
            if pattern.search(line):
                if in_code_fence:
                    severity = Severity.WARNING
                    msg = (
                        f"Line {i + 1} contains '{label}' inside a"
                        " code block - likely documentation."
                    )
                else:
                    severity = Severity.ERROR
                    msg = (
                        f"Line {i + 1} contains a data exfiltration"
                        f" pattern ('{label}'). This is a critical"
                        " security risk."
                    )

                findings.append(
                    SecurityFinding(
                        severity=severity,
                        category=FindingCategory.DATA_EXFILTRATION,
                        file=file_path,
                        line=i + 1,
                        message=msg,
                    )
                )
                break

    return findings


def scan_file_for_reverse_shells(
    file_path: str,
    content: str,
) -> list[SecurityFinding]:
    """Scan file content for reverse shell patterns.

    Context-aware: findings inside code fences are reported as WARNING
    instead of ERROR.
    """
    findings: list[SecurityFinding] = []
    lines = content.split("\n")
    in_code_fence = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        for label, pattern in _REVERSE_SHELL_PATTERNS:
            if pattern.search(line):
                if in_code_fence:
                    severity = Severity.WARNING
                    msg = (
                        f"Line {i + 1} contains '{label}' inside a"
                        " code block - likely documentation."
                    )
                else:
                    severity = Severity.ERROR
                    msg = (
                        f"Line {i + 1} contains a reverse shell"
                        f" pattern ('{label}'). This is a critical"
                        " security risk."
                    )

                findings.append(
                    SecurityFinding(
                        severity=severity,
                        category=FindingCategory.REVERSE_SHELL,
                        file=file_path,
                        line=i + 1,
                        message=msg,
                    )
                )
                break

    return findings


def scan_file_for_obfuscation(
    file_path: str,
    content: str,
) -> list[SecurityFinding]:
    """Scan file content for code obfuscation patterns.

    Context-aware: findings inside code fences are reported as WARNING
    instead of ERROR.
    """
    findings: list[SecurityFinding] = []
    lines = content.split("\n")
    in_code_fence = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        for label, pattern in _OBFUSCATION_PATTERNS:
            if pattern.search(line):
                if in_code_fence:
                    severity = Severity.WARNING
                    msg = (
                        f"Line {i + 1} contains '{label}' inside a"
                        " code block - likely documentation."
                    )
                else:
                    severity = Severity.ERROR
                    msg = (
                        f"Line {i + 1} contains an obfuscation"
                        f" pattern ('{label}'). This may hide"
                        " malicious behavior."
                    )

                findings.append(
                    SecurityFinding(
                        severity=severity,
                        category=FindingCategory.OBFUSCATION,
                        file=file_path,
                        line=i + 1,
                        message=msg,
                    )
                )
                break

    return findings


def scan_file_for_hidden_content(
    file_path: str,
    content: str,
) -> list[SecurityFinding]:
    """Scan file content for unicode deception.

    Detects invisible unicode characters (zero-width, RTL overrides)
    and homoglyph characters that could disguise content.
    """
    findings: list[SecurityFinding] = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        for char, char_name in _ZERO_WIDTH_CHARS.items():
            if char in line:
                findings.append(
                    SecurityFinding(
                        severity=Severity.ERROR,
                        category=FindingCategory.HIDDEN_CONTENT,
                        file=file_path,
                        line=i + 1,
                        message=(
                            f"Line {i + 1} contains {char_name}"
                            f" (U+{ord(char):04X}). Invisible"
                            " characters can hide malicious content."
                        ),
                    )
                )
                break

        for char, char_name in _RTL_OVERRIDE_CHARS.items():
            if char in line:
                findings.append(
                    SecurityFinding(
                        severity=Severity.ERROR,
                        category=FindingCategory.HIDDEN_CONTENT,
                        file=file_path,
                        line=i + 1,
                        message=(
                            f"Line {i + 1} contains RTL override"
                            f" ({char_name}, U+{ord(char):04X})."
                            " Text direction overrides can disguise"
                            " malicious content."
                        ),
                    )
                )
                break

        for char, char_name in _HOMOGLYPH_MAP.items():
            if char in line:
                findings.append(
                    SecurityFinding(
                        severity=Severity.WARNING,
                        category=FindingCategory.HIDDEN_CONTENT,
                        file=file_path,
                        line=i + 1,
                        message=(
                            f"Line {i + 1} contains a homoglyph"
                            f" that looks like {char_name}"
                            f" (U+{ord(char):04X}). Homoglyphs"
                            " can disguise malicious content"
                            " as benign."
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
        all_findings.extend(scan_file_for_exfiltration(rel_path, content))
        all_findings.extend(scan_file_for_reverse_shells(rel_path, content))
        all_findings.extend(scan_file_for_obfuscation(rel_path, content))
        all_findings.extend(scan_file_for_hidden_content(rel_path, content))

    error_count = sum(1 for f in all_findings if f.severity == Severity.ERROR)
    warning_count = sum(1 for f in all_findings if f.severity == Severity.WARNING)
    files_scanned = len(skill_files)

    if error_count:
        logger.warning(
            "Security scan found %d error(s) in %d file(s)",
            error_count,
            files_scanned,
        )
    else:
        logger.info(
            "Security scan passed: %d file(s) scanned, %d warning(s)",
            files_scanned,
            warning_count,
        )

    return SecurityScanResult(
        passed=error_count == 0,
        findings=all_findings,
        summary=(
            f"{error_count} error(s), {warning_count} warning(s) in {files_scanned} file(s) scanned"
        ),
    )
