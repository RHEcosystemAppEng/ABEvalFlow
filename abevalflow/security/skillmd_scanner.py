"""SKILL.md security scanner.

Scans markdown files in skill submissions for security risks including
prompt injection, credential access, data exfiltration, reverse shells,
and obfuscation patterns.

Patterns ported from harness-eval-lab (setup-eval).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Prompt injection patterns (17) ---

PROMPT_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore previous instructions",
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    ),
    ("disregard prior", re.compile(r"disregard\s+(all\s+)?(prior|previous|above)", re.I)),
    ("you are now", re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+", re.I)),
    ("system prompt override", re.compile(r"system\s*prompt\s*(override|injection|change)", re.I)),
    (
        "override instructions",
        re.compile(r"override\s+(all\s+)?(instructions|rules|guidelines)", re.I),
    ),
    ("new instructions", re.compile(r"new\s+instructions?\s*:", re.I)),
    ("jailbreak attempt", re.compile(r"(do\s+anything\s+now|developer\s+mode)", re.I)),
    (
        "prompt leak",
        re.compile(r"(reveal|show|print|output)\s+(your|the)\s+(system\s+)?prompt", re.I),
    ),
    ("role hijack", re.compile(r"forget\s+(everything|all|your)\s+(you|instructions|rules)", re.I)),
    ("hidden instruction", re.compile(r"<\s*(?:system|instruction|hidden)\s*>", re.I)),
    ("role play", re.compile(r"pretend\s+(?:to\s+be|you\s+are)\s+(?:a|an|the)\s+", re.I)),
    (
        "encoding evasion",
        re.compile(r"(?:in\s+base64|encode\s+(?:as|in|to)\s+base64|base64\s+encod)", re.I),
    ),
    ("repeat after me", re.compile(r"repeat\s+after\s+me", re.I)),
    (
        "bypass safety",
        re.compile(r"(?:ignore\s+safety|bypass\s+(?:filter|safety|restriction))", re.I),
    ),
    ("output control", re.compile(r"output\s+the\s+following\s+exactly", re.I)),
    ("markdown image exfiltration", re.compile(r"!\[.*?\]\(https?://", re.I)),
    (
        "translate evasion",
        re.compile(r"translate\s+(?:this|the\s+following)\s+(?:to|into)\s+", re.I),
    ),
]

# --- Sensitive path patterns (10) ---

SENSITIVE_PATH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("~/.ssh/", re.compile(r"~/\.ssh/", re.I)),
    ("~/.aws/credentials", re.compile(r"~/\.aws/credentials", re.I)),
    ("~/.config/gcloud", re.compile(r"~/\.config/gcloud", re.I)),
    ("~/.kube/config", re.compile(r"~/\.kube/config", re.I)),
    ("/etc/shadow", re.compile(r"/etc/shadow", re.I)),
    ("~/.netrc", re.compile(r"~/\.netrc", re.I)),
    ("~/.env", re.compile(r"~/\.env\b")),
    ("~/.docker/config.json", re.compile(r"~/\.docker/config\.json", re.I)),
    ("~/.npmrc", re.compile(r"~/\.npmrc\b")),
    ("~/.pypirc", re.compile(r"~/\.pypirc\b")),
]

# --- Sensitive environment variable patterns (9) ---

SENSITIVE_ENV_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API key (AI provider)", re.compile(r"\$(?:ANTHROPIC|OPENAI|GEMINI|GOOGLE)_API_KEY")),
    ("AWS secret", re.compile(r"\$(?:AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)")),
    ("database credential", re.compile(r"\$(?:DATABASE_URL|DB_PASSWORD)")),
    ("GitHub token", re.compile(r"\$(?:GITHUB_TOKEN|GH_TOKEN)")),
    ("secret/private key", re.compile(r"\$(?:SECRET_KEY|PRIVATE_KEY)")),
    ("Slack token", re.compile(r"\$SLACK_TOKEN")),
    ("Stripe secret", re.compile(r"\$STRIPE_SECRET_KEY")),
    ("JWT secret", re.compile(r"\$JWT_SECRET")),
    ("encryption key", re.compile(r"\$ENCRYPTION_KEY")),
]

# --- Dangerous command patterns (3) ---

DANGEROUS_COMMAND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("sudo", re.compile(r"\bsudo\s+")),
    ("chmod 777", re.compile(r"\bchmod\s+777\b")),
    ("chown root", re.compile(r"\bchown\s+root\b")),
]

# --- Data exfiltration patterns (8) ---

DATA_EXFIL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("curl post file contents", re.compile(r"curl\s+.*-d\s+\"\$\(cat\b", re.I)),
    ("curl with command substitution", re.compile(r"curl\s+.*--data.*\$\(", re.I)),
    ("wget post data", re.compile(r"wget\s+--post-data", re.I)),
    ("dns tunneling dig", re.compile(r"\bdig\s+.*\bTXT\b", re.I)),
    ("dns tunneling nslookup", re.compile(r"\bnslookup\s+.*-type=TXT", re.I)),
    (
        "webhook exfiltration",
        re.compile(
            r"(?:curl|wget|fetch)\s+.*(?:webhook|hooks\.|pipedream|requestbin|ngrok)",
            re.I,
        ),
    ),
    ("base64 pipe to network", re.compile(r"base64\s+.*\|\s*(?:curl|wget|nc)\b", re.I)),
    ("archive pipe to network", re.compile(r"tar\s+.*\|\s*(?:curl|wget|nc|ssh)\b", re.I)),
]

# --- Reverse shell patterns (10) ---

REVERSE_SHELL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("bash reverse shell", re.compile(r"bash\s+-i\s+>&\s*/dev/tcp/", re.I)),
    ("netcat exec", re.compile(r"\bnc\s+.*-e\s+/bin/", re.I)),
    ("ncat exec", re.compile(r"\bncat\s+.*--exec", re.I)),
    ("python socket shell", re.compile(r"python[23]?\s+-c\s+.*(?:socket|subprocess)", re.I)),
    ("perl socket shell", re.compile(r"perl\s+-e\s+.*(?:socket|Socket)", re.I)),
    ("ruby socket shell", re.compile(r"ruby\s+-rsocket\s+-e", re.I)),
    ("php socket shell", re.compile(r"php\s+-r\s+.*fsockopen", re.I)),
    ("socat exec", re.compile(r"\bsocat\s+.*exec:", re.I)),
    ("named pipe shell", re.compile(r"\bmknod\s+.*\bp\b.*(?:/bin/sh|bash)", re.I)),
    ("powershell reverse shell", re.compile(r"\bpowershell\s+.*(?:Net\.Sockets|TCPClient)", re.I)),
]

# --- Obfuscation patterns (8) ---

OBFUSCATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "eval with decode",
        re.compile(r"eval\s*\(\s*(?:atob|Buffer\.from|base64\.b64decode)\s*\(", re.I),
    ),
    ("char code construction", re.compile(r"String\.fromCharCode\s*\(", re.I)),
    ("hex escape sequence", re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}")),
    ("unicode escape sequence", re.compile(r"(?:\\u[0-9a-fA-F]{4}){4,}")),
    ("zero-width characters", re.compile(r"[​-‏﻿]")),
    ("tag characters", re.compile(r"[\U000e0000-\U000e007f]")),
    ("python dynamic exec", re.compile(r"exec\s*\(\s*(?:compile|__import__)\s*\(", re.I)),
    ("char code round-trip", re.compile(r"charCodeAt\b.*\bfromCharCode\b", re.I)),
]

# Example/quote context indicators
_EXAMPLE_RE = re.compile(r"(?:for\s+example|e\.g\.|such\s+as|like:)", re.I)


def _is_in_example_context(line: str) -> bool:
    """Check if a line is inside a quote or example context."""
    stripped = line.lstrip()
    if stripped.startswith(">") or stripped.startswith('"'):
        return True
    return bool(_EXAMPLE_RE.search(line))


def _make_rule_id(category: str, label: str) -> str:
    """Create a rule ID from category and label."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return f"{category}-{slug}"


def scan_file(file_path: Path, relative_to: Path | None = None) -> list[dict]:
    """Scan a single file for security issues.

    Args:
        file_path: Absolute path to the file to scan.
        relative_to: If provided, file_path in findings is relative to this.

    Returns:
        List of finding dicts.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Cannot read %s: %s", file_path, e)
        return []

    display_path = str(file_path.relative_to(relative_to)) if relative_to else str(file_path)
    lines = content.splitlines()
    findings: list[dict] = []
    in_code_fence = False

    all_pattern_groups: list[tuple[str, str, list[tuple[str, re.Pattern[str]]]]] = [
        ("prompt_injection", "high", PROMPT_INJECTION_PATTERNS),
        ("sensitive_path", "high", SENSITIVE_PATH_PATTERNS),
        ("sensitive_env", "high", SENSITIVE_ENV_PATTERNS),
        ("dangerous_command", "high", DANGEROUS_COMMAND_PATTERNS),
        ("data_exfiltration", "critical", DATA_EXFIL_PATTERNS),
        ("reverse_shell", "critical", REVERSE_SHELL_PATTERNS),
        ("obfuscation", "high", OBFUSCATION_PATTERNS),
    ]

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        is_example = _is_in_example_context(line)

        for category, base_severity, patterns in all_pattern_groups:
            for label, pattern in patterns:
                if pattern.search(line):
                    if in_code_fence or is_example:
                        severity = "low"
                    else:
                        severity = base_severity

                    findings.append(
                        {
                            "severity": severity,
                            "rule_id": _make_rule_id(category, label),
                            "message": (
                                f"Line {line_num}: {category.replace('_', ' ')} pattern '{label}'"
                            ),
                            "file_path": display_path,
                            "category": category,
                            "line": line_num,
                        }
                    )

    return findings


def scan_directory(directory: Path) -> dict:
    """Scan all markdown files in a directory for security issues.

    Args:
        directory: Path to the submission directory.

    Returns:
        Dict with "findings" key containing all findings across all files.
    """
    if not directory.is_dir():
        logger.error("Not a directory: %s", directory)
        return {"findings": []}

    md_files = sorted(directory.rglob("*.md"))
    if not md_files:
        logger.info("No markdown files found in %s", directory)
        return {"findings": []}

    all_findings: list[dict] = []
    for md_file in md_files:
        file_findings = scan_file(md_file, relative_to=directory)
        all_findings.extend(file_findings)

    logger.info(
        "Scanned %d files in %s, found %d findings",
        len(md_files),
        directory,
        len(all_findings),
    )
    return {"findings": all_findings}
