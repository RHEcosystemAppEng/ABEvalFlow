"""SKILL.md quality scanner.

Deterministic quality checks for skill submissions: description quality,
broken references, file completeness, imprecise instructions, unfinished
content, and generic advice.

Patterns adapted from harness-eval-lab (setup-eval).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_RELATIVE_LINK_RE = re.compile(r"\[.*?\]\(([^)]+)\)")

# --- Imprecise instruction patterns ---

_HEDGING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("try to", re.compile(r"\btry\s+to\s+\w+", re.I)),
    ("if possible", re.compile(r"\bif\s+(?:at\s+all\s+)?possible\b", re.I)),
    ("you might want to", re.compile(r"\byou\s+might\s+(?:want|wish)\s+to\b", re.I)),
    ("perhaps consider", re.compile(r"\bperhaps\s+(?:consider|you)\b", re.I)),
    ("consider using", re.compile(r"\bconsider\s+(?:using|adding|implementing)\b", re.I)),
]

_VAGUE_CONDITION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("if needed", re.compile(r"\bif\s+needed\b", re.I)),
    ("if appropriate", re.compile(r"\bif\s+appropriate\b", re.I)),
    ("as necessary", re.compile(r"\bas\s+necessary\b", re.I)),
    ("if applicable", re.compile(r"\bif\s+applicable\b", re.I)),
    ("when relevant", re.compile(r"\bwhen\s+relevant\b", re.I)),
]

# --- Unfinished content patterns ---

_UNFINISHED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("TODO", re.compile(r"\bTODO\s*:", re.I)),
    ("FIXME", re.compile(r"\bFIXME\s*:", re.I)),
    ("XXX", re.compile(r"\bXXX\s*:", re.I)),
    ("TBD", re.compile(r"\bTBD\b")),
    ("coming soon", re.compile(r"\bcoming\s+soon\b", re.I)),
    ("work in progress", re.compile(r"\bwork\s+in\s+progress\b", re.I)),
    ("not yet implemented", re.compile(r"\bnot\s+yet\s+(?:implemented|done|complete)\b", re.I)),
    (
        "placeholder bracket",
        re.compile(
            r"\[(?:INSERT|FILL\s+IN|CHANGE\s+THIS|UPDATE\s+THIS|REPLACE|YOUR)\s+[^\]]+\]",
            re.I,
        ),
    ),
    ("PLACEHOLDER", re.compile(r"\[PLACEHOLDER\]", re.I)),
]

# --- Generic advice patterns ---

_GENERIC_ADVICE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("follow best practices", re.compile(r"follow\s+(the\s+)?best\s+practices", re.I)),
    ("handle errors properly", re.compile(r"handle\s+errors\s+(?:properly|correctly|gracefully)", re.I)),
    ("write clean code", re.compile(r"write\s+clean[\s,]+readable\s+code", re.I)),
    ("ensure code quality", re.compile(r"ensure\s+(?:code\s+)?quality", re.I)),
    ("consider edge cases", re.compile(r"consider\s+(all\s+)?edge\s+cases", re.I)),
    ("use proper formatting", re.compile(r"use\s+proper\s+formatting", re.I)),
    ("be thorough", re.compile(r"be\s+thorough\s+(?:in|and|with)", re.I)),
]

_EXCLUDED_DIRS = {".git", "node_modules", "vendor", "__pycache__", ".venv"}


def _is_excluded(path: Path, base: Path) -> bool:
    try:
        parts = path.relative_to(base).parts
    except ValueError:
        return False
    return bool(_EXCLUDED_DIRS.intersection(parts))


def _is_in_code_fence_or_quote(line: str, in_code_fence: bool) -> bool:
    return in_code_fence or line.lstrip().startswith(">")


# --- Check functions ---


def check_description_quality(
    file_path: Path,
    relative_to: Path | None = None,
) -> list[dict]:
    """Check SKILL.md frontmatter for description quality issues."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    display_path = str(file_path.relative_to(relative_to)) if relative_to else str(file_path)
    findings: list[dict] = []

    fm_match = _FRONTMATTER_RE.match(content)
    if not fm_match:
        findings.append(
            {
                "severity": "medium",
                "rule_id": "quality-no-frontmatter",
                "message": "SKILL.md has no YAML frontmatter",
                "file_path": display_path,
                "category": "description_quality",
            }
        )
        return findings

    try:
        fm = yaml.safe_load(fm_match.group(1))
    except yaml.YAMLError:
        findings.append(
            {
                "severity": "medium",
                "rule_id": "quality-invalid-frontmatter",
                "message": "SKILL.md frontmatter is not valid YAML",
                "file_path": display_path,
                "category": "description_quality",
            }
        )
        return findings

    if not isinstance(fm, dict):
        return findings

    if "name" not in fm:
        findings.append(
            {
                "severity": "medium",
                "rule_id": "quality-missing-name",
                "message": "SKILL.md frontmatter is missing 'name' field",
                "file_path": display_path,
                "category": "description_quality",
            }
        )

    desc = fm.get("description", "")
    if not desc:
        findings.append(
            {
                "severity": "medium",
                "rule_id": "quality-missing-description",
                "message": "SKILL.md frontmatter is missing 'description' field",
                "file_path": display_path,
                "category": "description_quality",
            }
        )
    elif len(str(desc)) < 10:
        findings.append(
            {
                "severity": "medium",
                "rule_id": "quality-short-description",
                "message": (f"SKILL.md description is {len(str(desc))} characters, too short to be meaningful"),
                "file_path": display_path,
                "category": "description_quality",
            }
        )

    name = fm.get("name", "")
    if name and desc and str(name).strip().lower() == str(desc).strip().lower():
        findings.append(
            {
                "severity": "low",
                "rule_id": "quality-description-equals-name",
                "message": "SKILL.md description is identical to the name",
                "file_path": display_path,
                "category": "description_quality",
            }
        )

    return findings


def check_broken_references(
    file_path: Path,
    relative_to: Path | None = None,
) -> list[dict]:
    """Check markdown file for broken relative links."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    display_path = str(file_path.relative_to(relative_to)) if relative_to else str(file_path)
    findings: list[dict] = []
    checked: set[str] = set()
    in_code_fence = False

    for line_num, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence or stripped.startswith(">"):
            continue

        for match in _RELATIVE_LINK_RE.finditer(line):
            ref = match.group(1).strip()

            if ref.startswith(("http://", "https://", "#", "mailto:")):
                continue
            if "$" in ref or "{{" in ref:
                continue

            ref = ref.split("#")[0]
            if not ref:
                continue

            if ref in checked:
                continue
            checked.add(ref)

            ref_path = file_path.parent / ref
            if not ref_path.exists():
                findings.append(
                    {
                        "severity": "medium",
                        "rule_id": "quality-broken-reference",
                        "message": (f"Line {line_num}: link to '{ref}' points to a file that does not exist"),
                        "file_path": display_path,
                        "category": "broken_reference",
                        "line": line_num,
                    }
                )

    return findings


def check_file_completeness(directory: Path) -> list[dict]:
    """Check submission files for completeness issues."""
    findings: list[dict] = []

    instruction = directory / "instruction.md"
    if instruction.exists():
        try:
            content = instruction.read_text(encoding="utf-8", errors="replace")
            body = re.sub(r"^---.*?---\s*", "", content, flags=re.DOTALL).strip()
            body = re.sub(r"^#.*\n?", "", body).strip()
            if len(body) < 50:
                findings.append(
                    {
                        "severity": "high" if len(body) < 10 else "low",
                        "rule_id": "quality-thin-instruction",
                        "message": (f"instruction.md has only {len(body)} characters of body text"),
                        "file_path": "instruction.md",
                        "category": "file_completeness",
                    }
                )
        except OSError:
            pass

    tests_dir = directory / "tests"
    if tests_dir.is_dir():
        for py_file in tests_dir.glob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                if "assert" not in content and "pytest.raises" not in content:
                    rel = str(py_file.relative_to(directory))
                    findings.append(
                        {
                            "severity": "medium",
                            "rule_id": "quality-no-assertions",
                            "message": f"{rel} has no assert statements",
                            "file_path": rel,
                            "category": "file_completeness",
                        }
                    )
            except OSError:
                pass

    return findings


def check_imprecise_instructions(
    file_path: Path,
    relative_to: Path | None = None,
) -> list[dict]:
    """Check for vague, hedging, or ambiguous language."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    display_path = str(file_path.relative_to(relative_to)) if relative_to else str(file_path)
    findings: list[dict] = []
    in_code_fence = False

    all_patterns = [("hedging", label, pat) for label, pat in _HEDGING_PATTERNS] + [
        ("vague condition", label, pat) for label, pat in _VAGUE_CONDITION_PATTERNS
    ]

    for line_num, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if _is_in_code_fence_or_quote(line, in_code_fence):
            continue

        for category, label, pattern in all_patterns:
            if pattern.search(line):
                findings.append(
                    {
                        "severity": "low",
                        "rule_id": "quality-imprecise-instruction",
                        "message": f"Line {line_num}: '{label}' ({category})",
                        "file_path": display_path,
                        "category": "imprecise_instruction",
                        "line": line_num,
                    }
                )
                break

    return findings


def check_unfinished_content(
    file_path: Path,
    relative_to: Path | None = None,
) -> list[dict]:
    """Check for TODO, FIXME, placeholder markers, and deferred content."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    display_path = str(file_path.relative_to(relative_to)) if relative_to else str(file_path)
    findings: list[dict] = []
    in_code_fence = False

    for line_num, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if _is_in_code_fence_or_quote(line, in_code_fence):
            continue

        for label, pattern in _UNFINISHED_PATTERNS:
            if pattern.search(line):
                findings.append(
                    {
                        "severity": "medium",
                        "rule_id": "quality-unfinished-content",
                        "message": f"Line {line_num}: '{label}' marker found",
                        "file_path": display_path,
                        "category": "unfinished_content",
                        "line": line_num,
                    }
                )
                break

    return findings


def check_generic_advice(
    file_path: Path,
    relative_to: Path | None = None,
) -> list[dict]:
    """Check for generic filler text that adds no value."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    display_path = str(file_path.relative_to(relative_to)) if relative_to else str(file_path)
    findings: list[dict] = []
    in_code_fence = False

    for line_num, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if _is_in_code_fence_or_quote(line, in_code_fence):
            continue

        for label, pattern in _GENERIC_ADVICE_PATTERNS:
            if pattern.search(line):
                findings.append(
                    {
                        "severity": "low",
                        "rule_id": "quality-generic-advice",
                        "message": f"Line {line_num}: '{label}' is generic filler",
                        "file_path": display_path,
                        "category": "generic_advice",
                        "line": line_num,
                    }
                )
                break

    return findings


# --- Circular reference patterns ---

_SKILL_REF_PATTERNS = [
    re.compile(r"/(\w[\w-]+)(?:\s|$|[),\]])"),
    re.compile(r"(?:skill|command)[:\s]+[\"']?(\w[\w-]+)[\"']?", re.I),
    re.compile(
        r"(?:invokes?|calls?|triggers?|runs?)\s+[\"'`]?/?(\w[\w-]+)[\"'`]?",
        re.I,
    ),
]


def _extract_skill_references(body: str, own_name: str) -> set[str]:
    refs: set[str] = set()
    for pattern in _SKILL_REF_PATTERNS:
        for match in pattern.finditer(body):
            name = match.group(1)
            if name != own_name and len(name) > 1:
                refs.add(name)
    return refs


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    visited: set[str] = set()
    on_stack: set[str] = set()
    cycles: list[list[str]] = []
    path: list[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        on_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in graph:
                continue
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in on_stack:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        path.pop()
        on_stack.discard(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return cycles


def check_circular_references(directory: Path) -> list[dict]:
    """Detect circular reference chains between skills in a submission."""
    skills_dir = directory / "skills"
    if not skills_dir.is_dir():
        return []

    graph: dict[str, set[str]] = {}
    file_map: dict[str, str] = {}
    known_names: set[str] = set()

    for skill_md in skills_dir.rglob("SKILL.md"):
        skill_name = skill_md.parent.name
        if skill_name == "skills":
            skill_name = "root"
        try:
            content = skill_md.read_text(encoding="utf-8", errors="replace")
            fm_match = _FRONTMATTER_RE.match(content)
            if fm_match:
                try:
                    fm = yaml.safe_load(fm_match.group(1))
                    if isinstance(fm, dict) and "name" in fm:
                        skill_name = fm["name"]
                except yaml.YAMLError:
                    pass
            known_names.add(skill_name)
            body = re.sub(r"^---.*?---\s*", "", content, flags=re.DOTALL)
            refs = _extract_skill_references(body, skill_name)
            graph[skill_name] = refs
            file_map[skill_name] = str(skill_md.relative_to(directory))
        except OSError:
            continue

    for name in graph:
        graph[name] = graph[name] & known_names

    if len(graph) < 2:
        return []

    findings: list[dict] = []
    seen_cycles: set[str] = set()

    for cycle in _find_cycles(graph):
        key = " -> ".join(sorted(set(cycle[:-1])))
        if key in seen_cycles:
            continue
        seen_cycles.add(key)

        chain = " -> ".join(cycle)
        first = cycle[0]
        findings.append(
            {
                "severity": "medium",
                "rule_id": "quality-circular-reference",
                "message": f"Circular reference detected: {chain}",
                "file_path": file_map.get(first, ""),
                "category": "circular_reference",
            }
        )

    return findings


# --- Main entry point ---


def scan_directory(directory: Path) -> dict:
    """Run all deterministic quality checks on a submission directory."""
    if not directory.is_dir():
        logger.error("Not a directory: %s", directory)
        return {"findings": []}

    all_findings: list[dict] = []

    md_files = sorted(f for f in directory.rglob("*.md") if not _is_excluded(f, directory))

    for md_file in md_files:
        if md_file.name == "SKILL.md":
            all_findings.extend(check_description_quality(md_file, relative_to=directory))
        all_findings.extend(check_broken_references(md_file, relative_to=directory))
        all_findings.extend(check_imprecise_instructions(md_file, relative_to=directory))
        all_findings.extend(check_unfinished_content(md_file, relative_to=directory))
        all_findings.extend(check_generic_advice(md_file, relative_to=directory))

    all_findings.extend(check_file_completeness(directory))
    all_findings.extend(check_circular_references(directory))

    logger.info(
        "Quality scan: %d files, %d findings",
        len(md_files),
        len(all_findings),
    )
    return {"findings": all_findings}
