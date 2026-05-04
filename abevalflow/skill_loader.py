"""Fetch and place the agentic-contribution-skill for AI-assisted generation.

The skill is sourced from the agentic-collections repository and placed in
the correct folder for whichever agent system is configured (Claude, Cursor,
opencode, or plain API mode).  When running in API mode the skill content
is extracted as structured quality criteria for use in system prompts.

If the fetch fails the caller gets ``None`` and should proceed without the
skill (degraded quality, logged warning).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SKILL_REPO = "https://github.com/RHEcosystemAppEng/agentic-collections.git"
DEFAULT_SKILL_PATH = ".claude/skills/agentic-contribution-skill"

_AGENT_SKILL_DIRS: dict[str, str] = {
    "claude": ".claude/skills",
    "cursor": ".cursor/rules",
    "opencode": ".opencode/skills",
}


def fetch_skill(
    target_dir: Path,
    *,
    repo_url: str | None = None,
    skill_path: str | None = None,
) -> Path | None:
    """Shallow-clone the skill repo and extract the skill into *target_dir*.

    Returns the path to the SKILL.md file, or ``None`` on any failure.
    """
    repo_url = repo_url or os.environ.get("SKILL_REPO_URL", DEFAULT_SKILL_REPO)
    skill_path = skill_path or os.environ.get("SKILL_PATH", DEFAULT_SKILL_PATH)

    logger.info("Fetching skill from %s (path: %s)", repo_url, skill_path)

    with tempfile.TemporaryDirectory(prefix="skill-fetch-") as tmp:
        try:
            subprocess.run(
                [
                    "git", "clone", "--depth", "1", "--sparse",
                    "--filter=blob:none", repo_url, tmp,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            subprocess.run(
                ["git", "sparse-checkout", "set", skill_path],
                capture_output=True,
                text=True,
                check=True,
                cwd=tmp,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("Failed to fetch skill repo: %s", exc)
            return None

        src = Path(tmp) / skill_path
        if not (src / "SKILL.md").is_file():
            logger.warning("SKILL.md not found at %s", src)
            return None

        dest = target_dir / Path(skill_path).name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)

    skill_md = dest / "SKILL.md"
    logger.info("Skill available at %s", skill_md)
    return skill_md


def place_for_agent(
    skill_dir: Path,
    workspace_dir: Path,
    agent_type: str | None = None,
) -> Path:
    """Copy the fetched skill into the agent-specific folder under *workspace_dir*.

    Returns the destination directory.
    """
    agent_type = agent_type or os.environ.get("AGENT_TYPE", "api")

    if agent_type == "api":
        return skill_dir

    agent_skills_base = _AGENT_SKILL_DIRS.get(agent_type)
    if agent_skills_base is None:
        logger.warning("Unknown agent_type '%s', keeping skill in place", agent_type)
        return skill_dir

    dest = workspace_dir / agent_skills_base / skill_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(skill_dir, dest)
    logger.info("Placed skill for %s at %s", agent_type, dest)
    return dest


def extract_quality_criteria(skill_md_path: Path) -> str:
    """Parse SKILL.md and return structured quality criteria text.

    Extracts the workflow phases and quality standards from the skill
    definition so they can be used in a system prompt when running in
    API-only mode (no native agent skill loading).
    """
    if not skill_md_path.is_file():
        return ""

    content = skill_md_path.read_text()

    # Strip YAML frontmatter
    content = re.sub(r"^---\n.*?\n---\n", "", content, count=1, flags=re.DOTALL)

    sections: list[str] = []
    headings_of_interest = [
        "Workflow",
        "When to Use This Skill",
        "Common Issues",
        "Critical: Human-in-the-Loop",
        "Security Considerations",
    ]

    for heading in headings_of_interest:
        pattern = rf"(## .*{re.escape(heading)}.*?\n)(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            sections.append(match.group(1).strip() + "\n" + match.group(2).strip())

    if not sections:
        return content[:3000]

    return "\n\n".join(sections)
