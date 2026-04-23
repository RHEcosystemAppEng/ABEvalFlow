"""Tests for abevalflow/skill_loader.py."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from abevalflow import skill_loader

SAMPLE_SKILL_MD = """\
---
name: test-skill
description: A test skill
model: inherit
color: green
---

# /test-skill Skill

A test skill for evaluation.

## When to Use This Skill

**Use when**:
- Creating test files for evaluation

**Do NOT use when**:
- Running tests directly

## Workflow

### Phase 1: Discovery
Ask the user what they need.

### Phase 2: Generation
Generate the required files.

## Common Issues

### Issue 1: Missing files
**Fix**: Check the directory structure.

## Dependencies

None required.

## Security Considerations

Never expose API keys.
"""


@pytest.fixture()
def skill_dir(tmp_path: Path) -> Path:
    """Create a mock skill directory with SKILL.md."""
    d = tmp_path / "agentic-contribution-skill"
    d.mkdir()
    (d / "SKILL.md").write_text(SAMPLE_SKILL_MD)
    docs = d / "docs"
    docs.mkdir()
    (docs / "examples.md").write_text("# Examples\n\nSome examples here.\n")
    return d


class TestFetchSkill:
    @patch("abevalflow.skill_loader.subprocess.run")
    def test_successful_fetch(
        self,
        mock_run,
        tmp_path: Path,
        skill_dir: Path,
    ) -> None:
        target = tmp_path / "target"
        target.mkdir()

        def side_effect(*args, **kwargs):
            if "clone" in args[0]:
                clone_dir = Path(args[0][-1])
                skill_dest = clone_dir / ".claude" / "skills" / "agentic-contribution-skill"
                skill_dest.mkdir(parents=True)
                shutil.copytree(skill_dir, skill_dest, dirs_exist_ok=True)
            return None

        mock_run.side_effect = side_effect

        result = skill_loader.fetch_skill(target)
        assert result is not None
        assert result.name == "SKILL.md"
        assert result.is_file()

    @patch("abevalflow.skill_loader.subprocess.run")
    def test_fetch_failure_returns_none(
        self,
        mock_run,
        tmp_path: Path,
    ) -> None:
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")

        target = tmp_path / "target"
        target.mkdir()
        result = skill_loader.fetch_skill(target)
        assert result is None

    @patch("abevalflow.skill_loader.subprocess.run")
    def test_fetch_timeout_returns_none(
        self,
        mock_run,
        tmp_path: Path,
    ) -> None:
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("git", 60)

        target = tmp_path / "target"
        target.mkdir()
        result = skill_loader.fetch_skill(target)
        assert result is None


class TestPlaceForAgent:
    def test_api_mode_returns_same_dir(self, skill_dir: Path, tmp_path: Path) -> None:
        result = skill_loader.place_for_agent(skill_dir, tmp_path, "api")
        assert result == skill_dir

    def test_claude_mode(self, skill_dir: Path, tmp_path: Path) -> None:
        result = skill_loader.place_for_agent(skill_dir, tmp_path, "claude")
        expected = tmp_path / ".claude" / "skills" / "agentic-contribution-skill"
        assert result == expected
        assert (result / "SKILL.md").is_file()

    def test_unknown_agent_returns_same_dir(self, skill_dir: Path, tmp_path: Path) -> None:
        result = skill_loader.place_for_agent(skill_dir, tmp_path, "unknown-agent")
        assert result == skill_dir


class TestExtractQualityCriteria:
    def test_extracts_sections(self, skill_dir: Path) -> None:
        criteria = skill_loader.extract_quality_criteria(skill_dir / "SKILL.md")
        assert "Workflow" in criteria
        assert "When to Use This Skill" in criteria
        assert "Security Considerations" in criteria

    def test_strips_frontmatter(self, skill_dir: Path) -> None:
        criteria = skill_loader.extract_quality_criteria(skill_dir / "SKILL.md")
        assert "---" not in criteria
        assert "name: test-skill" not in criteria

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        criteria = skill_loader.extract_quality_criteria(tmp_path / "missing.md")
        assert criteria == ""

    def test_file_without_sections_returns_truncated_content(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare.md"
        bare.write_text("Just some plain text content without any sections.\n" * 100)
        criteria = skill_loader.extract_quality_criteria(bare)
        assert len(criteria) <= 3000
        assert "Just some plain text" in criteria
