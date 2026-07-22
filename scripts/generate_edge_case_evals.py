"""Generate ASE-compatible edge case evals from SKILL.md or edge_cases/*.md files.

Two modes:
1. LLM-generated (--generate-from-skill): Reads SKILL.md, uses LLM to generate
   skill-specific edge case scenarios with tailored assertions.
2. File-based (default): Converts edge_cases/*.md files into eval entries with
   default behavioral assertions.

Usage::

    # LLM-generated from SKILL.md
    python scripts/generate_edge_case_evals.py \\
        --submission-dir /workspace/submissions/my-submission \\
        --output-dir /workspace/reports/my-submission \\
        --generate-from-skill

    # File-based from edge_cases/*.md
    python scripts/generate_edge_case_evals.py \\
        --submission-dir /workspace/submissions/my-submission \\
        --output-dir /workspace/reports/my-submission
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_ASSERTIONS = [
    "The response handles the edge case appropriately without crashing or producing nonsensical output",
    "The response acknowledges limitations or uncertainty when the input is unusual",
    "The response remains helpful and coherent despite the edge case conditions",
]

EDGE_CASE_SYSTEM_PROMPT = """\
You are an expert at creating edge case evaluation tests for AI agent skills.

Your job is to generate 2-3 edge case scenarios that test unusual situations \
within a skill's domain. Read the skill definition carefully and create \
scenarios that test whether the skill's instructions cover edge conditions \
a user might realistically encounter."""

EDGE_CASE_PROMPT = """\
Read the skill definition below and generate 2-3 edge case scenarios. Each \
scenario should test an unusual but realistic situation within the skill's \
domain.

## Requirements

- Each scenario should be specific to what this skill does
- Assertions must be verifiable claims an LLM judge can evaluate
- 2-3 assertions per scenario maximum

## Output Format

Return a valid JSON object:
{{
  "skill_name": "{skill_name}",
  "evals": [
    {{
      "id": "edge-<descriptive-name>",
      "name": "Edge case: <short description>",
      "prompt": "<the edge case prompt to send to the agent>",
      "assertions": [
        "<specific claim about how the response should handle this>",
        "<another specific claim>"
      ]
    }}
  ]
}}

## Skill Content (SKILL.md)
{skill_content}

Output ONLY valid JSON — no markdown fences, no commentary."""


def _find_skill_content(submission_dir: Path) -> tuple[str, str]:
    """Find and read SKILL.md content. Returns (skill_name, content)."""
    skill_md = None
    if (submission_dir / "skills" / "SKILL.md").is_file():
        skill_md = submission_dir / "skills" / "SKILL.md"
    else:
        skills_dir = submission_dir / "skills"
        if skills_dir.is_dir():
            for child in sorted(skills_dir.iterdir()):
                if child.is_dir() and (child / "SKILL.md").is_file():
                    skill_md = child / "SKILL.md"
                    break

    if skill_md is None:
        raise FileNotFoundError("No SKILL.md found in submission")

    content = skill_md.read_text()

    skill_name = "unknown-skill"
    metadata_path = submission_dir / "metadata.yaml"
    if metadata_path.exists():
        with metadata_path.open() as f:
            meta = yaml.safe_load(f) or {}
        skill_name = meta.get("name", skill_name)

    return skill_name, content


def generate_edge_case_evals_from_skill(
    submission_dir: Path,
    max_retries: int = 3,
) -> dict | None:
    """Generate skill-specific edge case evals by analyzing SKILL.md via LLM.

    Returns dict with evals.json structure, or None on failure.
    """
    from abevalflow import llm_client

    skill_name, skill_content = _find_skill_content(submission_dir)
    logger.info("Generating edge case evals from SKILL.md for: %s", skill_name)

    for attempt in range(1, max_retries + 1):
        logger.info("Edge case generation attempt %d/%d", attempt, max_retries)

        raw = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": EDGE_CASE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": EDGE_CASE_PROMPT.format(
                        skill_name=skill_name,
                        skill_content=skill_content,
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=4096,
        )

        text = raw.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence) :]
                break
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            evals_data = json.loads(text)
        except json.JSONDecodeError:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                evals_data = json.loads(text[start:end])
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("Attempt %d: JSON parse failed: %s", attempt, exc)
                continue

        evals_list = evals_data.get("evals", [])
        if not evals_list:
            logger.warning("Attempt %d: Empty evals array", attempt)
            continue
        if len(evals_list) > 3:
            evals_list = evals_list[:3]
            evals_data["evals"] = evals_list

        valid = True
        for ev in evals_list:
            if not ev.get("prompt") or not ev.get("assertions"):
                logger.warning("Attempt %d: eval missing prompt or assertions", attempt)
                valid = False
                break
            if not ev.get("id", "").startswith("edge-"):
                ev["id"] = f"edge-{ev.get('id', 'unknown')}"
            if len(ev.get("assertions", [])) > 3:
                ev["assertions"] = ev["assertions"][:3]

        if valid:
            evals_data["skill_name"] = skill_name
            logger.info("Generated %d edge case evals", len(evals_list))
            return evals_data

    logger.error("Failed to generate edge case evals after %d attempts", max_retries)
    return None


def generate_edge_case_evals(
    submission_dir: Path,
    skill_name: str | None = None,
) -> dict | None:
    """Convert edge_cases/*.md files into an ASE evals.json structure."""
    edge_cases_dir = submission_dir / "edge_cases"
    if not edge_cases_dir.is_dir():
        return None

    md_files = sorted(edge_cases_dir.glob("*.md"))
    if not md_files:
        return None

    if skill_name is None:
        metadata_path = submission_dir / "metadata.yaml"
        if metadata_path.exists():
            with metadata_path.open() as f:
                meta = yaml.safe_load(f) or {}
            skill_name = meta.get("name", "unknown-skill")
        else:
            skill_name = "unknown-skill"

    evals = []
    for md_file in md_files:
        edge_name = md_file.stem
        prompt = md_file.read_text().strip()
        if not prompt:
            logger.warning("Skipping empty edge case file: %s", md_file.name)
            continue

        evals.append(
            {
                "id": f"edge-{edge_name}",
                "name": f"Edge case: {edge_name.replace('_', ' ')}",
                "prompt": prompt,
                "assertions": list(DEFAULT_ASSERTIONS),
            }
        )

    if not evals:
        return None

    return {
        "skill_name": skill_name,
        "evals": evals,
    }


def generate_single_edge_case_eval(
    md_file: Path,
    skill_name: str = "unknown-skill",
) -> dict | None:
    """Generate an ASE evals.json for a single edge case .md file."""
    prompt = md_file.read_text().strip()
    if not prompt:
        return None

    edge_name = md_file.stem
    return {
        "skill_name": skill_name,
        "evals": [
            {
                "id": f"edge-{edge_name}",
                "name": f"Edge case: {edge_name.replace('_', ' ')}",
                "prompt": prompt,
                "assertions": list(DEFAULT_ASSERTIONS),
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate ASE edge case evals from SKILL.md or edge_cases/*.md",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        required=True,
        help="Path to the submission directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write edge-case-evals.json",
    )
    parser.add_argument(
        "--skill-name",
        default=None,
        help="Skill name (default: from metadata.yaml)",
    )
    parser.add_argument(
        "--edge-case-file",
        type=Path,
        default=None,
        help="Generate evals for a single .md file instead of all edge_cases/",
    )
    parser.add_argument(
        "--generate-from-skill",
        action="store_true",
        help="Use LLM to generate skill-specific edge cases from SKILL.md",
    )

    args = parser.parse_args(argv)

    if not args.submission_dir.is_dir():
        logger.error("Submission directory does not exist: %s", args.submission_dir)
        return 1

    if args.generate_from_skill:
        result = generate_edge_case_evals_from_skill(args.submission_dir)
    elif args.edge_case_file:
        skill_name = args.skill_name or "unknown-skill"
        metadata_path = args.submission_dir / "metadata.yaml"
        if metadata_path.exists() and not args.skill_name:
            with metadata_path.open() as f:
                meta = yaml.safe_load(f) or {}
            skill_name = meta.get("name", skill_name)
        result = generate_single_edge_case_eval(args.edge_case_file, skill_name)
    else:
        result = generate_edge_case_evals(args.submission_dir, args.skill_name)

    if result is None:
        logger.info("No edge cases generated")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "edge-case-evals.json"
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %d edge case evals to %s", len(result["evals"]), output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
