"""Generate instruction.md and test_outputs.py from a skill definition.

When ``generation_mode: ai`` is set in metadata.yaml the submitter provides
only ``skills/SKILL.md``.  This script uses an LLM (via the OpenAI-compatible
client) to produce the remaining files the pipeline requires:

  - ``instruction.md``  — task description for the agent
  - ``tests/test_outputs.py`` — pytest verification of agent output
  - ``tests/llm_judge.py`` (optional) — LLM-based evaluation

In *api* mode the LLM is called directly with a system prompt that
incorporates quality criteria extracted from the agentic-contribution-skill.

In *agent* mode (claude, cursor, opencode) the skill is placed in the
agent-specific folder and the agent CLI is invoked to perform generation.

Exit codes: 0 = success, 1 = generation failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

from abevalflow import llm_client, skill_loader
from abevalflow.schemas import SubmissionMetadata

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert test engineer for AI skill evaluation pipelines.

Given an AI skill definition (SKILL.md) you must generate two files:

1. **instruction.md** — A clear task description that an AI agent will receive.
   It must describe what the agent should build or accomplish, including
   requirements, constraints, and expected outputs.  The task must be
   objectively verifiable by the accompanying tests.

2. **tests/test_outputs.py** — A pytest-compatible test file that verifies
   the agent's output.  Tests must:
   - Import from ``/workspace`` (the agent's working directory)
   - Be self-contained (no external fixtures)
   - Cover both success and edge cases
   - Use plain asserts (no custom frameworks)

{quality_criteria}

IMPORTANT RULES:
- Output ONLY valid JSON with keys "instruction_md" and "test_outputs_py".
- Each value is the full file content as a string.
- Do NOT include markdown fences or extra commentary.
- The instruction must be achievable by an AI agent in a single session.
- Tests must be deterministic and pass when the instruction is followed correctly.
"""

GENERATE_PROMPT_TEMPLATE = """\
Generate instruction.md and test_outputs.py for the following skill.

## Skill Name
{skill_name}

## Skill Description
{skill_description}

## Skill Content (SKILL.md)
{skill_content}

## Metadata
- Persona: {persona}
- Tags: {tags}

Respond with a single JSON object containing "instruction_md" and "test_outputs_py".
"""

AGENT_TASK_PROMPT = """\
I need you to generate test files for an AI skill evaluation.

The skill is defined in: {skill_md_path}

Please create:
1. {submission_dir}/instruction.md — a task description for an AI agent
2. {submission_dir}/tests/test_outputs.py — pytest tests verifying the output

The instruction should describe a task the agent must perform that exercises
the skill.  The tests must verify the agent's output from /workspace.

Read the SKILL.md first, then generate both files.
"""


def _load_submission(submission_dir: Path) -> tuple[SubmissionMetadata, str]:
    """Load metadata and skill content from the submission directory."""
    meta_path = submission_dir / "metadata.yaml"
    with meta_path.open() as f:
        raw = yaml.safe_load(f)
    metadata = SubmissionMetadata(**raw)

    skill_path = submission_dir / "skills" / "SKILL.md"
    skill_content = skill_path.read_text() if skill_path.is_file() else ""

    return metadata, skill_content


def _generate_via_api(
    submission_dir: Path,
    metadata: SubmissionMetadata,
    skill_content: str,
    quality_criteria: str,
) -> list[str]:
    """Call the LLM directly and parse structured JSON output."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        quality_criteria=(
            f"## Quality Criteria (from agentic-contribution-skill)\n\n{quality_criteria}"
            if quality_criteria
            else ""
        ),
    )

    user_prompt = GENERATE_PROMPT_TEMPLATE.format(
        skill_name=metadata.name,
        skill_description=metadata.description or "(not provided)",
        skill_content=skill_content,
        persona=metadata.persona or "general",
        tags=", ".join(metadata.tags) if metadata.tags else "none",
    )

    response_text = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=8192,
    )

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            raise ValueError("LLM response is not valid JSON")

    generated: list[str] = []

    instruction_content = parsed.get("instruction_md", "")
    if not instruction_content:
        raise ValueError("LLM did not produce instruction_md")
    (submission_dir / "instruction.md").write_text(instruction_content)
    generated.append("instruction.md")
    logger.info("Generated instruction.md (%d chars)", len(instruction_content))

    test_content = parsed.get("test_outputs_py", "")
    if not test_content:
        raise ValueError("LLM did not produce test_outputs_py")
    tests_dir = submission_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test_outputs.py").write_text(test_content)
    generated.append("tests/test_outputs.py")
    logger.info("Generated tests/test_outputs.py (%d chars)", len(test_content))

    llm_judge_content = parsed.get("llm_judge_py", "")
    if llm_judge_content:
        (tests_dir / "llm_judge.py").write_text(llm_judge_content)
        generated.append("tests/llm_judge.py")
        logger.info("Generated tests/llm_judge.py (%d chars)", len(llm_judge_content))

    return generated


def _generate_via_agent(
    submission_dir: Path,
    skill_dir: Path | None,
    agent_type: str,
    workspace_dir: Path,
) -> list[str]:
    """Invoke an agent CLI to perform generation with the skill loaded natively."""
    if skill_dir:
        skill_loader.place_for_agent(skill_dir, workspace_dir, agent_type)

    skill_md_path = submission_dir / "skills" / "SKILL.md"
    prompt = AGENT_TASK_PROMPT.format(
        skill_md_path=skill_md_path,
        submission_dir=submission_dir,
    )

    cli_commands: dict[str, list[str]] = {
        "claude": ["claude", "--print", "--dangerously-skip-permissions", "-p", prompt],
        "cursor": ["cursor", "--message", prompt],
        "opencode": ["opencode", "run", prompt],
    }

    cmd = cli_commands.get(agent_type)
    if cmd is None:
        raise ValueError(f"Unknown agent type: {agent_type}")

    logger.info("Invoking %s agent for generation", agent_type)
    try:
        subprocess.run(
            cmd,
            cwd=str(workspace_dir),
            check=True,
            timeout=300,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.error("%s CLI not found — is it installed?", agent_type)
        raise
    except subprocess.TimeoutExpired:
        logger.error("%s agent timed out after 300s", agent_type)
        raise

    generated: list[str] = []
    if (submission_dir / "instruction.md").is_file():
        generated.append("instruction.md")
    if (submission_dir / "tests" / "test_outputs.py").is_file():
        generated.append("tests/test_outputs.py")
    if (submission_dir / "tests" / "llm_judge.py").is_file():
        generated.append("tests/llm_judge.py")

    if not generated:
        raise RuntimeError(f"{agent_type} agent did not produce any files")

    return generated


def generate(
    submission_dir: Path,
    workspace_dir: Path,
    agent_type: str = "api",
) -> list[str]:
    """Run AI-assisted test generation and return list of generated file paths."""
    metadata, skill_content = _load_submission(submission_dir)

    if metadata.generation_mode != "ai":
        logger.info("generation_mode is '%s', nothing to generate", metadata.generation_mode)
        return []

    if not skill_content.strip():
        raise ValueError("skills/SKILL.md is empty — nothing to generate from")

    # Fetch the agentic-contribution-skill
    skill_cache = workspace_dir / "_skill_cache"
    skill_cache.mkdir(parents=True, exist_ok=True)
    skill_md = skill_loader.fetch_skill(skill_cache)
    skill_dir = skill_md.parent if skill_md else None

    if agent_type == "api":
        quality_criteria = ""
        if skill_md:
            quality_criteria = skill_loader.extract_quality_criteria(skill_md)
        return _generate_via_api(submission_dir, metadata, skill_content, quality_criteria)
    else:
        return _generate_via_agent(submission_dir, skill_dir, agent_type, workspace_dir)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="AI-assisted test generation")
    parser.add_argument("submission_dir", type=Path, help="Path to the submission directory")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="Pipeline workspace root (default: parent of submission_dir)",
    )
    parser.add_argument(
        "--agent-type",
        default=os.environ.get("AGENT_TYPE", "api"),
        help="Agent type: api, claude, cursor, opencode (default: api)",
    )
    args = parser.parse_args(argv)

    workspace = args.workspace_dir or args.submission_dir.parent
    if not args.submission_dir.is_dir():
        result = {"generated": [], "error": f"Not a directory: {args.submission_dir}"}
        print(json.dumps(result, indent=2))
        return 1

    try:
        generated = generate(args.submission_dir, workspace, args.agent_type)
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        result = {"generated": [], "error": str(exc)}
        print(json.dumps(result, indent=2))
        return 1

    result = {"generated": generated}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
