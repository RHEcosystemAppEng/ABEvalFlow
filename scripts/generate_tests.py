"""Generate instruction.md and test_outputs.py from a skill definition.

When ``generation_mode: ai`` is set in metadata.yaml the submitter provides
only ``skills/SKILL.md``.  This script uses an LLM (via the OpenAI-compatible
client) to produce the remaining files the pipeline requires:

  - ``instruction.md``  — task description for the agent
  - ``tests/test_outputs.py`` — pytest verification of agent output
  - ``tests/llm_judge.py`` (optional) — LLM-based evaluation

After each generation attempt the output is validated (structural checks +
LLM content review).  If validation fails, the errors are fed back into the
prompt and generation is retried up to ``max_retries`` times.

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
import re
import subprocess
import sys
from pathlib import Path

import yaml

from abevalflow import llm_client, skill_loader
from abevalflow.generation_validator import (
    check_markdown,
    check_python,
    content_check,
    structural_check,
)
from abevalflow.schemas import SubmissionMetadata

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3

SYSTEM_PROMPT = """\
You are an expert test engineer for AI skill evaluation pipelines.
{quality_criteria}
Output ONLY the requested file content — no markdown fences, no commentary.
"""

# --- Step 0: skill analysis ---------------------------------------------------

ANALYZE_SYSTEM_PROMPT = """\
You are an expert at analyzing AI agent skills for A/B evaluation.

Your job is to separate what a skill *uniquely contributes* from what any \
competent model already knows. This distinction is critical: the evaluation \
tests must target the skill's novel value-add, not general knowledge.
"""

ANALYZE_PROMPT = """\
Analyze the following skill definition. Identify:

1. **novel_aspects** — specific, opinionated, or non-obvious requirements that \
a model would likely NOT do (or do differently) without this skill. These are \
the skill's unique value.

2. **common_knowledge** — things any moderate-to-strong model already knows \
from training (standard patterns, well-known APIs, obvious approaches).

3. **test_focus_areas** — concrete things the instruction and tests should \
specifically target to measure whether having the skill actually helps. These \
should map directly to the novel aspects.

## Skill Content (SKILL.md)
{skill_content}

Output ONLY valid JSON with keys "novel_aspects", "common_knowledge", and \
"test_focus_areas". Each value must be a list of short strings.
"""

# --- Step 1: instruction.md ---------------------------------------------------

INSTRUCTION_PROMPT = """\
Given the following AI skill definition, write an **instruction.md** file.

The instruction must:
- Describe a concrete task an AI agent will receive
- Include requirements, constraints, and expected outputs
- Be achievable by an AI agent in a single session
- Be objectively verifiable by automated tests
- FOCUS on the novel aspects that the skill uniquely adds — design a task \
where having the skill gives a measurable advantage over not having it

## Skill Name
{skill_name}

## Skill Description
{skill_description}

## Skill Content (SKILL.md)
{skill_content}

## Skill Analysis
The following analysis identifies what this skill uniquely contributes \
versus what a model already knows. The instruction MUST target the novel \
aspects and test focus areas:

{skill_analysis}

## Metadata
- Persona: {persona}
- Tags: {tags}
{previous_errors}
Output ONLY the full content of instruction.md — nothing else.
"""

# --- Step 2: test_outputs.py --------------------------------------------------

TEST_PROMPT = """\
Given the skill definition and the instruction below, write a \
**tests/test_outputs.py** file that verifies the agent completed the task.

Tests must:
- Import from ``/workspace`` (the agent's working directory)
- Be self-contained (no external fixtures)
- Cover both success and edge cases
- Use plain asserts (no custom frameworks)
- Be deterministic and pass when the instruction is followed correctly
- MUST cover every test focus area listed below — these are what \
differentiate a skilled agent from an unskilled one

## Skill Content (SKILL.md)
{skill_content}

## Skill Analysis — Test Focus Areas
{skill_analysis}

## instruction.md (already generated)
{instruction_content}
{previous_errors}
Output ONLY valid Python source for test_outputs.py — nothing else.
"""

# --- Step 3 (optional): llm_judge.py -----------------------------------------

JUDGE_PROMPT = """\
Given the skill, instruction, and tests below, write an **optional** \
LLM-as-judge evaluator in ``tests/llm_judge.py``.

The judge should:
- Accept the agent's workspace path as input
- Use an LLM call to assess quality beyond what deterministic tests cover
- Produce a numeric score (0.0–1.0) and a short rationale
- Be self-contained (import its own LLM client)

If the skill and tests are simple enough that deterministic tests suffice, \
output exactly the word SKIP (nothing else).

## Skill Content (SKILL.md)
{skill_content}

## instruction.md
{instruction_content}

## tests/test_outputs.py
{test_content}

Output ONLY valid Python source for llm_judge.py, or the word SKIP.
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


def _error_block(errors: list[str] | None) -> str:
    if not errors:
        return ""
    lines = "\n".join(f"- {e}" for e in errors)
    return (
        f"\n## Previous Attempt Issues\n"
        f"Your previous attempt had these issues — fix them:\n{lines}\n"
    )


def _llm_call(system: str, user: str, *, max_tokens: int = 4096) -> str:
    """Single LLM call, strips markdown fences if the model wraps its output."""
    raw = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    text = raw.strip()
    for fence in ("```python", "```markdown", "```md", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            break
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _analyze_skill(skill_content: str) -> dict:
    """Step 0: Analyze the skill to separate novel aspects from common knowledge.

    Returns a dict with keys: novel_aspects, common_knowledge, test_focus_areas.
    Raises ValueError if the LLM response is not valid JSON or missing keys.
    """
    raw = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
            {"role": "user", "content": ANALYZE_PROMPT.format(skill_content=skill_content)},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    text = raw.strip()
    try:
        analysis = json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            raise ValueError("Skill analysis response is not valid JSON")

    required_keys = ("novel_aspects", "common_knowledge", "test_focus_areas")
    for key in required_keys:
        if key not in analysis or not isinstance(analysis[key], list):
            raise ValueError(f"Skill analysis missing or invalid key: {key}")
        if not analysis[key]:
            logger.warning("Skill analysis returned empty list for '%s'; proceeding", key)

    logger.info(
        "Step 0: skill analysis — %d novel, %d common, %d focus areas",
        len(analysis["novel_aspects"]),
        len(analysis["common_knowledge"]),
        len(analysis["test_focus_areas"]),
    )
    return analysis


def _format_analysis(analysis: dict) -> str:
    """Format the skill analysis dict into a readable text block for prompts."""
    lines: list[str] = []
    lines.append("**Novel aspects (skill's unique value):**")
    for item in analysis.get("novel_aspects", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**Common knowledge (model already knows):**")
    for item in analysis.get("common_knowledge", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**Test focus areas (what to specifically verify):**")
    for item in analysis.get("test_focus_areas", []):
        lines.append(f"- {item}")
    return "\n".join(lines)


def _generate_via_api(
    submission_dir: Path,
    metadata: SubmissionMetadata,
    skill_content: str,
    quality_criteria: str,
    previous_errors: list[str] | None = None,
) -> list[str]:
    """Generate files via 4 sequential LLM calls, validating after each step.

    Step 0: Analyze skill   (novel vs common knowledge breakdown)
    Step 1: instruction.md  (validated: exists, non-empty)
    Step 2: test_outputs.py (validated: exists, non-empty, compiles)
    Step 3: llm_judge.py    (optional; validated: compiles if produced)
    """
    system = SYSTEM_PROMPT.format(
        quality_criteria=(
            f"\n## Quality Criteria (from agentic-contribution-skill)\n\n{quality_criteria}"
            if quality_criteria
            else ""
        ),
    )

    generated: list[str] = []
    tests_dir = submission_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    # --- Step 0: skill analysis -----------------------------------------------
    analysis = _analyze_skill(skill_content)
    analysis_text = _format_analysis(analysis)

    # --- Step 1: instruction.md -----------------------------------------------
    instruction_text = _llm_call(
        system,
        INSTRUCTION_PROMPT.format(
            skill_name=metadata.name,
            skill_description=metadata.description or "(not provided)",
            skill_content=skill_content,
            skill_analysis=analysis_text,
            persona=metadata.persona or "general",
            tags=", ".join(metadata.tags) if metadata.tags else "none",
            previous_errors=_error_block(previous_errors),
        ),
    )
    (submission_dir / "instruction.md").write_text(instruction_text)
    logger.info("Step 1: generated instruction.md (%d chars)", len(instruction_text))

    errors = check_markdown(submission_dir / "instruction.md")
    if errors:
        raise ValueError(f"instruction.md validation failed: {errors}")
    generated.append("instruction.md")

    # --- Step 2: test_outputs.py ----------------------------------------------
    test_errors = [e for e in (previous_errors or []) if "test_outputs" in e.lower()]
    test_text = _llm_call(
        system,
        TEST_PROMPT.format(
            skill_content=skill_content,
            skill_analysis=analysis_text,
            instruction_content=instruction_text,
            previous_errors=_error_block(test_errors),
        ),
    )
    (tests_dir / "test_outputs.py").write_text(test_text)
    logger.info("Step 2: generated test_outputs.py (%d chars)", len(test_text))

    errors = check_python(tests_dir / "test_outputs.py")
    if errors:
        raise ValueError(f"test_outputs.py validation failed: {errors}")
    generated.append("tests/test_outputs.py")

    # --- Step 3: llm_judge.py (optional) --------------------------------------
    judge_text = _llm_call(
        system,
        JUDGE_PROMPT.format(
            skill_content=skill_content,
            instruction_content=instruction_text,
            test_content=test_text,
        ),
        max_tokens=2048,
    )
    if judge_text.strip().upper() != "SKIP":
        (tests_dir / "llm_judge.py").write_text(judge_text)
        logger.info("Step 3: generated llm_judge.py (%d chars)", len(judge_text))
        errors = check_python(tests_dir / "llm_judge.py")
        if errors:
            logger.warning("llm_judge.py failed validation, skipping: %s", errors)
            (tests_dir / "llm_judge.py").unlink(missing_ok=True)
        else:
            generated.append("tests/llm_judge.py")
    else:
        logger.info("Step 3: LLM chose to skip llm_judge.py")

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

    # WARNING: Claude's --dangerously-skip-permissions disables safety guardrails.
    # Only use agent mode with trusted submission sources. Untrusted SKILL.md
    # content could exploit an unrestricted agent to read secrets or write
    # arbitrary files. For untrusted submissions, use agent-type=api (default).
    cli_commands: dict[str, list[str]] = {
        "claude": ["claude", "--print", "--dangerously-skip-permissions", "-p", prompt],
        "cursor": ["cursor", "--message", prompt],
        "opencode": ["opencode", "run", prompt],
    }

    if agent_type != "api":
        logger.warning(
            "Agent mode '%s' executes with elevated permissions — "
            "ensure the submission source is trusted",
            agent_type,
        )

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
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[str]:
    """Run AI-assisted test generation with a validate-and-retry loop.

    After each generation attempt the output undergoes structural checks
    (existence, non-empty, py_compile) and an LLM content review.  On
    failure, errors are fed back into the next attempt.
    """
    metadata, skill_content = _load_submission(submission_dir)

    if metadata.generation_mode != "ai":
        logger.info("generation_mode is '%s', nothing to generate", metadata.generation_mode)
        return []

    if not skill_content.strip():
        raise ValueError("skills/SKILL.md is empty — nothing to generate from")

    skill_cache = workspace_dir / "_skill_cache"
    skill_cache.mkdir(parents=True, exist_ok=True)
    skill_md = skill_loader.fetch_skill(skill_cache)
    skill_dir = skill_md.parent if skill_md else None

    quality_criteria = ""
    if skill_md and agent_type == "api":
        quality_criteria = skill_loader.extract_quality_criteria(skill_md)

    last_errors: list[str] = []

    for attempt in range(1, max_retries + 1):
        logger.info("Generation attempt %d/%d", attempt, max_retries)

        try:
            if agent_type == "api":
                generated = _generate_via_api(
                    submission_dir,
                    metadata,
                    skill_content,
                    quality_criteria,
                    previous_errors=last_errors if last_errors else None,
                )
            else:
                generated = _generate_via_agent(
                    submission_dir, skill_dir, agent_type, workspace_dir,
                )
        except (ValueError, RuntimeError) as exc:
            last_errors = [str(exc)]
            logger.warning("Attempt %d generation error: %s", attempt, exc)
            if attempt == max_retries:
                raise
            continue

        struct_errors = structural_check(submission_dir)
        if struct_errors:
            last_errors = struct_errors
            logger.warning(
                "Attempt %d structural validation failed: %s", attempt, struct_errors,
            )
            if attempt == max_retries:
                raise ValueError(
                    f"Generation failed structural validation after {max_retries} "
                    f"attempts: {struct_errors}"
                )
            continue

        review = content_check(submission_dir)
        if not review["passed"]:
            last_errors = review["issues"]
            logger.warning(
                "Attempt %d content review failed: %s", attempt, review["issues"],
            )
            if attempt == max_retries:
                raise ValueError(
                    f"Generation failed content review after {max_retries} "
                    f"attempts: {review['issues']}"
                )
            continue

        logger.info("Generation validated on attempt %d", attempt)
        return generated

    raise ValueError(f"Generation failed after {max_retries} attempts")


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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("MAX_RETRIES", str(DEFAULT_MAX_RETRIES))),
        help=f"Max generation retry attempts (default: {DEFAULT_MAX_RETRIES})",
    )
    args = parser.parse_args(argv)

    workspace = args.workspace_dir or args.submission_dir.parent
    if not args.submission_dir.is_dir():
        result = {"generated": [], "error": f"Not a directory: {args.submission_dir}"}
        print(json.dumps(result, indent=2))
        return 1

    try:
        generated = generate(
            args.submission_dir, workspace, args.agent_type, args.max_retries,
        )
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
