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
    multi_reviewer_check,
    pytest_collect_check,
    scenario_coherence_check,
    structural_check,
)
from abevalflow.schemas import SubmissionMetadata
logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5

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

# --- Step 0.5: scenario brief -------------------------------------------------

SCENARIO_PROMPT = """\
You are designing a concrete test scenario for an A/B evaluation of an AI \
skill. Given the skill definition and analysis below, produce a \
**scenario brief** — a single JSON document that defines ALL the concrete \
data the evaluation will use.

The scenario brief is the SINGLE SOURCE OF TRUTH. Both the instruction \
(given to the agent) and the tests (verifying the agent's output) will be \
generated from this brief. Nothing may be invented outside of it.

## What to include

1. **project_files** — a dict mapping file paths to their full contents. \
These are the files the agent will analyze. Make them realistic, \
internally consistent, and rich enough to exercise the skill's novel \
aspects. Include 3-7 files.

2. **expected_outputs** — a dict mapping every output field/variable \
the agent must produce to its correct value. These are the ground-truth \
answers that tests will assert against. Every value must be derivable \
from the project_files + the skill's knowledge.

3. **rationale** — a dict mapping each expected output field to a short \
explanation of why that value is correct (referencing specific files or \
skill rules). This is for validation only — it will NOT appear in the \
instruction or tests.

## Design rules

- The scenario MUST exercise the skill's **novel aspects** (listed below) \
— include project files that create situations where the skill's unique \
knowledge matters.
- Expected outputs MUST be deterministically derivable from the project \
files. Do NOT include subjective or ambiguous expected values.
- Project file contents must be realistic and internally consistent \
(e.g. if pyproject.toml says Python >=3.11, requirements.txt should have \
compatible packages).
- Do NOT reference the skill by name anywhere in the project files.

## Skill Content (SKILL.md)
{skill_content}

## Skill Analysis
{skill_analysis}

Output ONLY valid JSON with keys "project_files", "expected_outputs", \
and "rationale". No markdown fences, no commentary.
"""

# --- Step 1: instruction.md ---------------------------------------------------

INSTRUCTION_PROMPT = """\
Given the scenario brief and skill definition below, write an \
**instruction.md** file.

The instruction must:
- Describe a concrete task an AI agent will receive
- Include the project files from the scenario brief VERBATIM as inline \
content so the agent has real data to work with
- Specify the expected output format and all required fields
- Be achievable by an AI agent in a single session
- Be objectively verifiable by automated tests
- FOCUS on the novel aspects that the skill uniquely adds

## CRITICAL: Output Format Contract

The instruction MUST tell the agent to write its solution to a single \
file at ``/solution/solution.md``. The file must contain:

1. A **narrative section** with the agent's reasoning and analysis
2. A **Variables Table** at the end — a markdown table with columns \
``| Variable | Value |`` listing every required output field and its value

Example (do NOT use these values — derive from the scenario brief):
```
| Variable | Value |
|----------|-------|
| APP_NAME | my-app |
| LANGUAGE | python |
```

This format is the contract between instruction, tests, and judge. \
Both ``test_outputs.py`` and ``llm_judge.py`` will read from \
``/solution/solution.md`` and parse the Variables Table.

## CRITICAL: Data Rules

1. **You MUST embed every file from the scenario brief's project_files** \
in the instruction, with its full contents. The agent needs actual data \
to analyze — never tell it to "guess from file names".

2. **DO NOT add project files that are not in the scenario brief.** The \
scenario brief is the single source of truth.

3. **DO NOT embed the expected answers** from the scenario brief in the \
instruction. The agent must derive them from the project files + its own \
knowledge (or the skill's knowledge, for skilled agents).

## CRITICAL: Instruction Isolation Rules

The instruction will be given to BOTH a skilled agent (with the SKILL.md) \
and an unskilled agent (WITHOUT the SKILL.md). To produce a valid A/B \
comparison you MUST follow these rules:

1. **DO NOT expose skill internals** — never mention the skill by name, \
reference its rules, tables, mappings, workflows, or internal logic in the \
instruction. The instruction must read as a natural task description, not \
as a summary of the skill.

2. **DO NOT embed answers** — the instruction must not contain the expected \
values, correct outputs, or decision criteria that only the skill defines. \
For example, do NOT say "the image should be registry.../nodejs-20" or \
"use priority order X". Just ask for the result and let the agent figure \
it out.

3. **DO NOT reference the skill document** — phrases like "per the skill", \
"as defined in SKILL.md", "using the /skill-name skill", or "the skill's \
table" are forbidden in the instruction.

4. **DO describe the task naturally** — state what the agent should do \
(e.g. "analyze this project and produce a detection report"), what inputs \
are available, and what output format is expected. The task should be \
understandable to someone who has never seen the skill.

5. **Let the skill provide the edge** — the advantage of having the skill \
should come from the knowledge it provides (correct mappings, priority \
orders, conventions), NOT from the instruction repeating that knowledge.

## Scenario Brief
{scenario_brief}

## Skill Name
{skill_name}

## Skill Description
{skill_description}

## Skill Content (SKILL.md)
{skill_content}

## Skill Analysis
{skill_analysis}

## Metadata
- Persona: {persona}
- Tags: {tags}
{previous_errors}
Output ONLY the full content of instruction.md — nothing else.
"""

# --- Step 2: test_outputs.py --------------------------------------------------

TEST_PROMPT = """\
Given the scenario brief, skill definition, and instruction below, write a \
**tests/test_outputs.py** file that verifies the agent completed the task.

Tests must:
- Be self-contained (no external fixtures)
- Use plain asserts (no custom frameworks)
- Be deterministic and pass when the instruction is followed correctly
- MUST cover every test focus area listed below — these are what \
differentiate a skilled agent from an unskilled one

## CRITICAL: Reading the Agent's Output

The agent writes its solution to ``/solution/solution.md``. This file \
contains a narrative section and a **Variables Table** (a markdown table \
with ``| Variable | Value |`` columns).

Your tests MUST:
1. Read from ``/solution/solution.md`` — this is the ONLY file to check
2. Parse the Variables Table to extract output values (write a helper \
function ``parse_variables_table(content: str) -> dict`` that extracts \
the Variable→Value mapping from the markdown table)
3. Also check the narrative/free-form content for qualitative aspects

## CRITICAL: Use the Scenario Brief as Ground Truth

The scenario brief contains **expected_outputs** — these are the ONLY \
correct values for the scenario. You MUST:

1. **Assert against expected_outputs values** — every test that checks a \
specific output value must use the exact value from the scenario brief's \
expected_outputs. Do NOT invent values.

2. **Do NOT add assertions for data not in expected_outputs** — if a \
field is not in expected_outputs, either skip it or test only that it is \
present and non-empty.

3. **Test the WHAT, not the HOW** — verify that the agent produced the \
correct output values, not that it followed a specific internal process.

## CRITICAL: Fair Testing — No Overfitting

**IMPORTANT: The reward is proportional (passed_tests / total_tests), \
NOT binary.** Each test that passes independently contributes to the \
score. This means every test must be self-contained and test ONE thing. \
A single over-strict assertion should not zero out the entire reward.

4. **Tests must be fair to both variants** — remember these tests run \
against both a skilled agent (with SKILL.md) and an unskilled agent \
(without it). Tests should measure genuine capability differences, not \
trick questions that only pass if the agent memorized a specific table.

5. **Do NOT overfit tests** — avoid testing for:
   - Exact formatting, punctuation, or whitespace
   - Specific phrasing in the narrative section
   - Implementation details the agent wasn't asked for
   - Hardcoded strings that only appear in the skill document
   Tests should validate that the agent correctly analyzed the input and \
   produced the right answers, NOT that it formatted them a specific way.

6. **Use flexible comparisons** — for string values:
   - Use case-insensitive comparisons (e.g. ``value.lower() == "python"``)
   - Accept reasonable synonyms (e.g. "fastapi" and "FastAPI")
   - For file paths, normalize before comparing (strip leading ``./``, \
     trailing ``/``, and compare with ``os.path.normpath``)
   - Never assert exact string equality when semantic equivalence is \
     what matters (e.g. ``"chart/"`` and ``"./chart"`` are the same path)

7. **Keep tests concise and complete** — aim for 15-25 focused test \
functions, not 40+. Every test function MUST have a complete body (no \
truncated or stub functions). If the file would be too long, reduce the \
number of tests rather than risk truncation. Initialize all module-level \
variables inside functions or fixtures, not at import time.

## Scenario Brief (GROUND TRUTH)
{scenario_brief}

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
Given the skill, instruction, and tests below, write an \
LLM-as-judge evaluator in ``tests/llm_judge.py``.

The judge should:
- Read the agent's output from ``/solution/solution.md`` (same file the \
  tests use — contains a narrative section and a Variables Table)
- Do NOT use argparse or CLI args — hardcode all paths
- Use an LLM call to assess quality beyond what deterministic tests cover
- Produce a numeric score (0.0–1.0) and a short rationale
- Write the result as JSON to /logs/verifier/reward.json (Harbor convention)
- Focus on aspects that deterministic tests cannot capture (reasoning \
  quality, completeness of analysis, correctness of approach)

## CRITICAL: PEP 723 inline script metadata

The script MUST start with PEP 723 inline script metadata so that
``uv run`` can auto-install dependencies. Use this exact format at the
top of the file (after the shebang if any):

```
# /// script
# dependencies = [
#   "openai>=1.30.0",
# ]
# ///
```

Use the ``openai`` SDK pointed at the LiteLLM proxy. Read these env vars
(they are injected by Harbor via [verifier.env] in task.toml):
- ``LLM_API_BASE`` — base URL of the OpenAI-compatible proxy
- ``LLM_API_KEY`` — API key for the proxy
- ``MODEL_NAME`` — model to use (e.g. "claude-sonnet")

Example client setup:
```python
client = openai.OpenAI(base_url=os.environ["LLM_API_BASE"], api_key=os.environ["LLM_API_KEY"])
```

Do NOT use the ``anthropic`` SDK — use only ``openai`` via the proxy.

## Skill Content (SKILL.md)
{skill_content}

## instruction.md
{instruction_content}

## tests/test_outputs.py
{test_content}

Output ONLY valid Python source for llm_judge.py — nothing else.
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


def _generate_scenario_brief(
    skill_content: str,
    analysis_text: str,
    system: str,
) -> tuple[dict, str]:
    """Step 0.5: Generate a scenario brief — the single source of truth.

    Returns (parsed_brief_dict, brief_as_json_string).
    """
    raw = _llm_call(
        system,
        SCENARIO_PROMPT.format(
            skill_content=skill_content,
            skill_analysis=analysis_text,
        ),
        max_tokens=8192,
    )

    try:
        brief = json.loads(raw)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            brief = json.loads(json_match.group())
        else:
            raise ValueError(f"Scenario brief is not valid JSON: {raw[:300]}")

    for key in ("project_files", "expected_outputs"):
        if key not in brief or not isinstance(brief[key], dict):
            raise ValueError(f"Scenario brief missing or invalid key: {key}")
        if not brief[key]:
            raise ValueError(f"Scenario brief has empty '{key}'")

    logger.info(
        "Step 0.5: scenario brief — %d project files, %d expected outputs",
        len(brief["project_files"]),
        len(brief["expected_outputs"]),
    )
    return brief, json.dumps(brief, indent=2)


def _generate_via_api(
    submission_dir: Path,
    metadata: SubmissionMetadata,
    skill_content: str,
    quality_criteria: str,
    previous_errors: list[str] | None = None,
) -> tuple[list[str], dict | None]:
    """Generate files via 5 sequential LLM calls, validating after each step.

    Step 0:   Analyze skill     (novel vs common knowledge breakdown)
    Step 0.5: Scenario brief    (single source of truth for project data)
    Step 1:   instruction.md    (validated: exists, non-empty)
    Step 2:   test_outputs.py   (validated: exists, non-empty, compiles)
    Step 3:   llm_judge.py      (optional; validated: compiles if produced)

    Returns (generated_files_list, scenario_brief_dict).
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

    # --- Step 0.5: scenario brief ---------------------------------------------
    brief, brief_json = _generate_scenario_brief(
        skill_content, analysis_text, system,
    )
    (submission_dir / "scenario_brief.json").write_text(brief_json)

    # --- Step 1: instruction.md -----------------------------------------------
    instruction_text = _llm_call(
        system,
        INSTRUCTION_PROMPT.format(
            scenario_brief=brief_json,
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
            scenario_brief=brief_json,
            skill_content=skill_content,
            skill_analysis=analysis_text,
            instruction_content=instruction_text,
            previous_errors=_error_block(test_errors),
        ),
        max_tokens=65536,
    )
    (tests_dir / "test_outputs.py").write_text(test_text)
    logger.info("Step 2: generated test_outputs.py (%d chars)", len(test_text))

    errors = check_python(tests_dir / "test_outputs.py")
    if errors:
        raise ValueError(f"test_outputs.py validation failed: {errors}")
    generated.append("tests/test_outputs.py")

    # --- Step 3: llm_judge.py (default: generated; skip via metadata) ---------
    if metadata.skip_llm_judge:
        logger.info("Step 3: skipping llm_judge.py (skip_llm_judge=true in metadata)")
    else:
        judge_text = _llm_call(
            system,
            JUDGE_PROMPT.format(
                skill_content=skill_content,
                instruction_content=instruction_text,
                test_content=test_text,
            ),
            max_tokens=65536,
        )
        (tests_dir / "llm_judge.py").write_text(judge_text)
        logger.info("Step 3: generated llm_judge.py (%d chars)", len(judge_text))
        errors = check_python(tests_dir / "llm_judge.py")
        if errors:
            logger.warning("llm_judge.py failed validation, removing: %s", errors)
            (tests_dir / "llm_judge.py").unlink(missing_ok=True)
        else:
            generated.append("tests/llm_judge.py")

    return generated, brief


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


def _generate_via_oracle(
    submission_dir: Path,
    metadata: SubmissionMetadata,
) -> list[str]:
    """Copy pre-baked files from oracle/ directory — no LLM needed.

    Expects the submission to contain an ``oracle/`` subdirectory with
    ``instruction.md``, ``test_outputs.py``, and optionally ``llm_judge.py``.
    """
    oracle_dir = submission_dir / "oracle"
    if not oracle_dir.is_dir():
        raise ValueError(
            f"Oracle mode requires an oracle/ directory in {submission_dir}"
        )

    import shutil

    tests_dir = submission_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    generated: list[str] = []

    src_instruction = oracle_dir / "instruction.md"
    if src_instruction.is_file():
        shutil.copy2(src_instruction, submission_dir / "instruction.md")
        generated.append("instruction.md")
        logger.info("Oracle: copied instruction.md (%d bytes)", src_instruction.stat().st_size)
    else:
        raise ValueError("oracle/instruction.md not found")

    src_test = oracle_dir / "test_outputs.py"
    if src_test.is_file():
        shutil.copy2(src_test, tests_dir / "test_outputs.py")
        generated.append("tests/test_outputs.py")
        logger.info("Oracle: copied test_outputs.py (%d bytes)", src_test.stat().st_size)
    else:
        raise ValueError("oracle/test_outputs.py not found")

    if not metadata.skip_llm_judge:
        src_judge = oracle_dir / "llm_judge.py"
        if src_judge.is_file():
            shutil.copy2(src_judge, tests_dir / "llm_judge.py")
            generated.append("tests/llm_judge.py")
            logger.info("Oracle: copied llm_judge.py (%d bytes)", src_judge.stat().st_size)

    return generated


CORRECTION_SYSTEM_PROMPT = """\
You are an expert test engineer fixing issues in AI-generated evaluation files.

You will receive the current instruction.md and tests/test_outputs.py along \
with a list of issues identified by reviewers. Fix ONLY the cited issues — \
preserve everything else.

Output the corrected file content when asked. No markdown fences, no commentary.
"""

CORRECTION_INSTRUCTION_PROMPT = """\
Fix the following issues in instruction.md. Preserve all content that is not \
related to the issues.

## Current instruction.md
{instruction_content}

## Issues to Fix
{issues}

Output ONLY the corrected instruction.md — nothing else.
"""

CORRECTION_TEST_PROMPT = """\
Fix the following issues in tests/test_outputs.py. Preserve all content that \
is not related to the issues.

## Current tests/test_outputs.py
{test_content}

## Issues to Fix
{issues}

Output ONLY the corrected test_outputs.py — nothing else.
"""


def _correction_pass(
    submission_dir: Path,
    issues: list[str],
    system: str = CORRECTION_SYSTEM_PROMPT,
) -> None:
    """Apply targeted corrections based on consolidated reviewer feedback."""
    issues_text = "\n".join(f"- {issue}" for issue in issues)
    logger.info("Correction pass: addressing %d issues", len(issues))

    instruction_path = submission_dir / "instruction.md"
    test_path = submission_dir / "tests" / "test_outputs.py"

    instruction_issues = [i for i in issues if "test" not in i.lower() or "instruction" in i.lower()]
    test_issues = [i for i in issues if "test" in i.lower() or "coverage" in i.lower()]

    if not test_issues:
        test_issues = issues
    if not instruction_issues:
        instruction_issues = issues

    if instruction_path.is_file():
        corrected = _llm_call(
            system,
            CORRECTION_INSTRUCTION_PROMPT.format(
                instruction_content=instruction_path.read_text(),
                issues="\n".join(f"- {i}" for i in instruction_issues),
            ),
        )
        if corrected.strip():
            instruction_path.write_text(corrected)
            logger.info("Correction pass: rewrote instruction.md (%d chars)", len(corrected))

    if test_path.is_file():
        corrected = _llm_call(
            system,
            CORRECTION_TEST_PROMPT.format(
                test_content=test_path.read_text(),
                issues="\n".join(f"- {i}" for i in test_issues),
            ),
        )
        if corrected.strip():
            test_path.write_text(corrected)
            logger.info("Correction pass: rewrote test_outputs.py (%d chars)", len(corrected))


def generate(
    submission_dir: Path,
    workspace_dir: Path,
    agent_type: str = "api",
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[str]:
    """Run AI-assisted test generation with iterative review-fix cycles.

    Each generation attempt undergoes:
      4. structural_check     — file existence, non-empty, ast.parse
      5. pytest_collect_check  — ``pytest --collect-only``
      6. multi_reviewer_check  — 3 sequential LLM reviewers
      7. correction_pass       — targeted LLM fix (if reviewers found issues)

    Once structural + pytest checks pass, the files are accepted.
    Review/fix cycles improve quality but never reject — no final gate.
    """
    metadata, skill_content = _load_submission(submission_dir)

    if metadata.generation_mode != "ai":
        logger.info("generation_mode is '%s', nothing to generate", metadata.generation_mode)
        return []

    if not skill_content.strip():
        raise ValueError("skills/SKILL.md is empty — nothing to generate from")

    # Oracle mode: copy pre-baked files, run deterministic checks only, no LLM
    if agent_type == "oracle":
        generated = _generate_via_oracle(submission_dir, metadata)
        struct_errors = structural_check(submission_dir)
        if struct_errors:
            raise ValueError(f"Oracle structural check failed: {struct_errors}")
        collect_errors = pytest_collect_check(submission_dir)
        if collect_errors:
            raise ValueError(f"Oracle pytest collect failed: {collect_errors}")
        logger.info("Oracle mode: %d files validated", len(generated))
        return generated

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

        # --- Steps 0-3: generate files -----------------------------------------
        scenario_brief = None
        try:
            if agent_type == "api":
                generated, scenario_brief = _generate_via_api(
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

        # --- Step 4: structural check ------------------------------------------
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

        # --- Step 5: pytest --collect-only -------------------------------------
        collect_errors = pytest_collect_check(submission_dir)
        if collect_errors:
            last_errors = collect_errors
            logger.warning(
                "Attempt %d pytest collect failed: %s", attempt, collect_errors,
            )
            if attempt == max_retries:
                raise ValueError(
                    f"Generation failed pytest collect after {max_retries} "
                    f"attempts: {collect_errors}"
                )
            continue

        # --- Step 5.5: scenario coherence check --------------------------------
        if scenario_brief is not None:
            coherence_errors = scenario_coherence_check(submission_dir, scenario_brief)
            if coherence_errors:
                last_errors = coherence_errors
                logger.warning(
                    "Attempt %d coherence check failed: %s", attempt, coherence_errors,
                )
                if attempt == max_retries:
                    raise ValueError(
                        f"Generation failed coherence check after {max_retries} "
                        f"attempts: {coherence_errors}"
                    )
                continue

        # --- Step 6: multi-reviewer check --------------------------------------
        review = multi_reviewer_check(submission_dir)
        if not review["passed"]:
            logger.info(
                "Attempt %d: reviewers found %d issues, running correction pass",
                attempt, len(review["issues"]),
            )

            # --- Step 7: correction pass ---------------------------------------
            _correction_pass(submission_dir, review["issues"])

            post_errors = structural_check(submission_dir)
            post_errors += pytest_collect_check(submission_dir)
            if post_errors:
                last_errors = post_errors
                logger.warning(
                    "Attempt %d post-correction validation failed: %s",
                    attempt, post_errors,
                )
                if attempt == max_retries:
                    raise ValueError(
                        f"Generation failed post-correction validation after "
                        f"{max_retries} attempts: {post_errors}"
                    )
                continue
        else:
            logger.info("Attempt %d: reviewers passed, no corrections needed", attempt)

        logger.info("Generation accepted on attempt %d", attempt)
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
