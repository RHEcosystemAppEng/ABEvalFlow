"""Validate AI-generated submission files before they proceed down the pipeline.

Two layers:
  1. structural_check — file existence, non-empty, Python compilation (no LLM cost)
  2. content_check   — lightweight LLM coherence review (single cheap call)

Used inside the generate-validate retry loop in scripts/generate_tests.py.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

from abevalflow import llm_client

logger = logging.getLogger(__name__)

CONTENT_CHECK_SYSTEM_PROMPT = """\
You are a QA gate for AI-generated skill evaluation files.

You will receive a skill definition (SKILL.md), a generated task instruction \
(instruction.md), and generated tests (test_outputs.py).

Determine whether the three files are coherent and usable:
1. Does the instruction faithfully describe a task that exercises the skill?
2. Do the tests actually verify the instruction requirements?
3. Is the task achievable by an AI agent in a single session?

Output ONLY valid JSON:
{"pass": true, "issues": []}          // if acceptable
{"pass": false, "issues": ["..."]}    // if problems found

Be strict but pragmatic — minor style issues are acceptable.
"""

CONTENT_CHECK_USER_TEMPLATE = """\
## skills/SKILL.md
```
{skill_content}
```

## instruction.md
```
{instruction_content}
```

## tests/test_outputs.py
```python
{test_content}
```

Is this set coherent and usable? Respond with JSON only.
"""


def check_markdown(path: Path) -> list[str]:
    """Validate a markdown file exists and is non-empty."""
    if not path.is_file():
        return [f"{path.name} is missing"]
    if not path.read_text().strip():
        return [f"{path.name} is empty"]
    return []


def check_python(path: Path) -> list[str]:
    """Validate a Python file exists, is non-empty, and compiles."""
    if not path.is_file():
        return [f"{path.name} is missing"]
    source = path.read_text()
    if not source.strip():
        return [f"{path.name} is empty"]
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path.name} has a SyntaxError: {exc}"]
    return []


def structural_check(submission_dir: Path) -> list[str]:
    """Check all generated files for existence, non-emptiness, and compilation.

    Returns a list of error strings (empty = all checks passed).
    """
    errors: list[str] = []
    errors.extend(check_markdown(submission_dir / "instruction.md"))
    errors.extend(check_python(submission_dir / "tests" / "test_outputs.py"))

    llm_judge = submission_dir / "tests" / "llm_judge.py"
    if llm_judge.is_file() and llm_judge.read_text().strip():
        errors.extend(check_python(llm_judge))

    return errors


def content_check(submission_dir: Path) -> dict:
    """Run a lightweight LLM coherence review of the generated files.

    Returns ``{"passed": True/False, "issues": [str]}``.
    """
    skill_content = _read_safe(submission_dir / "skills" / "SKILL.md")
    instruction_content = _read_safe(submission_dir / "instruction.md")
    test_content = _read_safe(submission_dir / "tests" / "test_outputs.py")

    user_prompt = CONTENT_CHECK_USER_TEMPLATE.format(
        skill_content=skill_content,
        instruction_content=instruction_content,
        test_content=test_content,
    )

    response_text = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": CONTENT_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            logger.warning("Content check returned non-JSON: %s", response_text[:200])
            return {"passed": False, "issues": ["LLM content check returned invalid JSON"]}

    passed = bool(result.get("pass", False))
    issues = result.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]

    return {"passed": passed, "issues": issues}


def _read_safe(path: Path) -> str:
    if path.is_file():
        return path.read_text()
    return "(file not present)"
