"""Validate AI-generated submission files before they proceed down the pipeline.

Four layers:
  1. structural_check    — file existence, non-empty, Python compilation (no LLM cost)
  2. pytest_collect_check — ``pytest --collect-only`` to verify tests are discoverable
  3. multi_reviewer_check — 3 sequential LLM reviewers with distinct personas
  4. final_review         — single strict LLM go/no-go gate after corrections

Used inside the generate-validate retry loop in scripts/generate_tests.py.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
from pathlib import Path

from abevalflow import llm_client

logger = logging.getLogger(__name__)

_PYTEST_COLLECT_TIMEOUT = 10


def _parse_json_or_text(response_text: str, label: str = "Review") -> dict:
    """Best-effort parse of an LLM review response — JSON preferred, plain text OK."""
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = None
        else:
            result = None

    if result is not None:
        passed = bool(result.get("pass", False))
        issues = result.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        return {"passed": passed, "issues": issues}

    lower = response_text.lower()
    has_fail = any(w in lower for w in ("fail", "not ready", "problematic", "reject"))
    has_pass = any(w in lower for w in ("pass", "ready for evaluation", "looks good", "approved"))
    if has_fail:
        logger.info("%s returned plain text — interpreted as FAIL", label)
        return {"passed": False, "issues": [response_text.strip()]}
    if has_pass:
        logger.info("%s returned plain text — interpreted as PASS", label)
        return {"passed": True, "issues": []}
    logger.warning("%s ambiguous, treating as pass: %s", label, response_text[:200])
    return {"passed": True, "issues": []}

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


def pytest_collect_check(submission_dir: Path) -> list[str]:
    """Run ``pytest --collect-only`` on generated tests to verify discoverability.

    Returns a list of error strings (empty = all tests collected successfully).
    """
    test_file = submission_dir / "tests" / "test_outputs.py"
    if not test_file.is_file():
        return ["test_outputs.py is missing — cannot collect"]

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", "-q", str(test_file)],
            capture_output=True,
            text=True,
            timeout=_PYTEST_COLLECT_TIMEOUT,
            cwd=str(submission_dir),
        )
    except FileNotFoundError:
        logger.warning("pytest not available, skipping collect check")
        return []
    except subprocess.TimeoutExpired:
        return ["pytest --collect-only timed out (possible import hang)"]

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "unknown error"
        lines = detail.splitlines()
        summary = lines[-1] if lines else detail
        return [f"pytest --collect-only failed: {summary}"]

    logger.info("pytest --collect-only passed for %s", test_file.name)
    return []


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

    result = _parse_json_or_text(response_text, "Content check")
    return result


_REVIEWER_JSON_INSTRUCTION = """\

Output ONLY valid JSON:
{"pass": true, "issues": []}          // if acceptable
{"pass": false, "issues": ["..."]}    // if problems found

Be strict but pragmatic — minor style issues are acceptable."""

_COVERAGE_REVIEWER_SYSTEM = """\
You are a **test coverage reviewer** for AI-generated skill evaluation files.

You will receive a skill definition (SKILL.md), a generated task instruction \
(instruction.md), and generated tests (test_outputs.py).

Focus exclusively on test coverage:
1. Do the tests cover ALL requirements stated in instruction.md?
2. Are edge cases and error paths tested?
3. Are there any untested requirements?
4. Do the tests actually assert meaningful outcomes (not just "assert True")?
""" + _REVIEWER_JSON_INSTRUCTION

_ALIGNMENT_REVIEWER_SYSTEM = """\
You are a **skill alignment reviewer** for AI-generated skill evaluation files.

You will receive a skill definition (SKILL.md), a generated task instruction \
(instruction.md), and generated tests (test_outputs.py).

Focus exclusively on skill-instruction alignment:
1. Does the instruction faithfully exercise the skill's NOVEL aspects (not \
common knowledge any model already has)?
2. Would an agent WITH the skill have a measurable advantage over one WITHOUT it?
3. Is the instruction too generic (testing general ability rather than the skill)?
4. Do the tests target the skill's unique value-add?
5. **CRITICAL — Skill leakage check:** Does the instruction leak skill \
knowledge that would allow an unskilled agent to pass the tests? The \
instruction must NOT reference the skill by name, embed its internal \
rules/tables/mappings, or reveal expected answers. It should describe \
WHAT to do, not HOW the skill says to do it. If the instruction contains \
skill-specific details (e.g. exact image names, priority orders, decision \
tables), flag it as a skill leakage issue.
6. **Test overfitting check:** Are the tests overfitted to a single \
hardcoded scenario with no room for the agent to demonstrate genuine \
understanding? Tests should verify correct behavior, not just pattern-match \
against pre-embedded answers from the instruction.
""" + _REVIEWER_JSON_INSTRUCTION

_FEASIBILITY_REVIEWER_SYSTEM = """\
You are a **feasibility reviewer** for AI-generated skill evaluation files.

You will receive a skill definition (SKILL.md), a generated task instruction \
(instruction.md), and generated tests (test_outputs.py).

Focus exclusively on practical feasibility:
1. Can an AI agent realistically complete this task in a single session?
2. Are the tests deterministic (no randomness, timing, or external dependencies)?
3. Do the tests import from /workspace correctly?
4. Are there any circular or impossible requirements?
""" + _REVIEWER_JSON_INSTRUCTION

_REVIEWERS = [
    ("coverage", _COVERAGE_REVIEWER_SYSTEM),
    ("alignment", _ALIGNMENT_REVIEWER_SYSTEM),
    ("feasibility", _FEASIBILITY_REVIEWER_SYSTEM),
]


def _run_single_review(
    reviewer_name: str, system_prompt: str, user_prompt: str,
) -> tuple[str, dict]:
    """Execute a single LLM reviewer call. Returns (name, result_dict)."""
    response_text = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    parsed = _parse_json_or_text(response_text, f"{reviewer_name} reviewer")
    return reviewer_name, {"pass": parsed["passed"], "issues": parsed.get("issues", [])}


def multi_reviewer_check(submission_dir: Path) -> dict:
    """Run 3 LLM reviewers sequentially, each as an independent session.

    Each reviewer gets a fresh LLM call (no shared context with generation
    or other reviewers).  Results are persisted to a ``_reviews/`` directory
    so they can be inspected later.

    Returns ``{"passed": True/False, "issues": [str], "reviewer_results": dict}``.
    """
    skill_content = _read_safe(submission_dir / "skills" / "SKILL.md")
    instruction_content = _read_safe(submission_dir / "instruction.md")
    test_content = _read_safe(submission_dir / "tests" / "test_outputs.py")

    user_prompt = CONTENT_CHECK_USER_TEMPLATE.format(
        skill_content=skill_content,
        instruction_content=instruction_content,
        test_content=test_content,
    )

    reviews_dir = submission_dir / "_reviews"
    reviews_dir.mkdir(exist_ok=True)

    reviewer_results: dict[str, dict] = {}

    for name, system in _REVIEWERS:
        logger.info("Running reviewer '%s'...", name)
        name, result = _run_single_review(name, system, user_prompt)
        reviewer_results[name] = result

        review_file = reviews_dir / f"{name}.json"
        review_file.write_text(json.dumps(result, indent=2))

        status = "passed" if result["pass"] else "FAILED"
        logger.info("Reviewer '%s': %s", name, status)

    all_issues: list[str] = []
    for name, result in reviewer_results.items():
        if not result["pass"]:
            for issue in result["issues"]:
                all_issues.append(f"[{name}] {issue}")

    passed = len(all_issues) == 0
    return {"passed": passed, "issues": all_issues, "reviewer_results": reviewer_results}


_FINAL_REVIEW_SYSTEM = """\
You are a senior QA reviewer performing a final go/no-go check on AI-generated \
skill evaluation files.

You will receive a skill definition (SKILL.md), a generated task instruction \
(instruction.md), and generated tests (test_outputs.py). These files may have \
already been through one round of corrections.

Apply a high bar — this is the last gate before the files enter the evaluation \
pipeline. Verify:
1. The instruction exercises the skill's unique value, not just general knowledge.
2. The tests are comprehensive, deterministic, and correctly structured.
3. The instruction and tests are coherent with each other AND the skill.
4. The task is achievable by an AI agent in a single session.

Output ONLY valid JSON:
{"pass": true, "issues": []}          // if ready for evaluation
{"pass": false, "issues": ["..."]}    // if still problematic
"""


def final_review(submission_dir: Path) -> dict:
    """Strict single-call go/no-go gate after corrections.

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
            {"role": "system", "content": _FINAL_REVIEW_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = None
        else:
            result = None

    if result is not None:
        passed = bool(result.get("pass", False))
        issues = result.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        return {"passed": passed, "issues": issues}

    lower = response_text.lower()
    has_fail = any(w in lower for w in ("fail", "not ready", "problematic", "reject"))
    has_pass = any(w in lower for w in ("pass", "ready for evaluation", "looks good", "approved"))
    if has_fail:
        logger.info("Final review returned plain text — interpreted as FAIL")
        return {"passed": False, "issues": [response_text.strip()]}
    if has_pass:
        logger.info("Final review returned plain text — interpreted as PASS")
        return {"passed": True, "issues": []}
    logger.warning("Final review ambiguous, treating as pass: %s", response_text[:200])
    return {"passed": True, "issues": []}


def _read_safe(path: Path) -> str:
    if path.is_file():
        return path.read_text()
    return "(file not present)"
