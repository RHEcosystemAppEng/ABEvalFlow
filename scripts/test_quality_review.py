"""Independent AI quality review of a skill submission.

Reviews the coherence between skill, instruction, and tests regardless of
whether the submission was manually authored or AI-generated.  Produces a
structured JSON assessment with pass/warn/fail recommendation.

This review is ADVISORY (non-blocking) - always exits 0 regardless of result.
The assessment is logged for informational purposes but does not fail the pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from abevalflow import llm_client
from abevalflow.schemas import SubmissionMetadata

logger = logging.getLogger(__name__)

REVIEW_SYSTEM_PROMPT_HARBOR = """\
You are a senior QA engineer reviewing AI skill evaluation submissions.
You must assess the quality of a submission that consists of:
- A skill definition (SKILL.md)
- A task instruction (instruction.md)
- Verification tests (test_outputs.py)
- Optionally an LLM judge (llm_judge.py)

Evaluate the following dimensions and assign each a score from 0.0 to 1.0:

1. **coherence** — Does the instruction faithfully test what the skill describes?
2. **coverage** — Do the tests adequately verify the instruction requirements?
3. **clarity** — Are the instruction and tests clear and unambiguous?
4. **feasibility** — Can an AI agent reasonably complete the task in a single session?
5. **robustness** — Do the tests cover edge cases and failure modes?

For each dimension, provide a brief finding (1-2 sentences).

Then provide an overall recommendation:
- "pass" — Submission is ready for evaluation (all scores >= 0.6)
- "warn" — Minor issues but can proceed (some scores 0.4-0.6)
- "fail" — Significant problems, should not proceed (any score < 0.4)

Output ONLY valid JSON with this structure:
{
  "dimensions": {
    "coherence": {"score": 0.0, "finding": "..."},
    "coverage": {"score": 0.0, "finding": "..."},
    "clarity": {"score": 0.0, "finding": "..."},
    "feasibility": {"score": 0.0, "finding": "..."},
    "robustness": {"score": 0.0, "finding": "..."}
  },
  "overall_score": 0.0,
  "recommendation": "pass|warn|fail",
  "summary": "One paragraph overall assessment"
}
"""

REVIEW_SYSTEM_PROMPT_ASE = """\
You are a senior QA engineer reviewing AI skill evaluation submissions.
You must assess the quality of a submission that uses the ASE (Agent Skills Eval) format:
- A skill definition (SKILL.md)
- Evaluation definitions (evals.json with prompts and assertions)

Evaluate the following dimensions and assign each a score from 0.0 to 1.0:

1. **coherence** — Do the eval prompts faithfully test what the skill describes?
2. **coverage** — Do the assertions adequately verify the expected behavior?
3. **clarity** — Are the prompts and assertions clear and unambiguous?
4. **feasibility** — Can an AI agent reasonably complete the tasks?
5. **robustness** — Do the evals cover different scenarios and edge cases?

For each dimension, provide a brief finding (1-2 sentences).

Then provide an overall recommendation:
- "pass" — Submission is ready for evaluation (all scores >= 0.6)
- "warn" — Minor issues but can proceed (some scores 0.4-0.6)
- "fail" — Significant problems, should not proceed (any score < 0.4)

Output ONLY valid JSON with this structure:
{
  "dimensions": {
    "coherence": {"score": 0.0, "finding": "..."},
    "coverage": {"score": 0.0, "finding": "..."},
    "clarity": {"score": 0.0, "finding": "..."},
    "feasibility": {"score": 0.0, "finding": "..."},
    "robustness": {"score": 0.0, "finding": "..."}
  },
  "overall_score": 0.0,
  "recommendation": "pass|warn|fail",
  "summary": "One paragraph overall assessment"
}
"""

REVIEW_USER_TEMPLATE_HARBOR = """\
Review the following submission.

## Skill Name
{skill_name}

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

{llm_judge_section}

Respond with the JSON assessment.
"""

REVIEW_USER_TEMPLATE_ASE = """\
Review the following ASE submission.

## Skill Name
{skill_name}

## skills/SKILL.md
```
{skill_content}
```

## evals/evals.json
```json
{evals_content}
```

Respond with the JSON assessment.
"""


def _read_file_safe(path: Path) -> str:
    if path.is_file():
        return path.read_text()
    return "(file not present)"


def _find_skill_md(submission_dir: Path) -> Path | None:
    """Find SKILL.md in various locations (newer and older formats)."""
    # Newer format: skills/SKILL.md
    direct = submission_dir / "skills" / "SKILL.md"
    if direct.is_file():
        return direct
    # Older Harbor format: skills/<skill-name>/SKILL.md
    skills_dir = submission_dir / "skills"
    if skills_dir.is_dir():
        for subdir in skills_dir.iterdir():
            if subdir.is_dir():
                nested = subdir / "SKILL.md"
                if nested.is_file():
                    return nested
    return None


def review_submission(submission_dir: Path) -> dict:
    """Run AI quality review and return the structured assessment."""
    meta_path = submission_dir / "metadata.yaml"
    with meta_path.open() as f:
        raw = yaml.safe_load(f)
    metadata = SubmissionMetadata(**raw)

    skill_md_path = _find_skill_md(submission_dir)
    skill_content = _read_file_safe(skill_md_path) if skill_md_path else "(file not present)"

    # Detect ASE vs Harbor format
    evals_path = submission_dir / "evals" / "evals.json"
    is_ase = evals_path.is_file()

    if is_ase:
        # ASE format: uses evals/evals.json
        evals_content = _read_file_safe(evals_path)
        system_prompt = REVIEW_SYSTEM_PROMPT_ASE
        user_prompt = REVIEW_USER_TEMPLATE_ASE.format(
            skill_name=metadata.name,
            skill_content=skill_content,
            evals_content=evals_content,
        )
    else:
        # Harbor format: uses instruction.md and tests/
        instruction_content = _read_file_safe(submission_dir / "instruction.md")
        test_content = _read_file_safe(submission_dir / "tests" / "test_outputs.py")

        llm_judge_path = submission_dir / "tests" / "llm_judge.py"
        llm_judge_section = ""
        if llm_judge_path.is_file():
            llm_judge_section = f"## tests/llm_judge.py\n```python\n{llm_judge_path.read_text()}\n```"

        system_prompt = REVIEW_SYSTEM_PROMPT_HARBOR
        user_prompt = REVIEW_USER_TEMPLATE_HARBOR.format(
            skill_name=metadata.name,
            skill_content=skill_content,
            instruction_content=instruction_content,
            test_content=test_content,
            llm_judge_section=llm_judge_section,
        )

    response_text = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
    )

    try:
        assessment = json.loads(response_text)
    except json.JSONDecodeError:
        import re

        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            assessment = json.loads(json_match.group())
        else:
            raise ValueError("LLM review response is not valid JSON")

    if "overall_score" not in assessment:
        dims = assessment.get("dimensions", {})
        scores = [d.get("score", 0.0) for d in dims.values() if isinstance(d, dict)]
        assessment["overall_score"] = sum(scores) / len(scores) if scores else 0.0

    if "recommendation" not in assessment:
        score = assessment["overall_score"]
        if score >= 0.6:
            assessment["recommendation"] = "pass"
        elif score >= 0.4:
            assessment["recommendation"] = "warn"
        else:
            assessment["recommendation"] = "fail"

    assessment["passed"] = assessment["recommendation"] != "fail"

    return assessment


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="AI quality review of submission")
    parser.add_argument("submission_dir", type=Path, help="Path to the submission directory")
    args = parser.parse_args(argv)

    if not args.submission_dir.is_dir():
        result = {"passed": False, "error": f"Not a directory: {args.submission_dir}"}
        print(json.dumps(result, indent=2))
        return 1

    try:
        assessment = review_submission(args.submission_dir)
    except Exception as exc:
        logger.error("Review failed: %s", exc)
        result = {"passed": False, "error": str(exc)}
        print(json.dumps(result, indent=2))
        return 1

    print(json.dumps(assessment, indent=2))
    # Always return 0 - this review is advisory (non-blocking)
    # The assessment result is logged but doesn't fail the pipeline
    if not assessment.get("passed", False):
        logger.warning(
            "Quality review recommendation: %s (advisory, non-blocking)", assessment.get("recommendation", "unknown")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
