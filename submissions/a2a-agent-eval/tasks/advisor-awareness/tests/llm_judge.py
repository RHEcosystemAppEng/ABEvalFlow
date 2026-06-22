#!/usr/bin/env python3
"""LLM Judge verifier for A2A agent responses.

Evaluates agent responses using an LLM-as-judge approach.
Reads the agent response and scores it based on predefined criteria.
"""

import json
import os
import sys

import litellm

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for AI agent responses.

Your task is to evaluate an AI agent's response about Red Hat Insights Advisor domain knowledge.
The user asked: "I'm planning to use you to help manage my Red Hat systems. Can you explain what Red Hat Insights Advisor is and give me an example of the kind of advisor recommendation you might show me for a RHEL system?"

Evaluate the response on these criteria (each scored 0-1):

1. **Domain accuracy** (0-1): Does the agent correctly describe what Advisor is (configuration/security/performance recommendations for RHEL systems)?
   - 1.0: Accurately describes Advisor as a recommendation engine for RHEL covering configuration, security, and performance
   - 0.5: Partially correct but missing key aspects or conflating Advisor with other Insights services
   - 0.0: Incorrect, vague, or off-topic description

2. **Specificity** (0-1): Does it give a concrete, plausible example recommendation?
   - 1.0: Provides a specific, realistic example (e.g., a named rule type, CVE, config issue, or performance finding)
   - 0.5: Gives a generic example without concrete detail
   - 0.0: No example provided or example is implausible

3. **Helpfulness** (0-1): Is the response actionable and useful?
   - 1.0: Clear, well-organized, and helps the user understand how Advisor would help them
   - 0.5: Somewhat useful but lacks clarity or actionable context
   - 0.0: Unhelpful, confusing, or refuses to answer

Respond ONLY with valid JSON in this exact format:
{
    "domain_accuracy": <float 0-1>,
    "specificity": <float 0-1>,
    "helpfulness": <float 0-1>,
    "overall_score": <float 0-1>,
    "reasoning": "<brief explanation>"
}

The overall_score should be the weighted average: (domain_accuracy*0.4 + specificity*0.35 + helpfulness*0.25)
"""


def evaluate_response(response_text: str) -> dict:
    """Use LLM to evaluate the agent's response."""
    model = os.environ.get("LLM_JUDGE_MODEL", "openai/claude-sonnet")
    api_base = os.environ.get("LLM_API_BASE", "http://localhost:4000")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-dummy")

    try:
        result = litellm.completion(
            model=model,
            api_base=api_base,
            api_key=api_key,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Agent's response to evaluate:\n\n{response_text}"},
            ],
            temperature=0.0,
            max_tokens=500,
        )

        content = result.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        return json.loads(content)

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM judge response: {e}", file=sys.stderr)
        print(f"Raw response: {content}", file=sys.stderr)
        return {"overall_score": 0.5, "reasoning": "Failed to parse judge response"}

    except Exception as e:
        print(f"LLM judge error: {e}", file=sys.stderr)
        return {"overall_score": 0.0, "reasoning": f"Judge error: {str(e)}"}


def main():
    if len(sys.argv) < 3:
        print("Usage: llm_judge.py <response_file> <reward_file>", file=sys.stderr)
        sys.exit(1)

    response_file = sys.argv[1]
    reward_file = sys.argv[2]

    with open(response_file) as f:
        response_text = f.read()

    print(f"Evaluating response ({len(response_text)} chars)...")

    evaluation = evaluate_response(response_text)

    print(f"Evaluation result: {json.dumps(evaluation, indent=2)}")

    score = evaluation.get("overall_score", 0.0)
    score = max(0.0, min(1.0, float(score)))

    with open(reward_file, "w") as f:
        f.write(str(score))

    print(f"Reward: {score}")

    details_file = reward_file.replace("reward.txt", "evaluation.json")
    with open(details_file, "w") as f:
        json.dump(evaluation, f, indent=2)


if __name__ == "__main__":
    main()
