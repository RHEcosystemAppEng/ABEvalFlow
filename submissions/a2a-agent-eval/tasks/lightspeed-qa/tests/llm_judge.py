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

Your task is to evaluate an AI agent's response to a capability discovery question.
The user asked: "What capabilities do you have? Please list the main things you can help me with."

Evaluate the response on these criteria (each scored 0-1):

1. **Relevance** (0-1): Does the response address the question about capabilities?
   - 1.0: Directly answers what the agent can do
   - 0.5: Partially addresses capabilities
   - 0.0: Off-topic or refuses to answer

2. **Completeness** (0-1): Are capabilities clearly listed?
   - 1.0: Lists multiple specific capabilities with examples
   - 0.5: Lists some capabilities but vague
   - 0.0: No capabilities listed

3. **Clarity** (0-1): Is the response well-organized and easy to understand?
   - 1.0: Well-structured, uses formatting (lists, headers)
   - 0.5: Readable but could be better organized
   - 0.0: Confusing or poorly formatted

4. **Honesty** (0-1): Does the agent accurately represent its abilities?
   - 1.0: Clear about what it can and cannot do
   - 0.5: Somewhat clear about limitations
   - 0.0: Misleading or no mention of limitations

Respond ONLY with valid JSON in this exact format:
{
    "relevance": <float 0-1>,
    "completeness": <float 0-1>,
    "clarity": <float 0-1>,
    "honesty": <float 0-1>,
    "overall_score": <float 0-1>,
    "reasoning": "<brief explanation>"
}

The overall_score should be the weighted average: (relevance*0.3 + completeness*0.3 + clarity*0.2 + honesty*0.2)
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
