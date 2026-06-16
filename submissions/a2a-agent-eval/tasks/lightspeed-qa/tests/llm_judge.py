#!/usr/bin/env python3
"""LLM Judge verifier for A2A multi-turn Lightspeed agent responses.

Evaluates a full multi-turn conversation using an LLM-as-judge approach.
Scores four dimensions and writes a weighted overall score to the reward file.
"""

import json
import os
import sys

import litellm

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for A2A-compliant AI agent conversations.

Your task is to evaluate a multi-turn conversation between a user and the Red Hat Lightspeed Agent —
an A2A agent specialized in Red Hat Insights and advisory services.

The conversation should contain four sequential steps:

1. **Advisor Capabilities**: User asks what Red Hat Insights Advisor can help with (specific examples expected).
2. **Domain-Specific Advisory Query**: User asks about CVEs or critical patches for RHEL 8 kernel 4.18.0-305.
3. **Graceful Degradation**: User asks to deploy a Kubernetes cluster on AWS (out of scope).
4. **Protocol Compliance**: User asks the agent to summarize the prior conversation.

The agent uses mock SSO (SKIP_JWT_VALIDATION=true). Live Red Hat inventory and
authentication-gated advisories may be unavailable. Score on response quality and domain
knowledge within those constraints, not on whether live system data was retrieved.

Evaluate the full conversation on these criteria (each scored 0-1):

1. **advisor_knowledge** (weight 0.30): Did the agent accurately describe Insights Advisor capabilities with specific examples?
   - 1.0: Lists concrete capabilities (configuration risks, patch recommendations, CVE exposure, compliance checks)
   - 0.5: Partially correct but vague or missing key Advisor aspects
   - 0.0: Incorrect, off-topic, or refuses to describe Advisor capabilities

2. **domain_query_handling** (weight 0.25): Did the agent handle the RHEL 8 CVE/patch query appropriately?
   - 1.0: Provides relevant CVE/patch guidance for the specified RHEL version, or clearly explains what data is needed via Insights
   - 0.5: Generic security advice without RHEL 8 / kernel specificity, or weak explanation of data requirements
   - 0.0: No useful guidance, wrong domain handling, or completely off-topic

3. **graceful_degradation** (weight 0.25): Did the agent gracefully decline the out-of-scope Kubernetes-on-AWS request?
   - 1.0: Politely declines and redirects to Red Hat Insights / advisory capabilities
   - 0.5: Declines but weak redirect, or partially attempts the out-of-scope task
   - 0.0: Attempts to deploy Kubernetes, hallucinates AWS steps, or crashes

4. **protocol_compliance** (weight 0.20): Did the agent maintain context and summarize the conversation coherently?
   - 1.0: Accurate, coherent summary of all prior turns demonstrating multi-turn context retention
   - 0.5: Partial summary missing key turns or lacking coherence
   - 0.0: No summary, wrong context, or empty response

Respond ONLY with valid JSON in this exact format:
{
    "advisor_knowledge": <float 0-1>,
    "domain_query_handling": <float 0-1>,
    "graceful_degradation": <float 0-1>,
    "protocol_compliance": <float 0-1>,
    "overall_score": <float 0-1>,
    "reasoning": "<brief explanation covering all four dimensions>"
}

The overall_score MUST be the weighted average:
(advisor_knowledge*0.30 + domain_query_handling*0.25 + graceful_degradation*0.25 + protocol_compliance*0.20)
"""


def evaluate_response(response_text: str) -> dict:
    """Use LLM to evaluate the agent's multi-turn conversation."""
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
                {"role": "user", "content": f"Full conversation to evaluate:\n\n{response_text}"},
            ],
            temperature=0.0,
            max_tokens=800,
        )

        content = result.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        evaluation = json.loads(content)

        weights = {
            "advisor_knowledge": 0.30,
            "domain_query_handling": 0.25,
            "graceful_degradation": 0.25,
            "protocol_compliance": 0.20,
        }
        weighted = sum(
            float(evaluation.get(dim, 0.0)) * weight for dim, weight in weights.items()
        )
        evaluation["overall_score"] = round(weighted, 4)

        return evaluation

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

    print(f"Evaluating multi-turn conversation ({len(response_text)} chars)...")

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
