"""LLM-as-Judge verifier for lightspeed-qa task.

This verifier uses an LLM to evaluate the agent's response on
correctness, helpfulness, and completeness.
"""

from pathlib import Path
from typing import Any

from abevalflow.harbor_agents.verifiers.llm_judge import LLMJudgeVerifier


EVALUATION_CRITERIA = ["correctness", "helpfulness", "completeness", "clarity"]

INSTRUCTION_FILE = Path(__file__).parent.parent / "instruction.md"


def grade(response_file: str | None = None) -> dict[str, Any]:
    """Grade the agent's response using LLM-as-judge.

    This function is called by Harbor after the agent completes its task.

    Args:
        response_file: Path to the agent's response file.
                      If not provided, looks for a2a_response.txt in workspace.

    Returns:
        Dict with 'reward' (float 0-1) and 'details' (evaluation breakdown).
    """
    if response_file:
        response_text = Path(response_file).read_text()
    else:
        workspace_response = Path("a2a_response.txt")
        agent_dir_response = Path("/home/agent/a2a_response.txt")

        if workspace_response.exists():
            response_text = workspace_response.read_text()
        elif agent_dir_response.exists():
            response_text = agent_dir_response.read_text()
        else:
            return {
                "reward": 0.0,
                "details": {"error": "No agent response file found"},
            }

    instruction = None
    if INSTRUCTION_FILE.exists():
        instruction = INSTRUCTION_FILE.read_text()

    verifier = LLMJudgeVerifier(
        criteria=EVALUATION_CRITERIA,
        instruction=instruction,
    )

    return verifier.grade(response_text)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1:
        result = grade(response_file=sys.argv[1])
    else:
        result = grade()

    print(json.dumps(result, indent=2))
