#!/bin/bash
set -e

# Run LLM judge verifier on the agent's response
# The A2A adapter writes the response to /logs/agent/a2a_response.txt

RESPONSE_FILE="/logs/agent/a2a_response.txt"
REWARD_FILE="/logs/verifier/reward.txt"

# Ensure logs directory exists
mkdir -p /logs/verifier

# Check if response file exists
if [ ! -f "$RESPONSE_FILE" ]; then
    echo "ERROR: Agent response file not found at $RESPONSE_FILE"
    echo "0" > "$REWARD_FILE"
    exit 1
fi

# Run LLM judge
python3 /tests/llm_judge.py "$RESPONSE_FILE" "$REWARD_FILE"

# Exit with success (reward file contains the score)
exit 0
