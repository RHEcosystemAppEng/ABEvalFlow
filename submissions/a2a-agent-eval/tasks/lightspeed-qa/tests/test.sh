#!/bin/bash
set -e

# Run LLM judge verifier on the agent's response
# Supports both container paths (/logs/...) and local environment paths

# Try container paths first, fall back to CWD-relative
if [ -d "/logs" ]; then
    LOGS_BASE="/logs"
    TESTS_BASE="/tests"
else
    LOGS_BASE="$(pwd)/logs"
    TESTS_BASE="$(pwd)/tests"
fi

RESPONSE_FILE="${LOGS_BASE}/agent/a2a_response.txt"
REWARD_FILE="${LOGS_BASE}/verifier/reward.txt"

# Ensure directories exist
mkdir -p "${LOGS_BASE}/verifier"

echo "Looking for response at: $RESPONSE_FILE"
echo "Will write reward to: $REWARD_FILE"

# Check if response file exists
if [ ! -f "$RESPONSE_FILE" ]; then
    echo "ERROR: Agent response file not found at $RESPONSE_FILE"
    ls -la "${LOGS_BASE}/" 2>/dev/null || echo "Logs base doesn't exist"
    ls -la "${LOGS_BASE}/agent/" 2>/dev/null || echo "Agent dir doesn't exist"
    echo "0" > "$REWARD_FILE"
    exit 0  # Exit success but with 0 reward
fi

echo "Response file found, running LLM judge..."

# Run LLM judge
python3 "${TESTS_BASE}/llm_judge.py" "$RESPONSE_FILE" "$REWARD_FILE"

echo "LLM judge completed"
exit 0
