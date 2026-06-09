# A2A Agent Evaluation Example

This example demonstrates how to evaluate an A2A-compliant agent (like `google-lightspeed-agent`) using Harbor and the ABEvalFlow pipeline.

## Prerequisites

1. **Deployed A2A Agent**: The agent must be deployed and accessible via HTTP(S)
2. **LiteLLM Proxy**: For the LLM-as-judge verifier to work
3. **Harbor**: Installed locally or available in the pipeline

## Quick Start

### Local Testing with Harbor

```bash
# Set the agent endpoint
export A2A_ENDPOINT="https://lightspeed-agent-lightspeed-agent-dev.apps.cn-ai-lab.2vn8.p1.openshiftapps.com"

# Run evaluation with Harbor
harbor run \
  -p examples/a2a-agent-eval/tasks/lightspeed-qa \
  --agent-import-path abevalflow.harbor_agents.a2a_adapter:A2AAgent \
  --ak endpoint=$A2A_ENDPOINT \
  --ak timeout=120 \
  -e podman \
  --n-attempts 3
```

### Via ABEvalFlow Pipeline (Tekton)

```bash
tkn pipeline start abevalflow-ci-pipeline \
  --param eval-engine=a2a \
  --param agent-endpoint=$A2A_ENDPOINT \
  --param submission-name=lightspeed-qa-eval
```

## Task Structure

```
tasks/lightspeed-qa/
├── task.toml           # Harbor task configuration
├── instruction.md      # Question/prompt for the agent
├── tests/
│   └── llm_judge.py    # LLM-as-judge verifier
└── expected/           # (Optional) expected outputs
```

### task.toml

Configures the Harbor task metadata and verifier location:

```toml
[task]
name = "lightspeed-qa-basic"
version = "1.0.0"
description = "Basic Q&A evaluation for lightspeed-agent"

[test]
file = "tests/llm_judge.py"
function = "grade"
```

### instruction.md

The prompt/question sent to the agent. This is read by the A2A adapter
and sent as the message content to the agent.

### tests/llm_judge.py

The verifier that scores the agent's response. It must implement a
`grade()` function that returns:

```python
{
    "reward": 0.85,  # Float between 0.0 and 1.0
    "details": {
        "criteria_scores": {...},
        "overall_explanation": "..."
    }
}
```

## Creating Custom Tasks

1. Copy `tasks/lightspeed-qa/` to a new directory
2. Update `task.toml` with new name/description
3. Modify `instruction.md` with your evaluation prompt
4. Customize `tests/llm_judge.py` criteria if needed

### Using Rule-Based Verification

For tasks with deterministic expected outputs, you can use rule-based
verification instead of LLM-as-judge:

```python
# tests/test_response.py
def grade() -> dict:
    response = Path("a2a_response.json").read_text()
    data = json.loads(response)
    
    result = data.get("result", {})
    artifacts = result.get("artifacts", [])
    
    # Check for expected content
    has_expected = any(
        "expected_keyword" in str(a) 
        for a in artifacts
    )
    
    return {
        "reward": 1.0 if has_expected else 0.0,
        "details": {"has_expected": has_expected}
    }
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_JUDGE_MODEL` | Model for LLM-as-judge | `openai/claude-sonnet` |
| `LLM_BASE_URL` | LiteLLM proxy URL | `http://litellm.ab-eval-flow.svc.cluster.local:4000` |
| `A2A_ENDPOINT` | Agent endpoint URL | (required) |

## Troubleshooting

### Agent timeout

Increase the timeout parameter:
```bash
--ak timeout=300
```

### Connection refused

Verify the agent is running and accessible:
```bash
curl -X POST $A2A_ENDPOINT \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"agent/state","params":{},"id":"1"}'
```

### LLM judge errors

Check LiteLLM proxy connectivity and model availability:
```bash
curl http://litellm.ab-eval-flow.svc.cluster.local:4000/v1/models
```
