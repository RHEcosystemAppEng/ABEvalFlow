# Task: Red Hat Lightspeed Agent Evaluation (Konflux)

You are evaluating the Red Hat Lightspeed Agent, an AI assistant specialized
in Red Hat Insights and advisory services. This evaluation runs as a Konflux
IntegrationTestScenario against the freshly-built container image.

Conduct the following conversation with the agent, sending each message in
sequence and recording all responses.

## Conversation Steps

### Step 1 — Advisor Capabilities
Ask the agent:
> "What can Red Hat Insights Advisor help me with? Please give me a few specific examples."

**Expected**: The agent lists concrete Insights Advisor capabilities (e.g., identifying configuration risks, patch recommendations, CVE exposure, compliance checks).

### Step 2 — Domain-Specific Advisory Query
Ask the agent:
> "I have a RHEL 8 system running kernel 4.18.0-305. Are there any known CVEs or critical patches I should apply?"

**Expected**: The agent provides relevant CVE/patch information for the specified RHEL version, or clearly explains what information it would need to look this up via Insights.

### Step 3 — Graceful Degradation
Ask the agent:
> "Can you help me deploy a Kubernetes cluster on AWS?"

**Expected**: The agent gracefully declines or redirects, making clear this is outside its scope (Red Hat Insights / advisory domain), without crashing or giving a nonsensical answer.

### Step 4 — Protocol Compliance
Ask the agent:
> "Summarize what we discussed in this conversation."

**Expected**: The agent provides a coherent summary of the prior conversation turns, demonstrating context retention and proper A2A multi-turn handling.

## Evaluation Criteria

The LLM judge will score the agent's responses across all four steps:
- **Advisor knowledge**: accurate and specific Insights Advisor capabilities
- **Domain query handling**: relevant CVE/patch guidance or clear data-requirement explanation
- **Graceful degradation**: politely declines out-of-scope requests without failure
- **Protocol compliance**: maintains context across turns, coherent multi-turn response
