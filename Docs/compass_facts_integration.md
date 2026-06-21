# Compass Facts API Integration

ABEvalFlow can push gate evaluation results to Red Hat Compass as Soundcheck facts.
This enables visibility of skill evaluation metrics directly in the Compass developer portal.

## Overview

When configured, ABEvalFlow will POST gate results to the Compass Facts API after each
gate evaluation completes. This provides real-time visibility into:

- Engine gate results (Harbor, ASE, A2A, MCPChecker)
- Security gate results (Cisco scanner)
- Quality gate results (LLM review)

## Configuration

Add `push_facts` configuration to your `metadata.yaml`:

```yaml
schema_version: "1.0"
name: my-skill
eval_engine: harbor

gate_policy:
  default_mode: warn
  combination: all_pass
  
  # Compass Facts API configuration
  push_facts:
    endpoint: https://compass.stage.redhat.com/api/soundcheck/facts/
    entity_ref: component:default/my-skill
    fact_ref_prefix: catalog:default/abevalflow_
  
  gates:
    harbor:
      mode: block
      threshold: 0.0
      push_fact: true    # Enable fact pushing for this gate
    cisco:
      mode: block
      push_fact: true    # Enable fact pushing for this gate
    llm-review:
      mode: warn
      push_fact: false   # Disable (default)
```

## Configuration Fields

### `push_facts` (optional)

Top-level configuration for the Facts API. When omitted or `null`, fact pushing is disabled.

| Field | Required | Description |
|-------|----------|-------------|
| `endpoint` | Yes | Compass Facts API URL |
| `entity_ref` | Yes | Compass entity reference (e.g., `component:default/my-skill`) |
| `fact_ref_prefix` | No | Prefix for fact references (default: `catalog:default/abevalflow_`) |

### Per-gate `push_fact` flag

Each gate in `gates` can have a `push_fact: true/false` flag:

- `true`: Push this gate's result to Compass after evaluation
- `false` (default): Do not push this gate's result

## Fact Payload Structure

Each gate result is pushed as a Soundcheck fact with this structure:

```json
{
  "facts": [
    {
      "factRef": "catalog:default/abevalflow_harbor",
      "entityRef": "component:default/my-skill",
      "data": {
        "gate_name": "harbor",
        "passed": true,
        "score": 0.85,
        "mode": "block",
        "message": "Harbor evaluation passed with uplift 0.15",
        "details": {
          "uplift": 0.15,
          "treatment_pass_rate": 0.9,
          "control_pass_rate": 0.75
        },
        "evaluated_at": "2026-06-21T12:00:00+00:00"
      }
    }
  ]
}
```

## Validation Warning

If `push_facts.endpoint` is configured but no gates have `push_fact: true`,
ABEvalFlow logs a warning:

```
WARNING: push_facts.endpoint is configured but no gates have push_fact=True.
No facts will be pushed. Add push_fact: true to gate configurations.
```

## Environments

| Environment | Endpoint |
|-------------|----------|
| Staging | `https://compass.stage.redhat.com/api/soundcheck/facts/` |
| Production | `https://compass.redhat.com/api/soundcheck/facts/` |

## Authentication

The Facts API may require authentication depending on your Compass configuration.
Contact the Compass team for authentication requirements specific to your organization.

## Error Handling

- If a fact push fails, ABEvalFlow logs a warning but continues processing
- Gate evaluation results are not affected by fact push failures
- Timeouts default to 30 seconds per push request

## Related

- [Scorecard Documentation](./scorecard.md)
- [Gate Policy Configuration](./gate_policy.md)
- [APPENG-5306](https://redhat.atlassian.net/browse/APPENG-5306) - Jira ticket
