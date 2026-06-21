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
    bearer_token: ${COMPASS_API_TOKEN}  # from env/secret
  
  gates:
    evaluation:          # category name (engine selected by eval-engine param)
      mode: block
      threshold: 0.0
      push_fact: true    # Enable fact pushing for this gate
    security:            # category name (uses cisco scanner)
      mode: block
      push_fact: true    # Enable fact pushing for this gate
    quality:             # category name (uses llm-review)
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
| `bearer_token` | No | Bearer token for API authentication. Supports env var substitution. |

### Per-gate `push_fact` flag

Each gate in `gates` can have a `push_fact: true/false` flag:

- `true`: Push this gate's result to Compass after evaluation
- `false` (default): Do not push this gate's result

## Gate Naming Convention

Gate results use category-based naming in Compass with implementation details included:

- **evaluation**: All evaluation engines (Harbor, ASE, A2A, MCPChecker)
- **security**: All security scanners (Cisco, Snyk, etc.)
- **quality**: All quality reviewers (LLM review)

The fact reference combines the category and implementation:
- `abevalflow_evaluation_harbor`
- `abevalflow_security_cisco`
- `abevalflow_quality_llm-review`

## Fact Payload Structure

Each gate result is pushed as a Soundcheck fact with this structure:

```json
{
  "facts": [
    {
      "factRef": "catalog:default/abevalflow_evaluation_harbor",
      "entityRef": "component:default/my-skill",
      "data": {
        "gate_name": "evaluation",
        "passed": true,
        "score": 0.85,
        "mode": "block",
        "message": "Harbor evaluation passed with uplift 0.15",
        "details": {
          "engine": "harbor",
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

The Compass Facts API requires authentication via bearer token. Configure the token in one of these ways:

### Option 1: Environment Variable (Recommended for Tekton)

Set the bearer token via environment variable and reference it in your config:

```yaml
push_facts:
  endpoint: https://compass.stage.redhat.com/api/soundcheck/facts/
  entity_ref: component:default/my-skill
  bearer_token: ${COMPASS_API_TOKEN}
```

In your Tekton Pipeline, mount the token from a Kubernetes Secret:

```yaml
env:
  - name: COMPASS_API_TOKEN
    valueFrom:
      secretKeyRef:
        name: compass-api-credentials
        key: token
```

Create the secret:

```bash
kubectl create secret generic compass-api-credentials \
  --namespace ab-eval-flow \
  --from-literal=token=your-compass-api-token
```

### Option 2: Direct Token (Development Only)

For local development, you can set the token directly (not recommended for production):

```yaml
push_facts:
  bearer_token: your-compass-api-token
```

### Obtaining a Token

Contact the Compass team for authentication credentials specific to your organization.
The token is typically issued per-service or per-team.

## Network Requirements

The Tekton task running the scorecard aggregation must be able to reach the Compass API endpoints.

### Required Connectivity

| Environment | Hostname | Port |
|-------------|----------|------|
| Staging | `compass.stage.redhat.com` | 443 |
| Production | `compass.redhat.com` | 443 |

### Egress NetworkPolicy

If your OpenShift cluster uses NetworkPolicies to restrict egress traffic, you may need to allow
outbound connections to Compass endpoints. Example NetworkPolicy:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-compass-egress
  namespace: ab-eval-flow
spec:
  podSelector:
    matchLabels:
      tekton.dev/taskRun: scorecard-aggregation
  policyTypes:
    - Egress
  egress:
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0  # Compass IPs may vary; use specific CIDRs if known
      ports:
        - protocol: TCP
          port: 443
```

### Proxy Configuration

If your cluster routes external traffic through an egress proxy, ensure the proxy allows
connections to `compass.stage.redhat.com` and `compass.redhat.com`.

Common proxy environment variables:

```yaml
env:
  - name: HTTPS_PROXY
    value: http://egress-proxy.example.com:8080
  - name: NO_PROXY
    value: .cluster.local,.svc,localhost
```

### DNS Resolution

If you encounter `[Errno -2] Name or service not known` errors, verify:

1. The cluster can resolve `compass.stage.redhat.com` via DNS
2. CoreDNS or your DNS provider is configured correctly
3. No DNS-based blocking is in place for external domains

Test from a debug pod:

```bash
kubectl run dns-test --rm -it --restart=Never --image=busybox -- nslookup compass.stage.redhat.com
```

## Error Handling

- If a fact push fails, ABEvalFlow logs a warning but continues processing
- Gate evaluation results are not affected by fact push failures
- Timeouts default to 30 seconds per push request
- Authentication failures (401/403) are logged with the specific error

## Related

- [Scorecard Documentation](./scorecard.md)
- [Gate Policy Configuration](./gate_policy.md)
- [APPENG-5306](https://redhat.atlassian.net/browse/APPENG-5306) - Jira ticket
