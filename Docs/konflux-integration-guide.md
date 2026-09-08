# Agentic Eval Flow Konflux Integration Guide

This guide explains how to integrate Agentic Eval Flow evaluation into any Konflux
application pipeline. Agentic Eval Flow provides generic evaluation tasks as Tekton
Bundles that can evaluate A2A agents, MCP servers, and skills.

## Architecture

Agentic Eval Flow publishes **8 core tasks** as Tekton Bundles:

```
parse-snapshot → prepare → test → [red-team] → evaluate → analyze-scorecard → store → emit-result
```

The `red-team` task is opt-in (controlled by `ENABLE_RED_TEAM` parameter, default `false`).

These tasks handle the entire evaluation lifecycle:

| Task | Purpose |
|------|---------|
| `parse-snapshot` | Extract component image and git info from a Konflux Snapshot |
| `prepare` | Clone and validate the submission definition |
| `test` | Run security scans and quality review (optional) |
| `evaluate` | Execute the evaluation engine (A2A, MCPChecker, Harbor, ASE) |
| `analyze-scorecard` | Produce a certification scorecard from results |
| `store` | Persist results to PostgreSQL/MinIO (optional) |
| `emit-result` | Map scorecard to Konflux's `TEST_OUTPUT` format |

Your application adds its own deployment and cleanup logic around these core
tasks. Agentic Eval Flow never deploys or manages your application.

## Parameter Contract

### Pipeline Parameters

```yaml
# Required by Konflux (provided automatically)
SNAPSHOT: "<Konflux Snapshot JSON>"

# What to evaluate
EVAL_ENGINE: "a2a"              # a2a | harbor | ase | mcpchecker
SUBMISSION_REPO_URL: ""         # Git repo containing the submission definition
SUBMISSION_DIR: ""              # Directory name under submissions/
SUBMISSION_REVISION: "main"     # Git ref for the submission repo

# Target endpoints (provide based on engine)
AGENT_ENDPOINT: ""              # Required for a2a: HTTP endpoint of the agent
MCP_URL: ""                     # Required for mcpchecker: URL of the MCP server

# LLM infrastructure (for judging)
LLM_API_BASE: ""                # LLM proxy URL (e.g. http://litellm.ns.svc:4000)
LLM_MODEL: "claude-sonnet"     # Model name for LLM-as-judge

# Execution mode
EVAL_MODE: "local"              # "local" or "remote"
WORKLOAD_CLUSTER_URL: ""        # Required when EVAL_MODE=remote
WORKLOAD_NAMESPACE: ""          # Required when EVAL_MODE=remote
WORKLOAD_CREDENTIALS_SECRET: "workload-cluster-credentials"  # Secret name

# Pipeline repo (for evaluation scripts)
PIPELINE_REPO_URL: "https://github.com/RHEcosystemAppEng/agentic_eval_flow.git"
PIPELINE_REPO_REVISION: "main"
```

### When to Use Each Parameter

| Eval Engine | Required Parameters |
|-------------|-------------------|
| `a2a` | `AGENT_ENDPOINT`, `SUBMISSION_*`, `LLM_*` |
| `mcpchecker` | `MCP_URL`, `SUBMISSION_*`, `LLM_*` |
| `harbor` | `SUBMISSION_*`, `LLM_*` |
| `ase` | `SUBMISSION_*`, `LLM_*` |

### Engine x Mode Validation Matrix

| Engine | Local | Remote |
|--------|-------|--------|
| `a2a` | Supported | **Fully tested** (E2E on Konflux) |
| `mcpchecker` | Supported | Supported (untested) |
| `ase` | Supported (no external endpoint needed) | Not yet implemented |
| `harbor` | Limited (local environment only, no scaffold/build) | Not supported (use standalone pipeline) |

For `harbor` in Konflux, the local mode runs with `environment.type: local` which
does not perform the full scaffold/build/eval cycle. For full Harbor A/B testing
with container registry support, use the standalone Agentic Eval Flow pipeline on OpenShift.

### Multi-component Applications

The `parse-snapshot` task defaults to `.components[0]` from the Snapshot. For
applications with multiple components, set the `component-name` parameter in
parse-snapshot to target the specific component you want to evaluate. Failing
to do so may result in evaluating the wrong component image.

## Evaluation Modes

### Local Mode (`EVAL_MODE=local`)

The evaluation runs directly inside the Tekton task step on the pipeline
cluster. Use this when:

- The target (agent/MCP server) is reachable from the pipeline cluster
- The target has a public Route or Ingress
- You're evaluating skills (Harbor/ASE) that don't need an external endpoint

No workload cluster credentials are needed in this mode.

### Remote Mode (`EVAL_MODE=remote`)

The evaluation runs as a Pod on a separate workload cluster. Use this when:

- The pipeline cluster (Konflux) can't reach the target's cluster-internal Services
- The target is deployed on a different cluster from where the pipeline runs
- You need the eval Pod co-located with the target for network access

Required in this mode:
- `WORKLOAD_CLUSTER_URL` -- API URL of the workload cluster
- `WORKLOAD_NAMESPACE` -- Namespace to create the eval Pod in
- A Secret (named by `WORKLOAD_CREDENTIALS_SECRET`) with a `token` key containing
  a ServiceAccount token for the workload cluster

## Submissions

A **submission** is the evaluation definition package. It tells Agentic Eval Flow what
to test and how to judge results. Structure depends on the eval engine:

### A2A Agent Submission

```
submission/
  metadata.yaml          # name, eval_engine: a2a, experiment config
  tasks/
    <task-name>/
      task.toml          # Harbor task configuration
      instruction.md     # Multi-turn conversation instructions
      tests/test.sh      # Verifier entry point
      tests/llm_judge.py # LLM-as-judge scorer
      environment/Dockerfile
```

### MCP Server Submission

```
submission/
  metadata.yaml          # name, eval_engine: mcpchecker
  eval.yaml              # MCPChecker evaluation config
  mcp-config.yaml        # MCP server connection config (can use $MCP_URL)
```

### Skill Submission (ASE)

```
submission/
  metadata.yaml          # name, eval_engine: ase
  skills/
    <skill-name>/
      SKILL.md           # Skill definition
      evals/evals.json   # Evaluation scenarios
```

### Skill Submission (Harbor)

```
submission/
  metadata.yaml          # name, eval_engine: harbor
  tasks/
    <task-name>/
      task.toml
      instruction.md
      tests/test.sh
      environment/Dockerfile
```

### metadata.yaml Reference

```yaml
name: my-evaluation              # Unique evaluation name
description: What this evaluates
version: "1.0.0"
eval_engine: a2a                  # a2a | harbor | ase | mcpchecker

experiment:
  n_trials: 5                    # Number of evaluation trials

security_scan: disabled           # disabled | warn | block
skip_quality_review: true         # Skip LLM quality review

gate_policy:                      # Optional certification gates
  default_mode: warn
  combination: all_pass
  gates:
    evaluation:
      mode: block
      threshold: 0.0
```

Submissions can live in any Git repository. The pipeline accepts
`SUBMISSION_REPO_URL` and `SUBMISSION_DIR` to locate them.

## Integration Patterns

### Pattern 1: Pre-deployed Target (simplest)

If your agent or MCP server is already running (e.g., a long-lived service),
use Agentic Eval Flow's reference pipeline directly:

```yaml
apiVersion: appstudio.redhat.com/v1beta2
kind: IntegrationTestScenario
metadata:
  name: abevalflow-eval
  namespace: <your-tenant-namespace>
  labels:
    test.appstudio.openshift.io/optional: "true"
spec:
  application: <your-app-name>
  contexts:
    - description: AI evaluation via Agentic Eval Flow
      name: application
  resolverRef:
    resolver: git
    resourceKind: pipelinerun
    params:
      - name: url
        value: https://github.com/RHEcosystemAppEng/agentic_eval_flow
      - name: revision
        value: main
      - name: pathInRepo
        value: pipeline/integration/konflux-eval-pipelinerun.yaml
  params:
    - name: EVAL_ENGINE
      value: "a2a"
    - name: AGENT_ENDPOINT
      value: "http://my-agent.my-namespace.svc:8000"
    - name: SUBMISSION_REPO_URL
      value: "https://github.com/myorg/my-submissions.git"
    - name: SUBMISSION_DIR
      value: "my-agent-eval"
    - name: LLM_API_BASE
      value: "http://litellm.my-namespace.svc:4000"
```

### Pattern 2: Pipeline-deployed Target

If your target needs to be deployed for each evaluation run, create your own
pipeline that wraps Agentic Eval Flow's core tasks with deploy/cleanup steps.

See the [full working example](https://github.com/ikrispin/abevalflow-konflux-example)
for the Google Lightspeed Agent.

Key steps:
1. Create a `deploy-<app>.yaml` task that deploys your application and outputs
   the endpoint URL as a task result
2. Create a `cleanup-<app>.yaml` task for the `finally:` block
3. Create a PipelineRun that chains: deploy → Agentic Eval Flow core tasks → cleanup
4. Create a submission definition for your evaluation scenarios
5. Create an IntegrationTestScenario pointing to your pipeline

### Pattern 3: MCP Server Evaluation

```yaml
# In your IntegrationTestScenario params:
- name: EVAL_ENGINE
  value: "mcpchecker"
- name: MCP_URL
  value: "http://my-mcp-server.my-namespace.svc:3000"
- name: SUBMISSION_REPO_URL
  value: "https://github.com/myorg/my-submissions.git"
- name: SUBMISSION_DIR
  value: "my-mcp-server-eval"
```

## Secrets

### Required Secrets by Mode

| Secret | When Required |
|--------|--------------|
| `workload-cluster-credentials` | `EVAL_MODE=remote` only |
| `llm-credentials` | When LLM proxy needs a real API key |

### Optional Secrets

| Secret | Purpose |
|--------|---------|
| `compass-facts-api` | Push scorecard facts to Red Hat Compass |
| `ab-eval-db-credentials` | Store results in PostgreSQL |
| `minio-credentials` | Upload artifacts to MinIO/S3 |
| `monitoring-slack-webhook` | Send degradation alerts to Slack |
| `a2a-agent-credentials` | Bearer token for JWT-protected A2A agents (`eval-engine=a2a` only) |

### Creating Workload Cluster Credentials

On the workload cluster:
```bash
oc create sa abevalflow-deployer -n <your-namespace>
oc adm policy add-role-to-user edit -z abevalflow-deployer -n <your-namespace>
oc create token abevalflow-deployer -n <your-namespace> --duration=8760h
```

Store the token in your Konflux tenant namespace:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: workload-cluster-credentials
  namespace: <your-tenant-namespace>
type: Opaque
stringData:
  token: "<token-from-above>"
```

See `config/konflux/secrets-template.yaml` for the full template.

### Creating A2A Agent Credentials

Only needed when `eval-engine=a2a` and the target agent requires a Bearer
token (e.g. JWT-protected endpoints). If this secret doesn't exist, the
`agent-auth-token-secret` lookup is optional and no `Authorization` header
is sent — existing no-auth agents are unaffected.

```bash
oc create secret generic a2a-agent-credentials \
  --from-literal=token="<your-agent-jwt>" \
  -n <your-tenant-namespace>
```

If your secret has a different name, pass it via the `agent-auth-token-secret`
pipeline parameter (default: `a2a-agent-credentials`).

## Tekton Bundles

The core tasks are published as Tekton Bundles to Quay.io:

| Bundle | Task |
|--------|------|
| `quay.io/rh-ee-ikrispin/abevalflow-task-parse-snapshot:0.1` | parse-snapshot |
| `quay.io/rh-ee-ikrispin/abevalflow-task-prepare:0.1` | prepare |
| `quay.io/rh-ee-ikrispin/abevalflow-task-test:0.1` | test |
| `quay.io/rh-ee-ikrispin/abevalflow-task-evaluate:0.1` | evaluate |
| `quay.io/rh-ee-ikrispin/abevalflow-task-analyze-scorecard:0.1` | analyze-scorecard |
| `quay.io/rh-ee-ikrispin/abevalflow-task-store:0.1` | store |
| `quay.io/rh-ee-ikrispin/abevalflow-task-red-team:0.1` | red-team (opt-in) |
| `quay.io/rh-ee-ikrispin/abevalflow-task-emit-result:0.1` | emit-result |

To rebuild bundles after editing task YAML:
```bash
cd pipeline/integration
make bundles
```

## Quick Start

1. **Choose your pattern** from the Integration Patterns section above
2. **Create a submission** defining your evaluation scenarios
3. **Provision secrets** in your Konflux tenant namespace
4. **Create an IntegrationTestScenario** in your tenant namespace
5. **Push a change** to your application -- Konflux triggers the evaluation

For a complete working example, see:
[github.com/ikrispin/abevalflow-konflux-example](https://github.com/ikrispin/abevalflow-konflux-example)
