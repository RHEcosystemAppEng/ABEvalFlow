# Manual Trigger Guide — Harbor, ASE & A2A

Quick reference for manually triggering evaluations against the CI and monitoring pipelines.

---

## A2A Monitoring Trigger Sources

A2A monitoring runs are triggered automatically by three sources (plus manual PipelineRuns):

| Source | When it fires | Agent mode | Key params |
|--------|---------------|------------|------------|
| **Quay push webhook** | New image pushed to `quay.io/ecosystem-appeng/google-lightspeed-agent` | Ephemeral deploy | `agent-image`, `agent-tag`; leave `agent-endpoint` empty |
| **LiteLLM config change** | Push to `main` on `RHEcosystemAppEng/ABEvalFlow` modifying `config/litellm/*` | Existing deployed agent | `agent-endpoint` only |
| **10-day CronJob** | Scheduled every 10 days (`0 6 */10 * *`) | Existing deployed agent | `agent-endpoint` from canary pack config |

**Quay push:** EventListener trigger `quay-push-trigger` deploys a temporary instance of the pushed image, evaluates it, then cleans up. Requires a Quay repository notification pointing at the EventListener URL (pending Ilona access).

**LiteLLM config change:** EventListener trigger `litellm-config-push-trigger` re-tests against the existing agent at `http://lightspeed-agent.ab-eval-flow.svc:8000`. Requires a GitHub webhook on `RHEcosystemAppEng/ABEvalFlow` pointing at the EventListener URL.

**10-day CronJob:** The `abevalflow-monitoring` CronJob performs a health check on `/.well-known/agent.json` before triggering; skips eval if the agent is unhealthy.

**Pre-flight health check:** Every A2A pipeline run includes an `a2a-pre-flight` step (between deploy and eval) that verifies `/.well-known/agent.json` returns HTTP 200. The run fails fast if the agent is unreachable.

**Manual A2A params:**
- **Existing agent:** set `agent-endpoint` (e.g. `http://lightspeed-agent.ab-eval-flow.svc:8000`)
- **Ephemeral deploy:** set `agent-image` + `agent-tag`, leave `agent-endpoint` empty

See [A2A Trigger Types](#a2a-trigger-types) below for full PipelineRun examples.

**Working examples — Monitoring pipeline:**
- Harbor: https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/harbor-verify-t6spp/logs?taskName=analyze-and-check-degradation
- ASE: https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/ase-verify-sfmlw/logs?taskName=analyze-and-check-degradation
- A2A: https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/a2a-local-env-x8nbj/logs?taskName=analyze-and-check-degradation

**Working examples — CI pipeline:**
- Harbor (all files): https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/ci-harbor-with-instr-p92vj
- Harbor (AI generation): https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/ci-harbor-gen-instr-mn425
- ASE (all files): https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/ci-ase-with-evals-7thhz
- ASE (AI generation): https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/ci-ase-gen-evals-89p6d
- A2A: https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/ci-a2a-qqjfp

---

## Prerequisites

- `oc` CLI logged in to the cluster (`oc whoami` should return your user)
- Namespace: `ab-eval-flow`

---

# Monitoring Pipeline (`abevalflow-monitoring-pipeline`)

Runs evaluations with degradation check and Slack notifications. No security scan or quality review.

## Harbor Run (hello-world)

Uses `skill-submissions/hello-world` — a minimal Harbor task submission.

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: harbor-verify-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-monitoring-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/skill-submissions.git"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "hello-world"
    - name: eval-engine
      value: "harbor"
    - name: pipeline-repo-revision
      value: "main"
    - name: llm-model
      value: "claude-sonnet"
    - name: llm-api-base
      value: "http://litellm.ab-eval-flow.svc:4000"
    - name: llm-api-key
      value: "mock"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

**Expected result:** Treatment and control both score 1.000, recommendation `pass`.

---

## ASE Run (hello-world-full)

Uses `skill-submissions/hello-world-full` — a skill submission with `evals.json`.

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ase-verify-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-monitoring-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/skill-submissions.git"
    - name: revision
      value: "eval/hello-world-full"
    - name: submission-dir
      value: "hello-world-full"
    - name: eval-engine
      value: "ase"
    - name: pipeline-repo-revision
      value: "main"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

**Expected result:** Score 1.000, recommendation `pass`.

---

## A2A Run (lightspeed-agent)

Uses `ABEvalFlow/a2a-agent-eval` — evaluates the deployed Lightspeed A2A agent.
The agent must already be running at `http://lightspeed-agent.ab-eval-flow.svc:8000`.

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: a2a-verify-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-monitoring-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/ABEvalFlow.git"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "a2a-agent-eval"
    - name: eval-engine
      value: "a2a"
    - name: pipeline-repo-revision
      value: "main"
    - name: agent-endpoint
      value: "http://lightspeed-agent.ab-eval-flow.svc:8000"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

**Expected result:** 3/3 trials pass, mean reward 1.000.

---

## A2A Trigger Types

A2A monitoring runs can be triggered three ways (plus manual runs). Choose the right agent deployment mode for each:

| Trigger | Agent deployment | Key params |
|---------|------------------|------------|
| Quay push webhook | Ephemeral (new image) | `agent-image` + `agent-tag` |
| LiteLLM config change | Existing deployed agent | `agent-endpoint` |
| 10-day scheduled | Existing deployed agent | `agent-endpoint` |
| Manual PipelineRun | Either mode | See examples below |

**Ephemeral deploy** (`agent-image` + `agent-tag`): The pipeline deploys a temporary instance of the pushed image, evaluates it, then cleans up. Leave `agent-endpoint` empty.

**Existing agent** (`agent-endpoint`): The pipeline connects to an already-running agent (typically `http://lightspeed-agent.ab-eval-flow.svc:8000`). Do not set `agent-image`/`agent-tag` unless you want a fresh deploy.

### 1. Quay Push Webhook

Fires when a new image is pushed to `quay.io/ecosystem-appeng/google-lightspeed-agent` (excluding `sha256:` digest tags and `on-pr-*` PR tags).

- EventListener trigger: `quay-push-trigger` in `event-listener.yaml`
- Deploys an ephemeral instance of the new image for testing, then cleans up
- **Pending:** Ilona to configure the Quay repository notification pointing at the EventListener URL

### 2. LiteLLM Config Change

Fires when any file under `config/litellm/` (e.g., `config/litellm/configmap.yaml`) is pushed to the `main` branch of `RHEcosystemAppEng/ABEvalFlow`.

- EventListener trigger: `litellm-config-push-trigger`
- Uses the existing deployed agent at `http://lightspeed-agent.ab-eval-flow.svc:8000`
- **Requires:** GitHub webhook on the ABEvalFlow repo pointing to:
  `https://el-submission-listener-ab-eval-flow.apps.cn-ai-lab.2vn8.p1.openshiftapps.com`

### 3. 10-Day Scheduled

Fires via the `abevalflow-monitoring` CronJob every 10 days (`schedule: "0 6 */10 * *"`).

- Uses the existing deployed `google-lightspeed-agent` endpoint from the canary pack config
- Performs a health check on `/.well-known/agent.json` before triggering; skips eval if unhealthy

### Manual Trigger (A2A)

Example PipelineRun for an A2A monitoring run against the existing deployed agent:

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: a2a-monitoring-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-monitoring-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/ABEvalFlow.git"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "a2a-agent-eval"
    - name: eval-engine
      value: "a2a"
    - name: pipeline-repo-revision
      value: "main"
    - name: agent-endpoint
      value: "http://lightspeed-agent.ab-eval-flow.svc:8000"
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

Example for ephemeral deploy (Quay webhook equivalent — evaluates a specific image tag):

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: a2a-ephemeral-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-monitoring-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/ABEvalFlow.git"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "a2a-agent-eval"
    - name: eval-engine
      value: "a2a"
    - name: pipeline-repo-revision
      value: "main"
    - name: agent-image
      value: "quay.io/ecosystem-appeng/google-lightspeed-agent"
    - name: agent-tag
      value: "<tag-from-quay-push>"
    - name: agent-endpoint
      value: ""
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

---

# CI Pipeline (`abevalflow-pipeline`)

Full evaluation run: prepare → test → evaluate → analyze → store. No degradation check or Slack.
Security scan and quality review can be enabled/disabled per run.

## CI Harbor — all files present (no generation)

Uses `skill-submissions/hello-world` — has `instruction.md` and `tests/`. Generation is enabled
but skips because files already exist. Expects no `generated/` folder in MinIO.

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ci-harbor-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/skill-submissions.git"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "hello-world"
    - name: eval-engine
      value: "harbor"
    - name: pipeline-repo-revision
      value: "main"
    - name: enable-generation
      value: "true"
    - name: enable-security-scan
      value: "false"
    - name: enable-quality-review
      value: "false"
    - name: llm-model
      value: "claude-sonnet"
    - name: llm-api-base
      value: "http://litellm.ab-eval-flow.svc:4000"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

**Expected result:** Score 1.000, no `generated/` folder in MinIO, `debug/` folder present.

---

## CI Harbor — AI generation (no instruction.md)

Uses `skill-submissions/hello-world-no-instr` (PR #106) — only has `skills/SKILL.md` and
`metadata.yaml` with `generation_mode: ai`. Pipeline generates `instruction.md`,
`test_outputs.py`, and `llm_judge.py`.

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ci-harbor-gen-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/skill-submissions.git"
    - name: revision
      value: "eval/hello-world-no-instr"
    - name: submission-dir
      value: "hello-world-no-instr"
    - name: eval-engine
      value: "harbor"
    - name: pipeline-repo-revision
      value: "main"
    - name: enable-generation
      value: "true"
    - name: enable-security-scan
      value: "false"
    - name: enable-quality-review
      value: "true"
    - name: llm-model
      value: "claude-sonnet"
    - name: llm-api-base
      value: "http://litellm.ab-eval-flow.svc:4000"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

**Expected result:** `generated/` folder in MinIO with AI-header files, quality review score reported.

---

## CI ASE — all files present (no generation)

Uses `skill-submissions/hello-world-full` — has `evals/evals.json`. Expects no `generated/` folder.

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ci-ase-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/skill-submissions.git"
    - name: revision
      value: "eval/hello-world-full"
    - name: submission-dir
      value: "hello-world-full"
    - name: eval-engine
      value: "ase"
    - name: pipeline-repo-revision
      value: "main"
    - name: enable-generation
      value: "true"
    - name: enable-security-scan
      value: "false"
    - name: enable-quality-review
      value: "false"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

**Expected result:** Score 1.000, no `generated/` folder in MinIO.

---

## CI ASE — AI generation (no evals.json)

Uses `skill-submissions/hello-world-minimal` (PR #104) — only has `SKILL.md`. Pipeline generates
`evals/evals.json` with `_generated_by: "ai"`.

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ci-ase-gen-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/skill-submissions.git"
    - name: revision
      value: "eval/hello-world-minimal"
    - name: submission-dir
      value: "hello-world-minimal"
    - name: eval-engine
      value: "ase"
    - name: pipeline-repo-revision
      value: "main"
    - name: enable-generation
      value: "true"
    - name: enable-security-scan
      value: "false"
    - name: enable-quality-review
      value: "false"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

**Expected result:** `generated/evals.json` in MinIO with `_generated_by: "ai"`.

---

## CI MCPChecker — ExploitIQ

Uses `skill-submissions/exploitiq-mcp-eval` (branch `test/mcpchecker-exploitiq`).
Requires `exploitiq-mcp-credentials` secret with a fresh `oc whoami -t` token from `ai-dev03`.

**Refresh token before running:**
```bash
oc login --token=<your-token> --server=https://api.ai-dev03.kni.syseng.devcluster.openshift.com:6443
NEW_TOKEN=$(oc whoami -t)
oc config use-context "ab-eval-flow/api-cn-ai-lab-2vn8-p1-openshiftapps-com:6443/gziv@redhat.com"
oc create secret generic exploitiq-mcp-credentials \
  --from-literal=MCP_URL="https://exploitiq-mcp-server-exploit-iq-testings.apps.ai-dev03.kni.syseng.devcluster.openshift.com/mcp" \
  --from-literal=MCP_BEARER_TOKEN="$NEW_TOKEN" \
  -n ab-eval-flow --dry-run=client -o yaml | oc apply -f -
```

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ci-mcp-exploitiq-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/skill-submissions.git"
    - name: revision
      value: "test/mcpchecker-exploitiq"
    - name: submission-dir
      value: "exploitiq-mcp-eval"
    - name: eval-engine
      value: "mcpchecker"
    - name: pipeline-repo-revision
      value: "main"
    - name: enable-generation
      value: "false"
    - name: enable-security-scan
      value: "false"
    - name: enable-quality-review
      value: "false"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

**Expected result:** All tasks pass (health_check, list_reports, analyze_cve).

---

## CI A2A

Uses `ABEvalFlow/a2a-agent-eval`. Security scan and quality review disabled (no test files).

```bash
oc create -f - <<'YAML'
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ci-a2a-
  namespace: ab-eval-flow
spec:
  pipelineRef:
    name: abevalflow-pipeline
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/ABEvalFlow.git"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "a2a-agent-eval"
    - name: eval-engine
      value: "a2a"
    - name: pipeline-repo-revision
      value: "main"
    - name: agent-endpoint
      value: "http://lightspeed-agent.ab-eval-flow.svc:8000"
    - name: security-scan
      value: "disabled"
    - name: enable-test-quality-review
      value: "false"
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
YAML
```

**Expected result:** Mean reward 1.000.

---

## Monitoring & Cleanup

```bash
# Watch all runs
oc get pipelinerun -n ab-eval-flow --watch

# Tail logs for a specific run/task
oc logs -n ab-eval-flow <run-name>-evaluate-pod -c step-harbor-eval -f
oc logs -n ab-eval-flow <run-name>-evaluate-pod -c step-ase-eval -f
oc logs -n ab-eval-flow <run-name>-evaluate-pod -c step-a2a-eval -f
oc logs -n ab-eval-flow <run-name>-analyze-and-check-degradation-pod -c step-check-degradation -f

# Delete all failed runs
oc get pipelinerun -n ab-eval-flow --no-headers | grep "False" \
  | awk '{print $1}' > /tmp/failed.txt && while read r; do oc delete pipelinerun -n ab-eval-flow "$r"; done < /tmp/failed.txt
```

---

## Slack Notifications

Every completed monitoring run sends a Slack message to the team channel:
- ✅ `[ENGINE] Monitoring Pass` — score + baseline + ratio
- 🚨 `[ENGINE] Performance Degradation Detected` — score dropped below threshold

The Run ID in the message is a clickable link to the OpenShift console.

---

## Active Image

All eval steps (`harbor-eval`, `a2a-eval`) use `eval-base:local-env` — built from
Harbor `feature/local-environment` branch with `claude-code` pre-installed.
This is the canonical image; do not revert to `:latest`.

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `monitor.py: No such file` | Old `pipeline-repo-revision` pointing to deleted branch | Ensure `pipeline-repo-revision: main` |
| `NonZeroAgentExitCodeError` in Harbor eval | Local registry `eval-base` out of sync with main registry | Re-run the skopeo sync (see `infrastructure_ops.md`) |
| `Generated files: []` despite generation running | LLM call failed silently | Check stderr output in logs (now shown on failure) |
| `generated/` folder missing despite AI generation | Files already existed (no AI header) or generation failed | Check `STEP-GENERATE-TESTS` logs for WARNING output |
| `debug/` folder missing in MinIO for Harbor runs | `results-dir` not passed to store task | Fixed in PR #32 — ensure `pipeline-repo-revision: main` |
| Degradation shows `0.00% → 0.00%` | `store` runs after `check-degradation`; monitor reads old DB runs | Fixed: current score passed from `report.json` via `--current-score` |
| No Slack alert despite degradation | `\|\|` block caught exit code 1 from `monitor.py` | Fixed: only exit code 2 (error) is non-blocking now |
| Slack Run ID shows `None` | `--run-id` not passed to `monitor.py` | Fixed: `--run-id "$(params.pipeline-run-id)"` now wired in task |
