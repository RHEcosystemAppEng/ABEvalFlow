# Manual Trigger Guide — Harbor, ASE & A2A

Quick reference for manually triggering evaluations against the monitoring pipeline.

**Working examples:**
- Harbor: https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/harbor-verify-t6spp/logs?taskName=analyze-and-check-degradation
- ASE: https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/ase-verify-sfmlw/logs?taskName=analyze-and-check-degradation
- A2A: https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/a2a-local-env-x8nbj/logs?taskName=analyze-and-check-degradation

---

## Prerequisites

- `oc` CLI logged in to the cluster (`oc whoami` should return your user)
- Namespace: `ab-eval-flow`
- Pipeline: `abevalflow-monitoring-pipeline`

---

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
      value: "APPENG-4911/monitoring"
    - name: harbor-fork-revision
      value: "feature/local-environment"
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
      value: "APPENG-4911/monitoring"
    - name: harbor-fork-revision
      value: "feature/local-environment"
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
      value: "APPENG-4911/monitoring"
    - name: submission-dir
      value: "a2a-agent-eval"
    - name: eval-engine
      value: "a2a"
    - name: pipeline-repo-revision
      value: "APPENG-4911/monitoring"
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
oc get pipelinerun -n ab-eval-flow --no-headers | grep -E "False|Failed" \
  | awk '{print $1}' | xargs -r oc delete pipelinerun -n ab-eval-flow
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
| `monitor.py: No such file` | `pipeline-repo-revision` defaulted to `main` | Always pass `pipeline-repo-revision: APPENG-4911/monitoring` |
| `NonZeroAgentExitCodeError` in Harbor eval | Local registry `eval-base` out of sync with main registry | Re-run the skopeo sync (see `infrastructure_ops.md`) |
| `monitor.py: unrecognized arguments` | Cluster cloned old `monitor.py` before push | Ensure latest commit is on `APPENG-4911/monitoring` and retrigger |
| Degradation shows `0.00% → 0.00%` | `store` runs after `check-degradation`; monitor reads old DB runs | Fixed: current score passed from `report.json` via `--current-score` |
| No Slack alert despite degradation | `\|\|` block caught exit code 1 from `monitor.py` | Fixed: only exit code 2 (error) is non-blocking now |
| Slack Run ID shows `None` | `--run-id` not passed to `monitor.py` | Fixed: `--run-id "$(params.pipeline-run-id)"` now wired in task |
