# Manual Trigger Guide — Harbor & ASE (Hello-World)

Quick reference for manually triggering evaluations against the monitoring pipeline.

**Working examples:**
- Harbor: https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/harbor-verify-t6spp/logs?taskName=analyze-and-check-degradation
- ASE: https://console-openshift-console.apps.cn-ai-lab.2vn8.p1.openshiftapps.com/k8s/ns/ab-eval-flow/tekton.dev~v1~PipelineRun/ase-verify-sfmlw/logs?taskName=analyze-and-check-degradation

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
  generateName: harbor-test-
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
  generateName: ase-test-
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

---

## Monitoring & Cleanup

```bash
# Watch all runs
oc get pipelinerun -n ab-eval-flow --watch

# Tail logs for a specific run/task
oc logs -n ab-eval-flow <run-name>-evaluate-pod -c step-harbor-eval -f
oc logs -n ab-eval-flow <run-name>-evaluate-pod -c step-ase-eval -f

# Delete all failed runs
oc get pipelinerun -n ab-eval-flow --no-headers | grep -E "False|Failed" \
  | awk '{print $1}' | xargs -r oc delete pipelinerun -n ab-eval-flow
```

---

## Active Image

All eval steps (harbor-eval, a2a-eval) use `eval-base:local-env` — built from Harbor `feature/local-environment` branch with `claude-code` pre-installed. This is the canonical image; do not revert to `:latest`.

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `monitor.py: No such file` | `pipeline-repo-revision` defaulted to `main` | Always pass `pipeline-repo-revision: APPENG-4911/monitoring` |
| `NonZeroAgentExitCodeError` in Harbor eval | Local registry `eval-base` out of sync with main registry | Re-run the skopeo sync (see `infrastructure_ops.md`) |
| Degradation shows `0.00% → 0.00%` | `store` runs after `check-degradation`; monitor reads old DB runs | Fixed: current score now passed directly from `report.json` via `--current-score` |
| No Slack alert despite degradation | `||` block caught exit code 1 from `monitor.py` | Fixed: only exit code 2 (error) is non-blocking now |
