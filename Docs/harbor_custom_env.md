# Harbor Custom Environment — Integration Contract for ABEvalFlow

> **Status:** Current (replaces the skills_eval_corrections fork runtime dependency)
> **Related:** [harbor_openshift_backend.md](./harbor_openshift_backend.md)

---

## Runtime contract

Classic Harbor A/B (`eval_engine: harbor`) and AEH Harbor runs share:

| Dependency | Pin / source |
|---|---|
| Harbor | `harbor==0.20.0` (PyPI; stock upstream) |
| Kubernetes client | `kubernetes>=32.0.0` |
| AEH K8s env | `agent_eval.harbor.kubernetes.KubernetesEnvironment` on `PYTHONPATH` (pinned AEH SHA in eval-base) |
| ABEvalFlow plugin | `abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment` on `PYTHONPATH` |

**Custom env selection** (prebuilt / OpenShift CI only):

JobConfig `environment.import_path`:

```yaml
environment:
  import_path: abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment
```

Classic Harbor CI uses that YAML field only (`harbor run -c <config.yaml> -y`).
AEH still passes Harbor’s deprecated `--environment-import-path` when
`agent_eval.harbor.run` would otherwise override the config.

**Do not** use stock Harbor `environment.type: openshift` (oc CLI / custom SCC model) for shared OpenShift CI.

---

## How ABEvalFlow invokes Harbor

The composite evaluate task (`pipeline/tasks/phases/evaluate.yaml`) Harbor path:

1. Uses `eval-base:local-env` (stock Harbor + AEH + baked `abevalflow`)
2. Generates **two** Harbor job configs via `scripts/generate_eval_config.py`
3. Writes treatment/control digests into each task’s `task.toml` as `docker_image`
4. Runs `harbor run -c treatment-config.yaml` then `harbor run -c control-config.yaml` (env from YAML `import_path`)

### Per-variant config (prebuilt)

```yaml
job_name: my-submission-treatment
jobs_dir: /workspace/eval-results/my-submission/treatment
n_attempts: 20
environment:
  import_path: abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment
  delete: true
  override_memory_mb: 2048
  override_storage_mb: 10240
  kwargs:
    cpu_request: "100m"
    memory_limit_multiplier: 1.5
agents:
  - name: claude-code
    model_name: claude-sonnet
tasks:
  - path: /workspace/tasks-treatment/my-submission
```

Corresponding `task.toml` fragment:

```toml
[environment]
docker_image = "registry/ns/my-submission@sha256:abc..."
cpus = 1
memory_mb = 2048
```

### Local / dev (`eval_mode=local-build`)

```yaml
environment:
  type: docker
  force_build: true
  delete: true
```

No OpenShift import path; Harbor builds from each task’s Dockerfile.
**Local / privileged Docker only** — not supported on shared OpenShift CI
(no docker-in-docker). Cluster pipelines should use `eval_mode=prebuilt`.

### Result directory layout

```
eval-results/<submission-name>/
    treatment/
        <job-name>/
            <task-name>__<uuid>/result.json
            ...
    control/
        <job-name>/
            ...
```

Analyze/publish remain compatible as long as this `jobs/` shape holds.

---

## Cluster RBAC (unchanged)

Pipeline ServiceAccount needs: create/get/delete pods, exec, pull secrets.
**No** `harbor-task-scc` / root-capable custom SCC.

Trial pods target OpenShift `restricted-v2` SCC. The custom env adds emptyDir mounts for `/workspace` and `/tmp` and ensures Harbor EnvironmentPaths exist before verifier redirect.

---

## Eval-base image

Built from [`templates/Dockerfile.base`](../templates/Dockerfile.base):

```bash
./scripts/build_base_image.sh --tag latest
./scripts/build_base_image.sh --tag local-env
```

Pins: `HARBOR_VERSION=0.20.0`, `AEH_SHA` (default AEH v1.20.0).

---

## Historical note

The `skills_eval_corrections` Harbor fork (`environment.type: openshift` + `kwargs.image_ref`) is **no longer a required runtime dependency**. See [harbor_openshift_backend.md](./harbor_openshift_backend.md) for the historical fork handoff.

---

## Restricted-v2 smoke checklist

After rebuilding `eval-base:latest` and `eval-base:local-env`:

1. Trigger one Harbor submission (treatment+control), small `n_trials`, restricted-v2 namespace.
2. Confirm evaluate logs: no `git+…/skills_eval_corrections` install; `harbor==0.20.0` / import path present.
3. Confirm trial pods start and finish; `result.json` under `eval-results/.../treatment|control`.
4. Confirm analyze/publish still consume the result layout.
5. Optional: compare mean-reward shape vs a prior fork-path baseline run.
