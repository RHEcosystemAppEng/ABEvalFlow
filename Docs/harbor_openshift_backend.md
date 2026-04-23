# Harbor OpenShift Backend — Handoff Document

> **Jira:** APPENG-4906 (Phase 4 — Harbor OpenShift Backend)
> **Target repo:** [RHEcosystemAppEng/skills_eval_corrections](https://github.com/RHEcosystemAppEng/skills_eval_corrections) (Harbor fork)
> **Fork PRs:** [#1 — OpenShift environment backend](https://github.com/RHEcosystemAppEng/skills_eval_corrections/pull/1), [#2 — per-task environment_kwargs](https://github.com/RHEcosystemAppEng/skills_eval_corrections/pull/2)
> **Full spec:** See [implementation_plan.md](./implementation_plan.md), Phase 4

---

## What Was Built

An `OpenShiftEnvironment` class in the Harbor fork that enables
`harbor run --env openshift`. This backend manages trial Pod lifecycle
on OpenShift and supports two modes: pre-built container images (from
the Tekton pipeline) and local build via podman.

## Where in the Harbor Fork

| File | Action |
|---|---|
| `src/harbor/environments/openshift.py` | New backend |
| `src/harbor/environments/k8s_client_manager.py` | Shared K8s client management (independent instances per caller) |
| `src/harbor/models/environment_type.py` | `OPENSHIFT = "openshift"` added to enum |
| `src/harbor/models/trial/config.py` | `environment_kwargs` field added to `TaskConfig` (PR #2) |
| `src/harbor/job.py` | `_env_config_for_task` merges per-task kwargs into env config (PR #2) |
| `tests/unit/environments/test_openshift.py` | Unit tests with mocked K8s API |
| `tests/unit/models/test_trial_task_config.py` | Tests for per-task kwargs merge behavior |
| `examples/configs/openshift-*.yaml` | Config examples for both modes + per-task kwargs |

## Reference Implementation

The GKE backend (`src/harbor/environments/gke.py`) implements the full
`BaseEnvironment` interface from `src/harbor/environments/base.py`.

## Key Differences from GKE

### `_init_client`
- GKE: `gcloud container clusters get-credentials` + `load_kube_config()`
- **OpenShift:** `load_incluster_config()` when running inside the cluster
  (Tekton), falls back to `load_kube_config()` for local dev

### `_build_and_push_image`
- GKE: `gcloud builds submit` (Cloud Build)
- **OpenShift — two modes:**

| Mode | Behavior | When to use |
|------|----------|-------------|
| **Prebuilt** (`image_ref` kwarg set) | Verify image is pullable, skip building | Default pipeline flow — Tekton builds with Buildah |
| **Local build** (`force_build: true`) | Build with podman, push to specified registry | Local dev, skipping the build-push Tekton step |

**Prebuilt contract:**
- **Input:** Immutable image ref (e.g., `image-registry.openshift-image-registry.svc:5000/ab-eval-flow/my-skill@sha256:abc...`)
- **Behavior:** Verify image exists using the trial ServiceAccount
- **Output:** Return the image ref unchanged
- **Failure:** Raise `ImageNotFoundError` / `ImageNotPullableError`

**Local build contract:**
- **Input:** `environment/Dockerfile` in the task directory, `registry` kwarg for push target
- **Behavior:** `podman build` + `podman push` to the specified registry
- **Kwargs:** `registry` (push target URL), `tls_verify` (default `"true"`)

### `start`, `stop`, `exec`, `upload_file/dir`, `download_file/dir`
- Same K8s API patterns as GKE — the `kubernetes` Python client is identical

## Environment kwargs

Kwargs are passed via `environment.kwargs` in the job config YAML or
via `--ek key=value` on the CLI. Per-task kwargs (PR #2) override
global kwargs when set.

| Kwarg | Description | Default |
|-------|-------------|---------|
| `namespace` | OpenShift namespace for trial Pods | Required |
| `image_ref` | Digest-based image ref (prebuilt mode) | — |
| `registry` | Push target URL (local-build mode) | — |
| `cpu_request` | CPU request for trial Pods (e.g. `"250m"`) | — |
| `tls_verify` | TLS verification for registry (e.g. `"false"`) | `"true"` |

### Per-task environment_kwargs (PR #2)

`TaskConfig` now supports an `environment_kwargs` dict that merges into
the global `environment.kwargs` (task-level overrides global). This
enables treatment and control variants with different image refs in a
single Harbor job:

```yaml
tasks:
  - path: /workspace/tasks-treatment/my-submission
    environment_kwargs:
      image_ref: "registry/ns/my-submission@sha256:abc..."
  - path: /workspace/tasks-control/my-submission
    environment_kwargs:
      image_ref: "registry/ns/my-submission@sha256:def..."
```

ABEvalFlow currently runs each variant as a separate Harbor job (two
`harbor run` invocations). Per-task kwargs enables a single-job
alternative if needed in the future.

## Pod Security Requirements

Trial Pods run with OpenShift's default security constraints. The
`readOnlyRootFilesystem` is intentionally **not** set — many agent
workloads need filesystem writes. Instead, `HOME=/tmp` is injected to
direct writes to a writable location.

Verify the target cluster uses `restricted-v2` SCC (OpenShift 4.11+).

## Trial Execution

- **N = 20** attempts per variant (treatment + control = 40 total sessions)
- Image refs come as params from the build-push Tekton task (digest-based)
- LLM endpoint via environment variable — backend is agnostic to LLM access mode
- Configurable per-trial timeout and global evaluation timeout
- Resource requests/limits per trial Pod

## RBAC Requirements

The pipeline ServiceAccount in `ab-eval-flow` namespace needs:

| Resource | Verbs | Purpose |
|---|---|---|
| Pods | create, get, list, watch, delete | Trial Pod lifecycle |
| Pods/exec | create | Agent/verifier execution inside trial Pods |
| Pods/log | get | Retrieving trial output |
| Secrets | get | LLM credentials injection |
| Events | get, list | Diagnosing hung/failed Pods |

## K8s Client Management

The `BaseK8sClientManager` (shared by GKE and OpenShift backends) was
refactored in PR #2 to return **independent `CoreV1Api` instances** per
caller, each backed by its own `ApiClient`. This prevents concurrent
`kubernetes.stream.stream()` calls (which monkey-patch `ApiClient.call_api`)
from interfering with regular REST calls in other coroutines.

## Testing Strategy

- **Unit tests:** Mock K8s API with `pytest` + `unittest.mock`. No live cluster needed.
- **Integration tests:** Test against OpenShift developer sandbox (ROSA/OSD).
  Do **not** use Kind/Minikube — they won't catch SCC/Routes differences.

## Tekton Task (in ABEvalFlow repo)

The `harbor-eval` Tekton task (`pipeline/tasks/harbor-eval.yaml`) invokes
Harbor from the ABEvalFlow pipeline:

- Installs Harbor from the fork via `pip install git+<fork-url>@<revision>`
- Generates two per-variant job configs using `scripts/generate_eval_config.py`
- Runs `harbor run -c treatment-config.yaml` then `harbor run -c control-config.yaml`
- Parses `result.json` files to compute pass rates as Tekton results
- Supports `prebuilt` and `local-build` eval modes via a pipeline param

## Definition of Done

- [x] `OpenShiftEnvironment` backend implemented (PR #1)
- [x] Per-task `environment_kwargs` support (PR #2)
- [x] Independent K8s client instances per caller (PR #2)
- [x] Config examples for prebuilt, local-build, and per-task modes (PR #2)
- [x] Unit tests with mocked K8s API
- [ ] 40 trial Pods complete (20 treatment + 20 control) on live cluster
- [ ] Cleanup verified — no stale Pods after evaluation
- [ ] Retry behavior validated for transient failures
- [ ] Integration test passes on OpenShift sandbox

## LLM Access Modes (for reference)

The backend doesn't need to know which mode is used — it just passes
env vars to trial Pods:

| Mode | Env Var | Infrastructure |
|---|---|---|
| Direct API key | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | None |
| opencode + self-hosted | Model endpoint URL | Self-hosted model |
| Vertex AI + LiteLLM | `LITELLM_BASE_URL` | LiteLLM Deployment |
