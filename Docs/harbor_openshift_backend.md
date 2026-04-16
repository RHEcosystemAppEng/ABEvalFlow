# Harbor OpenShift Backend — Handoff Document

> **Jira:** APPENG-4906 (Phase 4 — Harbor OpenShift Backend)
> **Target repo:** [RHEcosystemAppEng/skills_eval_corrections](https://github.com/RHEcosystemAppEng/skills_eval_corrections) (Harbor fork)
> **Full spec:** See [implementation_plan.md](./implementation_plan.md), Phase 4 (lines 270–343)

---

## What to Build

A new `OpenShiftEnvironment` class in the Harbor fork that enables `harbor run --env openshift`. This backend manages trial Pod lifecycle on OpenShift using pre-built container images (built by ABEvalFlow's Tekton pipeline).

## Where in the Harbor Fork

| File | Action |
|---|---|
| `src/harbor/environments/openshift_environment.py` | Create — new backend |
| `src/harbor/models/environment_type.py` | Edit — add `OPENSHIFT = "openshift"` to enum |
| Backend registration (entry point or factory) | Edit — register so `--env openshift` works |
| `tests/` | Create — unit tests with mocked K8s API |

## Reference Implementation

Use the GKE backend as your template: `src/harbor/environments/gke.py` (~1044 lines). It implements the full `BaseEnvironment` interface from `src/harbor/environments/base.py`.

## Key Differences from GKE

### `_init_client`
- GKE: `gcloud container clusters get-credentials` + `load_kube_config()`
- **OpenShift:** `load_incluster_config()` when running inside the cluster (Tekton), or `load_kube_config()` for local dev

### `_build_and_push_image`
- GKE: `gcloud builds submit` (Cloud Build)
- **OpenShift: No-op.** Image build/push is handled by Tekton (Phase 3). The backend receives a digest-based image reference and only verifies the image exists and is pullable.

**Contract:**
- **Input:** Immutable image ref (e.g., `image-registry.openshift-image-registry.svc:5000/ab-eval-flow/my-skill@sha256:abc...`)
- **Behavior:** Verify image exists using the trial ServiceAccount
- **Output:** Return the image ref unchanged
- **Failure:** Raise `ImageNotFoundError` / `ImageNotPullableError` with the image ref and SA identity

### `_image_exists`
- GKE: `gcloud artifacts docker images describe`
- **OpenShift:** Query internal registry API or use `skopeo inspect`

### `start`, `stop`, `exec`, `upload_file/dir`, `download_file/dir`
- Same K8s API patterns as GKE — the `kubernetes` Python client is identical

## Pod Security Requirements

Trial Pods must run with hardened security context:

```yaml
securityContext:
  runAsNonRoot: true
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
  seccompProfile:
    type: RuntimeDefault
```

Since `readOnlyRootFilesystem: true` prevents writes, mount `emptyDir` volumes for:
- `/tmp`
- Agent cache directories (varies by agent)

Verify the target cluster uses `restricted-v2` SCC (OpenShift 4.11+).

## Trial Execution

- **N = 20** attempts per variant (skilled + unskilled = 40 total sessions)
- Image refs come as params from the build-push Tekton task (digest-based)
- LLM endpoint via environment variable — backend is agnostic to LLM access mode
- Configurable per-trial timeout and global evaluation timeout
- Resource requests/limits per trial Pod

## RBAC Requirements

The pipeline ServiceAccount in `ab-eval-flow` namespace needs:

| Resource | Verbs | Purpose |
|---|---|---|
| Pods | create, get, list, watch, delete | Trial Pod lifecycle |
| ConfigMaps | get, list | Trial configuration |
| Secrets | get | LLM credentials injection |
| Events | get, list | Diagnosing hung/failed Pods |
| PVCs | get, list, create | Pipeline workspaces |
| ImageStreams | get, list | Registry access |

## Testing Strategy

- **Unit tests:** Mock K8s API with `pytest` + `unittest.mock`. No live cluster needed.
- **Integration tests:** Test against OpenShift developer sandbox (ROSA/OSD). Do **not** use Kind/Minikube — they won't catch SCC/Routes differences.

## Tekton Task (in ABEvalFlow repo, not Harbor fork)

A `pipeline/tasks/harbor-eval.yaml` Tekton Task will also be needed in ABEvalFlow to invoke Harbor. This task:
- Accepts `skilled-image-ref` and `unskilled-image-ref` as params
- Runs `harbor run --env openshift` with the image refs
- Collects results to workspace/PVC

This can be built after the backend is functional.

## Definition of Done

- [ ] 40 trial Pods complete (20 skilled + 20 unskilled)
- [ ] Cleanup verified — no stale Pods after evaluation
- [ ] Retry behavior validated for transient failures
- [ ] Unit tests pass with mocked K8s API
- [ ] Integration test passes on OpenShift sandbox

## LLM Access Modes (for reference)

The backend doesn't need to know which mode is used — it just passes env vars to trial Pods:

| Mode | Env Var | Infrastructure |
|---|---|---|
| Direct API key | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | None |
| opencode + self-hosted | Model endpoint URL | Self-hosted model |
| Vertex AI + LiteLLM | `LITELLM_BASE_URL` | LiteLLM Deployment |
