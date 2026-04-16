# Harbor Fork — Integration Requirements for ABEvalFlow

> **Target repo:** [RHEcosystemAppEng/skills_eval_corrections](https://github.com/RHEcosystemAppEng/skills_eval_corrections)
> **Open PR:** [#1 — feat: add OpenShift environment backend](https://github.com/RHEcosystemAppEng/skills_eval_corrections/pull/1)
> **ABEvalFlow branch:** `APPENG-4906/harbor-eval-task`

---

## ABEvalFlow-Side Implementation (current state)

### How ABEvalFlow invokes Harbor

The `harbor-eval` Tekton task (`pipeline/tasks/harbor-eval.yaml`) runs a single
step that:

1. Installs Harbor from the fork via `pip install git+<fork-url>@<revision>`
2. Generates **two separate Harbor job configs** (one per variant) using
   `scripts/generate_eval_config.py`
3. Runs `harbor run -c treatment-config.yaml` followed by
   `harbor run -c control-config.yaml`
4. Parses `result.json` files from each variant's results directory to compute
   pass rates

### Per-variant config structure

Each config is a standard Harbor `JobConfig` YAML with a **single task** and
the image ref set via global `environment.kwargs.image_ref`:

```yaml
# treatment-config.yaml
job_name: my-submission-treatment
jobs_dir: /workspace/eval-results/my-submission/treatment
n_attempts: 20
environment:
  type: openshift
  delete: true
  kwargs:
    image_ref: "registry/ns/my-submission@sha256:abc..."
  override_cpus: 1
  override_memory_mb: 2048
  override_storage_mb: 10240
agents:
  - {}
tasks:
  - path: /workspace/tasks-treatment/my-submission
```

Control config is identical but with the control image ref and task path.

### Result directory layout

```
eval-results/<submission-name>/
    treatment/
        <job-name>/
            <task-name>__<uuid>/result.json
            <task-name>__<uuid>/result.json
            ...  (N trials)
    control/
        <job-name>/
            <task-name>__<uuid>/result.json
            ...  (N trials)
```

### What ABEvalFlow reads from metadata.yaml

The config generator extracts these fields from `SubmissionMetadata`:

| Field | Maps to | Default |
|-------|---------|---------|
| `experiment.n_trials` | `n_attempts` | 20 |
| `agent_timeout_sec` | `agent_timeout_multiplier` (ratio vs 600s) | 1.0x |
| `verifier_timeout_sec` | `verifier_timeout_multiplier` (ratio vs 120s) | 1.0x |
| `agent_setup_timeout_sec` | `agent_setup_timeout_multiplier` (ratio vs 600s) | 1.0x |
| `build_timeout_sec` | `environment_build_timeout_multiplier` (ratio vs 600s) | 1.0x |
| `cpus` | `environment.override_cpus` | 1 |
| `memory_mb` | `environment.override_memory_mb` | 2048 |
| `storage_mb` | `environment.override_storage_mb` | 10240 |

### Eval modes

| Mode | `--ek image_ref` | Image source | When to use |
|------|------------------|--------------|-------------|
| `prebuilt` | Set to digest ref from build-push task | Tekton builds with Buildah, pushes to internal registry | Default pipeline flow |
| `local-build` | Not set; `force_build: true` | Harbor builds from `environment/Dockerfile` in each task dir | Local dev, or when skipping the build-push step |

### Tekton results emitted

| Result | Description |
|--------|-------------|
| `treatment-pass-rate` | Decimal string (e.g. `"0.8500"`) |
| `control-pass-rate` | Decimal string (e.g. `"0.6000"`) |
| `results-dir` | Absolute path to the results base directory |

---

## What the Harbor fork must support

### Required (blocking)

**1. `harbor run -c <config.yaml>` with OpenShift environment**

The fork's `OpenShiftEnvironment` must handle the full trial lifecycle when
invoked with `--env openshift` (or `environment.type: openshift` in config):

- Accept `image_ref` via `environment.kwargs` — verify the image is pullable,
  skip building
- Create trial Pods with the pre-built image
- Execute agent + verifier inside the Pod via `exec`
- Upload/download files via tar-over-exec
- Write `result.json` with `verifier_result.reward` (float: 1.0 = pass, 0.0 = fail)
- Clean up Pods after each trial (`delete: true`)

**Status:** Implemented in PR #1. Needs merge.

**2. `environment.kwargs` passthrough in config-based invocation**

When `harbor run -c config.yaml` is used, the `environment.kwargs` dict from
the config must be passed to the environment's `__init__`. This is how
`image_ref` reaches `OpenShiftEnvironment`.

**Status:** Verify this works in `harbor jobs start` with YAML config
(vs CLI `--ek` flags). The CLI path (`--ek image_ref=...`) is tested;
the config path should behave identically but needs confirmation.

**3. `result.json` output format**

ABEvalFlow's pass-rate parser expects:

```json
{
  "verifier_result": {
    "reward": 1.0
  }
}
```

Where `reward > 0.0` means pass. This is Harbor's standard format — no change
needed, but any deviation would break the parser.

### Nice-to-have (not blocking)

**4. Per-task `environment_kwargs` support**

ABEvalFlow currently runs each variant as a separate Harbor job to work around
the global `environment.kwargs`. If we later want to run both variants in a
single job (e.g., for sweep-based workflows), per-task `environment_kwargs`
would be needed.

Proposed change: add `environment_kwargs: dict[str, Any]` to `TaskConfig` in
`src/harbor/models/trial/config.py`. When `Job` creates `TrialConfig` instances,
merge per-task kwargs into the trial's `EnvironmentConfig.kwargs` (task-level
overrides global).

```yaml
# Single-job example (requires this fork change):
tasks:
  - path: /workspace/tasks-treatment/my-submission
    environment_kwargs:
      image_ref: "registry/ns/my-submission@sha256:abc..."
  - path: /workspace/tasks-control/my-submission
    environment_kwargs:
      image_ref: "registry/ns/my-submission@sha256:def..."
```

---

## Handoff Doc Alignment (WS3A)

The existing handoff doc (`Docs/harbor_openshift_backend.md`) has diverged from
the actual implementation in PR #1. These items should be updated:

| Section | Current (outdated) | Correct |
|---------|-------------------|---------|
| File path | `openshift_environment.py` | `openshift.py` |
| Build modes | `_build_and_push_image` is no-op only | Supports pre-built (`--ek image_ref=`) AND local podman build |
| Pod security | `readOnlyRootFilesystem: true` | Intentionally unset — many agent workloads need writes; `HOME=/tmp` is injected instead |
| RBAC table | ConfigMaps, Secrets, PVCs, ImageStreams | Only Pods + exec + Secrets used in practice |
| Naming | "skilled / unskilled" | "treatment / control" |
| Trial count | "20 skilled + 20 unskilled" | "20 treatment + 20 control" |
| Tekton params | `skilled-image-ref` / `unskilled-image-ref` | `treatment-image-ref` / `control-image-ref` |
| Definition of Done | "20 skilled + 20 unskilled" | "20 treatment + 20 control" |

### Additional Notes

- The OpenShift backend supports a `cpu_request` kwarg (`--ek cpu_request=<val>`)
  for clusters with tight resource constraints — not documented in the handoff doc.
- The `--ek registry=<url>` kwarg enables local podman build+push to a specified
  registry — also undocumented.
