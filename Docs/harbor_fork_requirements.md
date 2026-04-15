# Harbor Fork — Required Changes for ABEvalFlow Integration

> **Target repo:** [RHEcosystemAppEng/skills_eval_corrections](https://github.com/RHEcosystemAppEng/skills_eval_corrections)
> **Open PR:** [#1 — feat: add OpenShift environment backend](https://github.com/RHEcosystemAppEng/skills_eval_corrections/pull/1)
> **ABEvalFlow branch:** `APPENG-4906/harbor-eval-task`

---

## 1. Per-Task `environment_kwargs` Support

### Problem

ABEvalFlow evaluates two variants (treatment and control) with different pre-built
images in a single Harbor job. The current `--ek image_ref=<ref>` mechanism sets the
image ref globally on `EnvironmentConfig.kwargs`, so all tasks in a job share the
same image. This prevents using a single sweep config for both variants.

### Proposed Change

Add an optional `environment_kwargs` field to `TaskConfig` in
`src/harbor/models/trial/config.py`:

```python
class TaskConfig(BaseModel):
    path: Path
    git_url: str | None = None
    git_commit_id: str | None = None
    overwrite: bool = False
    download_dir: Path | None = None
    source: str | None = None
    environment_kwargs: dict[str, Any] = Field(default_factory=dict)  # <-- new
```

### Merge Logic

When the `Job` creates `TrialConfig` instances from `JobConfig`, it should merge
per-task kwargs into the trial's environment config:

```python
# In the trial creation loop (conceptual):
trial_env_config = job_config.environment.model_copy(deep=True)
trial_env_config.kwargs = {**trial_env_config.kwargs, **task.environment_kwargs}
```

Task-level kwargs override global kwargs for the same key (task wins).

### Result

A single sweep config can specify different `image_ref` values per variant:

```yaml
job_name: my-submission-eval
jobs_dir: eval-results
n_attempts: 20
environment:
  type: openshift
  delete: true
agents:
  - name: claude-code
    model_name: claude-sonnet-4-5
tasks:
  - path: /workspace/tasks-treatment/my-submission
    environment_kwargs:
      image_ref: "registry/ns/my-submission@sha256:abc..."
  - path: /workspace/tasks-control/my-submission
    environment_kwargs:
      image_ref: "registry/ns/my-submission@sha256:def..."
```

### Testing

- Unit test: verify that `TrialConfig` created from a `TaskConfig` with
  `environment_kwargs` has them merged into `environment.kwargs`.
- Unit test: verify task-level kwargs override global kwargs.
- Unit test: verify `TaskConfig` without `environment_kwargs` behaves identically
  to current behavior (backwards compatible).

---

## 2. Handoff Doc Alignment (WS3A)

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
