# Harbor OpenShift Backend -- Historical Fork Handoff (archived)

> **Status:** Historical -- classic Harbor A/B no longer depends on this fork at runtime.
> **Current contract:** [harbor_custom_env.md](./harbor_custom_env.md)
> **Former Jira:** APPENG-4906 (Phase 4 -- Harbor OpenShift Backend)
> **Former target repo:** [RHEcosystemAppEng/skills_eval_corrections](https://github.com/RHEcosystemAppEng/skills_eval_corrections)

---

## Current model (summary)

Agentic Eval Flow classic Harbor CI now uses:

- Stock / pinned upstream Harbor (`harbor==0.20.0`)
- Custom environment via Harbor `environment.import_path` /
  `--environment-import-path` pointing at
  `abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment`
- Prebuilt trial images via task.toml `docker_image` (pipeline scaffold + crane)
- `restricted-v2` friendly pods (emptyDir `/workspace` + `/tmp`; no oc Binary Builds; no harbor-task-scc)

**Do not** adopt upstream Harbor's built-in `environment.type: openshift` (oc CLI,
custom SCC, often root-capable pods) as the default for shared OpenShift CI.

---

## What the fork built (archive)

An `OpenShiftEnvironment` class in the Harbor fork enabled
`harbor run --env openshift` with:

| File (fork) | Role |
|---|---|
| `src/harbor/environments/openshift.py` | OpenShift backend |
| `src/harbor/environments/k8s_client_manager.py` | Shared K8s client |
| `src/harbor/models/environment_type.py` | `OPENSHIFT = "openshift"` enum |
| Per-task `environment_kwargs` | Merge task-level kwargs into env config |

Prebuilt mode used `environment.kwargs.image_ref`. Local build used
`force_build: true` + podman.

That contract is **superseded** by the custom-env + `docker_image` model above.
The fork may remain for history but is not required to run Agentic Eval Flow Harbor A/B.

---

## Why not stock Harbor `type: openshift`?

Upstream Harbor's native OpenShift backend typically needs `oc`, Binary Builds,
and a custom SCC / root-capable pods. That conflicts with Tekton prebuilt image
flow and shared cluster restricted-v2 policies. Agentic Eval Flow keeps the AEH-style
Kubernetes client + import-path plugin instead.

---

## Migration map

| Fork (old) | Current |
|---|---|
| `pip install git+…/skills_eval_corrections@…` | `harbor==0.20.0` in eval-base / fallback pip |
| `environment.type: openshift` | `environment.import_path: …OpenShiftEnvironment` |
| `kwargs.image_ref` | `task.toml` `[environment].docker_image` |
| Pipeline params `harbor-fork-url` / `harbor-fork-revision` | Removed |

See [harbor_custom_env.md](./harbor_custom_env.md) for the full runtime contract and smoke checklist.
