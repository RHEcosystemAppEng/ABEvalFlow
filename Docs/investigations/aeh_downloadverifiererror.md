# AEH DownloadVerifierDirError Investigation

## Issue Description

When running AEH (Agent-Eval-Harness) evaluations on OpenShift, Harbor trial pods fail with:

```
DownloadVerifierDirError: Failed to download verifier directory from environment
```

**Root Cause**: OpenShift's `restricted-v2` Security Context Constraint (SCC) enforces `readOnlyRootFilesystem: true` for all containers. Harbor trial pods need to write to `/workspace` and `/tmp` directories, but these are read-only in the trial pod's root filesystem.

**Solution Requirement**: Add `emptyDir` volume mounts for `/workspace` and `/tmp` to provide writable storage that complies with OpenShift security policies.

## Constraints

1. **Cannot modify upstream code**: We must NOT modify agent-eval-harness or Harbor upstream code
2. **Must use upstream versions**: Work with upstream Harbor (v0.0.18) and agent-eval-harness
3. **Available resources**: We have cloned repositories and virtual environments available for investigation

## Environment Setup

- **AEH Image**: `quay.io/ecosystem-appeng/agent-eval-harness:v1.0.3`
- **Harbor Version**: 0.0.18 (from agent-eval-harness)
- **OpenShift SCC**: restricted-v2 (enforces read-only root filesystem)
- **Local Development**:
  - agent-eval-harness cloned at `/Users/gziv/Dev/agent-eval-harness`
  - ABEvalFlow pipeline repo at `/Users/gziv/Dev/ABEvalFlow`

## Attempts and Results

### Attempt 1: Custom OpenShiftEnvironment Class ❌

**Approach**: Created custom `OpenShiftEnvironment` class extending `KubernetesEnvironment` to add emptyDir mounts.

**Implementation**:
1. Created `abevalflow/harbor_extensions/openshift_environment.py`:
   ```python
   from agent_eval.harbor.kubernetes import KubernetesEnvironment

   class OpenShiftEnvironment(KubernetesEnvironment):
       def _pod_manifest(self, image: str, env: dict) -> dict:
           manifest = super()._pod_manifest(image, env)
           pod_spec = manifest["spec"]
           container = pod_spec["containers"][0]

           container.setdefault("volumeMounts", []).extend([
               {"name": "workspace", "mountPath": "/workspace"},
               {"name": "tmp", "mountPath": "/tmp"},
           ])
           pod_spec.setdefault("volumes", []).extend([
               {"name": "workspace", "emptyDir": {}},
               {"name": "tmp", "emptyDir": {}},
           ])

           return manifest
   ```

2. Added `abevalflow` module to AEH v1.0.3 image:
   ```dockerfile
   COPY --chown=1001:0 abevalflow /opt/agent-eval-harness/abevalflow
   ```

3. Modified `scripts/run_aeh.py` to pass `--environment-import-path`:
   ```python
   cmd.extend(["--environment-import-path",
               "abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment"])
   ```

4. Updated Tekton evaluate task to export PYTHONPATH:
   ```bash
   export PYTHONPATH="$PIPELINE_DIR:${PYTHONPATH:-}"
   ```

**Evidence from logs**:
```
Running: /opt/app-root/bin/python -m agent_eval.harbor.run --config ...
  --environment-import-path abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment

harbor: harbor run -p ... --environment-import-path
  abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment
```

**Result**: ❌ **Still fails with DownloadVerifierDirError**

The custom environment class is being used (confirmed in logs), but the error persists.

### Attempt 2: Patching eval.yaml environment.type ❌

**Approach**: Patch eval.yaml to set `environment.type` to our custom class, relying on config file instead of CLI arguments.

**Implementation**:
```python
def _patch_eval_config_for_openshift(self, config: Path, tasks_dir: Path | None) -> Path:
    eval_config["environment"]["type"] = \
        "abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment"

    # Write to same directory to preserve relative paths
    patched_config = config.parent / f"{config.stem}-openshift.yaml"
    yaml.dump(eval_config, open(patched_config, "w"))
    return patched_config
```

**Problem Discovered**: agent_eval.harbor.run has a **default value** for `--env` parameter (`"kubernetes"`), so even without passing `--env` explicitly, it uses the default environment mapping which overrides the config file.

**Result**: ❌ **Did not work** - reverted to explicit `--environment-import-path` approach

## Current Status

**Commit**: `a8263f6` - `fix: pass --environment-import-path explicitly for OpenShiftEnvironment`

**Test Run**: `aeh-v103-test-whsz7`

**Status**: ❌ **Still failing with DownloadVerifierDirError**

Despite successfully:
- Creating custom OpenShiftEnvironment class
- Including abevalflow module in v1.0.3 image
- Passing --environment-import-path correctly
- Confirming the custom class is being used (from logs and trial `config.json`)

### Step 1 evidence (debug tarball `2026-07-16__13-51-50.tar.gz`)

Confirmed from `case-001__XpJxhiR/exception.txt` / `result.json`:

- Environment import path used: `abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment`
- Wrapper: `DownloadVerifierDirError: Failed to download verifier directory from environment`
- **`__cause__`**: `tarfile.ReadError: empty file` while AEH `KubernetesEnvironment.download_dir` opens the tar stream from:
  `tar cf - -C /logs/verifier . | base64 -w0`
- Host `verifier/` directory in the job artifact is **empty**
- Related: `download_file /logs/artifacts: No such file or directory` (best-effort artifact download)
- Agent did reach `/workspace` and teed to `/logs/agent/claude-code.txt` (agent then failed with `Unknown command: /aeh-hello-world-single` — separate from verifier FS)

Interpretation: download of `/logs/verifier` returned empty stdout (typical when `tar` fails on a missing/unreadable directory but `base64` still exits 0). Current `OpenShiftEnvironment` mounts only `/workspace` and `/tmp`, not Harbor paths `/logs`, `/tests`, `/solution`.

## Step 2 KEEP_RUN probe (`aeh-keeprun-probe-zdksm`)

Trial pod `aeh-case-001-bmkubbm-env`:

| Check | Result |
|---|---|
| `OpenShiftEnvironment` mounts present | Yes: emptyDir `/workspace`, `/tmp` |
| `readOnlyRootFilesystem` | **Not set** — root overlay is `rw` |
| Writability `/logs|/tests|/solution|/tmp|/workspace` | **All writable** without extra mounts |
| `/logs/verifier` after failed trial | **Missing** |
| `tar … \| base64` on missing dir | `out_len=0` (base64 masks tar failure) |

**Decision gate:** emptyDir expansion for `/logs` is **not** the root cause on this cluster.

## Confirmed root cause

Harbor **shared-verifier** mode builds:
`/tests/test.sh > /logs/verifier/test-stdout.txt`
**before** `test.sh` runs. Shared verifier does **not** call `empty_dirs(/logs/verifier)` (only separate-verifier mode does).

Reproduced on the kept pod:
1. Without `/logs/verifier` → redirect fails → `test.sh` never runs → download empty → `DownloadVerifierDirError`
2. With `mkdir -p /logs/verifier` first → `test.sh` produces `reward.json` successfully

## Fix

In `abevalflow/harbor_extensions/openshift_environment.py`:
1. After `start()`, `mkdir -p` Harbor paths including `/logs/verifier` (no chmod — SCC UIDs get EPERM)
2. Override `download_dir` with `set -o pipefail` so missing dirs fail clearly
3. Keep `/workspace`+`/tmp` emptyDirs; log injected mounts
4. `aggregate_aeh.py` emits `AnalysisResult`-compatible fields so analyze does not crash
5. `aggregate_scorecard.py` accepts `eval-engine=aeh`; `AEHEngine` handles `None` mean_reward

## Retest (`aeh-verifier-fix3-k9mbp`)

Blocked initially by PVC quota (`persistentvolumeclaims=10`); freed by deleting 5 old failed PipelineRuns and clearing stuck Terminating PVC finalizers.

**Result:** `DownloadVerifierDirError` **gone**. Trial completed with rewards:
- Exceptions: 0
- Exit_Success: 1.000
- File_Created: 0.000 / Reward: 0.000 (real eval miss — agent did not create `output/greeting.txt`)
- `step-aeh-eval` exits 1 due to AEH `REGRESSIONS` detection (not infra FS)

Remaining non-blocker: evaluate task fails on regression exit code; analyze skipped because evaluate failed.

## Files Changed

- `abevalflow/harbor_extensions/__init__.py` (created)
- `abevalflow/harbor_extensions/openshift_environment.py` (created)
- `containers/agent-eval-harness/Containerfile` (modified - added abevalflow COPY)
- `scripts/run_aeh.py` (modified - added environment-import-path logic)
- `pipeline/tasks/phases/evaluate.yaml` (modified - added PYTHONPATH export)
- `pipeline/pipelines/ci-pipeline-dev.yaml` (modified - updated to v1.0.3)

## Key Learnings

1. **agent_eval.harbor.run argument precedence**:
   - `--environment-import-path` (highest)
   - `--env` with default value "kubernetes"
   - config file `environment.type` (lowest)

2. **Relative paths in config files**: When patching eval.yaml, must write to same directory as original to preserve relative paths (cases/, skills/, etc.)

3. **Module availability**: Custom environment classes must be importable where agent_eval.harbor.run executes (dispatcher pod), achieved via PYTHONPATH and image COPY

## References

- OpenShift restricted-v2 SCC: https://docs.openshift.com/container-platform/4.x/authentication/managing-security-context-constraints.html
- Harbor KubernetesEnvironment: `/Users/gziv/Dev/agent-eval-harness/agent_eval/harbor/kubernetes.py`
- agent_eval.harbor.run: `/Users/gziv/Dev/agent-eval-harness/agent_eval/harbor/run.py`
