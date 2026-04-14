# Implementation Plan: Skill Evaluation Pipeline & Harbor Execution Strategy

> Based on [ADR: Skill Evaluation Pipeline and Harbor Execution Strategy](./ADR_Skill_Evaluation_Pipeline_and_Harbor_Execution_Strategy.txt)

---

## Overview

Build an automated, Tekton-orchestrated pipeline on OpenShift that accepts skill submissions, validates them, scaffolds treatment/control container variants, builds images, runs Harbor evaluations via a custom OpenShift backend, and produces statistical reports comparing treatment vs. control performance.

### Non-Goals

- Tessl Cloud integration — Tessl has no self-hosted option, only supports Claude via API key (not Vertex), and pricing is uncertain. Harbor (open-source, self-hostable) is the chosen evaluation engine.
- Tessl-driven evaluation runs are out of scope for this pipeline; any future integration would require a separate ADR.

### Two-Repository Model (ADR Decision #1)

This pipeline spans two repositories:

| Repository | Purpose | Contents |
|---|---|---|
| **[ABEvalFlow](https://github.com/RHEcosystemAppEng/ABEvalFlow)** (this repo) | Pipeline definitions, scripts, templates, config | Tekton YAML, Python scripts, Jinja2 templates, Harbor backend |
| **[agentic-collections](https://github.com/RHEcosystemAppEng/agentic-collections)** | Skills, tasks, tests (post-evaluation) | Persona-based plugins (`rh-sre`, `rh-developer`, `ocp-admin`, etc.), 100+ skills |

The `tasks-treatment/` and `tasks-control/` directories generated during scaffolding are **ephemeral workspace artifacts** — they exist only during a pipeline run, not as permanent directories in either repo.

### Harbor Fork

The Harbor fork is at [GuyZivRH/skills_eval_corrections](https://github.com/GuyZivRH/skills_eval_corrections), forked from `harbor-framework/harbor`. The OpenShift backend will be added here as a new environment type alongside the existing GKE, Docker, Podman, Daytona, Modal, etc.

### LLM Access Strategy (Updated from ADR)

The ADR originally specified Google Vertex AI + LiteLLM proxy. Since then, two additional options have become available:

| Mode | Agent | LLM Access | LiteLLM Required? |
|---|---|---|---|
| **Direct API key** | Claude Code (or other) | Provider API directly (Anthropic, OpenAI, etc.) | No |
| **opencode + self-hosted** | opencode wrapper | Local/self-hosted model (e.g., vLLM, Ollama) | No |
| **Vertex AI + proxy** | Claude Code | Google Vertex AI via LiteLLM proxy | Yes |

LiteLLM is **optional infrastructure**, needed only for the Vertex AI path. The pipeline and Harbor backend should be agnostic to the LLM access mode — the agent inside the trial container is configured via environment variables.

---

## Phase 0 — Project Bootstrap

### 0.1 Repository Structure

```
ABEvalFlow/
├── Docs/                           # ADR, plans, design docs
├── pipeline/                       # Tekton pipeline definitions
│   ├── pipeline.yaml               # Main Pipeline resource
│   ├── triggers/
│   │   ├── event-listener.yaml     # EventListener for git pushes
│   │   ├── trigger-template.yaml   # TriggerTemplate
│   │   ├── trigger-binding.yaml    # TriggerBinding
│   │   └── interceptor.yaml        # Filters pushes to submissions/ path
│   └── tasks/
│       ├── validate.yaml           # Step 2
│       ├── scaffold.yaml           # Step 3
│       ├── build-push.yaml         # Steps 4-5
│       ├── harbor-eval.yaml        # Step 6
│       ├── analyze-report.yaml     # Step 7
│       └── publish-store.yaml      # Step 8
├── templates/                      # Jinja2 templates for scaffolding
│   ├── Dockerfile.j2
│   ├── test.sh.j2
│   └── task.toml.j2
├── scripts/                        # Python scripts used by pipeline tasks
│   ├── validate.py                 # Validation logic (Step 2)
│   ├── scaffold.py                 # Scaffolding logic (Step 3)
│   ├── analyze.py                  # Analysis & report generation (Step 7)
│   ├── publish.py                  # Publish & store logic (Step 8)
│   ├── generate_tests.py           # AI-generated tests (optional path)
│   ├── ai_review.py                # Independent AI evaluation
│   └── monitor.py                  # Degradation detection (Phase 9)
├── config/                         # K8s manifests for supporting infra
│   ├── litellm-config.yaml         # Optional: only for Vertex AI mode
│   └── pipeline-config.yaml
├── tests/                          # Pipeline and script tests
│   ├── test_validate.py
│   ├── test_scaffold.py
│   ├── test_analyze.py
│   └── test_monitor.py
├── .claude/claude.md               # Instructions for Claude Code
├── pyproject.toml                  # Python project configuration
└── README.md
```

### 0.2 Python Project Setup

- Initialize `pyproject.toml` with dependencies: `pydantic`, `jinja2`, `pyyaml`, `matplotlib`, `scipy`, `kubernetes` (Python client), `tenacity`.
- Existing `.venv` (Python 3.13) is compatible — no need to recreate. Pipeline base image uses `ubi9/python-311` (3.11+); local dev uses the existing venv.
- Use `uv` for dependency management (consistent with Harbor fork and agentic-collections).

### 0.3 Pre-Requisites Checklist

| Dependency | Purpose | Notes |
|---|---|---|
| OpenShift Cluster | Runtime | With Pipelines operator installed; verify target version supports `restricted-v2` SCC |
| Tekton (OpenShift Pipelines) | Pipeline orchestration | Operator install |
| Submissions repository | Trigger source for skill submissions | Needs webhook configured to the EventListener route |
| EventListener route/URL | Webhook target | Exposed via OpenShift Route; URL needed for GitHub webhook setup |
| Submissions repo deploy key (read) | Clone submission on pipeline trigger | Stored as OpenShift Secret |
| Quay.io registry | Image storage | Service account + push secret |
| Harbor fork | Evaluation engine | [GuyZivRH/skills_eval_corrections](https://github.com/GuyZivRH/skills_eval_corrections) with OpenShift backend |
| LLM access | Agent inference | One of: direct API key, opencode+self-hosted, or Vertex AI+LiteLLM |
| PVC or MinIO (S3) | Artifact storage | For reports, logs, images |
| agentic-collections deploy key (write) | Publish passing skills | Token/deploy key stored as OpenShift Secret |
| LiteLLM (optional) | LLM proxy | Only needed for Vertex AI mode; deploy as HA Deployment/Service |

### 0.4 Definition of Done

- [ ] Repo structure created with all directories.
- [ ] `pyproject.toml` valid, `uv sync` succeeds.
- [ ] Virtual environment functional with all dependencies.

---

## Phase 1 — Submission & Validation (Steps 1-2)

### 1.0 Submission Contract

Canonical filename is **`instruction.md`** (not `instructions.md` — the ADR body uses the singular form; this is normative).

A skill submission directory must follow this structure:

```
my-skill-name/
├── instruction.md          # Task description (required, non-empty)
├── skills/                 # Must contain SKILL.md (required, canonical name)
├── docs/                   # Reference documentation (optional)
├── tests/
│   ├── test_outputs.py     # Verification tests (required, must compile)
│   └── llm_judge.py        # LLM-based judge (optional, must compile if present)
├── supportive/             # Mock MCPs, data files (optional, <50MB total)
└── metadata.yaml           # Description, persona, etc. (required, Pydantic-validated)
```

### 1.1 Submission Schema (`metadata.yaml`)

**Goal:** Define and enforce the skill submission metadata.

- [ ] Create a **Pydantic model** for `metadata.yaml` schema validation.
  - Fields: `schema_version` (for forward compatibility), `name`, `description`, `persona`, `version`, `author`, `tags` (optional), `generation_mode` (`manual` | `ai` — source of truth for dual path).
- [ ] Document the submission contract in a dedicated reference doc.

### 1.2 Validation Script (`scripts/validate.py`)

**Goal:** A standalone Python script that validates a submitted skill directory.

Checks to implement:
1. `instruction.md` exists and is non-empty.
2. `skills/SKILL.md` exists and is non-empty (canonical filename for agent recognition).
3. `test_outputs.py` compiles (`py_compile`).
4. `llm_judge.py` compiles if present.
5. `metadata.yaml` passes Pydantic schema validation.
6. `supportive/` total size < 50 MB.

Exit codes: `0` = pass, `1` = validation failure (with structured JSON error output).

### 1.3 Tekton Trigger Setup

**Goal:** Automatically trigger the pipeline on git push.

- [ ] Create `EventListener` that watches the **submissions repository**.
- [ ] Create `TriggerBinding` to extract repo URL, branch, commit SHA, and skill directory path. Document the exact path prefix (e.g., `submissions/`) to prevent "pipeline never fires" issues.
- [ ] Create `TriggerTemplate` that instantiates a `PipelineRun`.
- [ ] Add `Interceptor` to filter only pushes that modify the `submissions/` path prefix.

### 1.4 Validation Tekton Task (`pipeline/tasks/validate.yaml`)

- Base image: `ubi9/python-311`.
- Workspace: cloned submission repo. Enforce workspace size guard (align max with PVC sizing in Phase 8.3) to prevent oversized submissions from exhausting PVC space.
- Runs `scripts/validate.py` against the submitted skill directory.
- Emits results to pipeline (success/failure + error details).

### 1.5 Definition of Done

- [ ] Invalid submissions fail with deterministic structured JSON errors.
- [ ] All 6 validation checks pass/fail correctly.
- [ ] Tekton triggers fire on push to `submissions/` path.
- [ ] Unit tests pass for `validate.py`.

---

## Phase 2 — Scaffolding (Step 3)

### 2.1 Jinja2 Templates

**Goal:** Create templates that generate the correct Dockerfiles and supporting files.

- [x] `Dockerfile.j2` — Unified template using `copy_pairs` loop; COPYs strategy-determined directories plus common files (`tests/`, `supportive/`, `instruction.md`).
- [ ] `test.sh.j2` — Entry script that runs the agent, then executes `test_outputs.py` and optional `llm_judge.py`.
- [ ] `task.toml.j2` — Harbor task configuration.

### 2.2 Scaffold Script (`scripts/scaffold.py`)

**Goal:** Generate two complete task directories from a submission.

- Input: path to validated submission directory.
- Output (ephemeral workspace artifacts, not permanent repo dirs):
  - `tasks-treatment/<skill-name>/` — treatment variant with rendered Dockerfile, test.sh, task.toml.
  - `tasks-control/<skill-name>/` — control variant (baseline).
- Renders templates with context from `metadata.yaml`, directory inspection, and experiment strategy (which determines copy specs per variant).

### 2.3 Scaffold Tekton Task (`pipeline/tasks/scaffold.yaml`)

- Runs `scripts/scaffold.py`.
- Outputs two workspace directories for downstream build tasks.

### 2.4 Definition of Done

- [x] Both variants produced with correct Dockerfile COPY directives via strategy-driven `copy_pairs`.
- [x] Treatment variant includes strategy-determined dirs (e.g., skills/docs for skill experiments); control excludes them.
- [x] `test.sh` and `task.toml` render correctly for both variants.
- [x] Unit tests pass for `scaffold.py`.

---

## Phase 3 — Build & Push Images (Steps 4-5)

### 3.1 Build Task (`pipeline/tasks/build-push.yaml`)

**Goal:** Build both treatment and control images and push to registry.

- **Build tool constraint:** ADR Decision #5 specifies `docker buildx`. However, OpenShift clusters run CRI-O (not Docker) and do not provide a Docker daemon in pods. Using `docker buildx` inside unprivileged Tekton steps requires a Docker-in-Docker sidecar or socket mount, both of which require privileged access and contradict the security posture. **Buildah** (`buildah bud` + `buildah push`) is the standard rootless, daemonless alternative on OpenShift and runs in `ubi9` base images without privilege escalation. This constraint must be reconciled with ADR Decision #5 before implementation — likely by adopting Buildah for OpenShift.
- Builds from the scaffolded directories.
- Tags: `<registry>/<namespace>/<skill-name>:treatment-<commit-sha>` and `<registry>/<namespace>/<skill-name>:control-<commit-sha>`.
- Push to **internal OpenShift registry** for evaluation (per ADR decision #6).
- Quay promotion happens in Phase 6 (not here) to avoid double-push.

### 3.2 Registry Configuration

- [ ] Create image pull/push secrets for Quay.io.
- [ ] Configure OpenShift internal registry access for pipeline ServiceAccount.
- [ ] Define image retention policy (default: 30 days on Quay for reproducibility).
- [ ] Add `latest-treatment` / `latest-control` floating tags per skill for the monitoring pipeline. **Note:** Digest-based references remain the source of truth for reproducibility; floating tags are monitoring convenience only and may race under concurrent runs.

### 3.3 Image Reference Handoff

The `build-push` Tekton task must emit two **results** for downstream consumption:

- `treatment-image-ref` — full digest-based reference (e.g., `registry/ns/skill@sha256:...`)
- `control-image-ref` — same format

The `pipeline.yaml` wires these to the `harbor-eval` task:

```yaml
params:
  - name: treatment-image
    value: "$(tasks.build-push.results.treatment-image-ref)"
  - name: control-image
    value: "$(tasks.build-push.results.control-image-ref)"
```

Use digest-based references (not mutable tags) between tasks to avoid tag mutation between push and eval.

### 3.4 Definition of Done

- [ ] Both variants built and pushed to OpenShift internal registry.
- [ ] `treatment-image-ref` and `control-image-ref` emitted as Tekton results (digest-based).
- [ ] Push secrets functional.

---

## Phase 4 — Harbor OpenShift Backend (Step 6)

### 4.1 OpenShift Environment Backend (in Harbor fork)

**Goal:** Create a new `OpenShiftEnvironment` class extending `BaseEnvironment` (from `src/harbor/environments/base.py`) in the [Harbor fork](https://github.com/GuyZivRH/skills_eval_corrections).

The GKE backend (`src/harbor/environments/gke.py`, ~1044 lines) serves as the reference implementation. The OpenShift backend must implement the full `BaseEnvironment` interface:

| Method | GKE Implementation | OpenShift Replacement |
|---|---|---|
| `_init_client` | `gcloud container clusters get-credentials` + `load_kube_config()` | `load_incluster_config()` (in-cluster SA token) or `load_kube_config()` (local dev) |
| `_build_and_push_image` | `gcloud builds submit` (Cloud Build) | **No-op — see contract below** |
| `_image_exists` | `gcloud artifacts docker images describe` | Query OpenShift internal registry API or Quay API via `skopeo inspect` or registry HTTP API |
| `start` | Creates Pod spec, waits for ready | Same pattern, using `load_incluster_config()` instead of gcloud auth |
| `stop` | Deletes Pod, waits for termination | Same pattern |
| `exec` | `kubectl exec` via K8s stream API | Same — the `kubernetes` Python client is identical |
| `upload_file` / `upload_dir` | tar + K8s stream stdin | Same — portable across K8s distributions |
| `download_file` / `download_dir` | tar + K8s stream stdout | Same |

**`_build_and_push_image` contract:** In this pipeline, image build/push is owned by Tekton Steps 4-5. The Harbor OpenShift backend only orchestrates trial Pod lifecycle using pre-built images.

- **Input:** Immutable image reference (digest-based, received from Tekton results).
- **Behavior:** Verify image exists and is pullable using the trial ServiceAccount.
- **Output:** Return the exact image reference unchanged.
- **Failure:** Raise a typed error (`ImageNotFoundError` / `ImageNotPullableError`) with actionable message including the image ref and SA identity.

Additional requirements:
- Add `OPENSHIFT = "openshift"` to `EnvironmentType` enum in `src/harbor/models/environment_type.py`.
- Use `KubernetesClientManager` singleton pattern (same as GKE) but init via `load_incluster_config()`.
- Pod security context: `runAsNonRoot: true`, `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false`, drop all Linux capabilities, seccomp profile `RuntimeDefault`. Mount `emptyDir` volumes for writable paths agents/tests require (e.g., `/tmp`, agent cache dirs) since `readOnlyRootFilesystem` prevents writes to the root filesystem.
- Verify target cluster uses `restricted-v2` SCC (OpenShift 4.11+) or configure an equivalent.
- Proper cleanup (Pod deletion after trial completion).

### 4.2 Harbor Fork Integration

- [ ] Add `openshift_environment.py` in `src/harbor/environments/`.
- [ ] Add `OPENSHIFT` to `EnvironmentType` enum.
- [ ] Register the backend so `harbor run --env openshift` selects it.
- [ ] Pin a fork SHA in the pipeline image; review upstream quarterly for drift.

### 4.3 Testing Strategy

- **Unit/CI tests:** Mock Kubernetes API server using `pytest` + `responses`/`unittest.mock` for the `kubernetes` client. No live cluster needed.
- **Integration tests:** Test against an OpenShift developer sandbox (ROSA/OSD). Kind/Minikube won't catch SCC/Routes differences.

### 4.4 Trial Execution Configuration

- The `harbor-eval` Tekton task accepts `treatment-image-ref` and `control-image-ref` as **params** wired from Phase 3 results.
- N = configurable attempts per variant (default 20, treatment + control = 40 total sessions).
- Configure resource requests/limits per trial Pod.
- LLM endpoint configured via environment variable — backend is agnostic to whether it points to LiteLLM, a direct API, or a self-hosted model.
- Trial Pod timeout: configurable, with a global evaluation timeout.

### 4.5 RBAC Requirements

The pipeline ServiceAccount needs (prefer named Secrets for least-privilege where policy requires):

| Resource | Verbs | Purpose |
|---|---|---|
| Pods | create, get, list, watch, delete | Trial Pod lifecycle (watch for efficient state tracking) |
| ConfigMaps | get, list | Trial configuration |
| Secrets | get | LLM credentials injection via `envFrom` |
| Events | get, list | Diagnosing hung/failed trial Pods |
| PVCs | get, list, create | Pipeline workspaces and artifacts |
| ImageStreams (OpenShift) | get, list | Registry access — validate against target cluster's API version |

### 4.6 Definition of Done

- [ ] Trial Pods complete (N per variant × 2 variants, default 40 total).
- [ ] Cleanup verified — no stale Pods after evaluation.
- [ ] Retry behavior validated for transient failures.
- [ ] Unit tests pass with mocked K8s API.
- [ ] Integration test passes on OpenShift sandbox.

---

## Phase 5 — Analysis & Reporting (Step 7)

### 5.1 Analysis Script (`scripts/analyze.py`)

**Goal:** Consume Harbor output and produce a statistical report.

Metrics to compute:
- **Pass rate** per variant (treatment, control).
- **Uplift (gap):** `pass_rate_treatment - pass_rate_control`.
- **Statistical significance:** p-value via Fisher's exact test or chi-squared.
- **Heatmap generation:** matplotlib/seaborn figures saved as PNG.
- **LLM judge scores** (when `llm_judge.py` is present): include a qualitative score summary section. Define a schema for `llm_judge.py` output (JSON with `score`, `rationale`) to ensure `analyze.py` can reliably parse it.

Output: Markdown (or HTML) report with:
- Summary statistics table.
- Embedded heatmap figures.
- Links to detailed trial logs.
- Pass/fail recommendation based on configurable threshold.
- LLM judge summary (if applicable).
- Estimated vs. actual token spend (when available from LLM provider — not all modes expose usage).
- **Run provenance block:** commit SHA(s), Harbor fork SHA, image digest(s), model identifier/version, pipeline run ID, timestamp.

### 5.2 Analyze Tekton Task (`pipeline/tasks/analyze-report.yaml`)

- Collects Harbor output from workspace/PVC.
- Runs `scripts/analyze.py`.
- Stores report artifacts to PVC or S3 (MinIO).

### 5.3 Definition of Done

- [ ] Report includes uplift + p-value + artifact links + run provenance.
- [ ] Heatmaps generated and embedded.
- [ ] LLM judge scores aggregated when present.
- [ ] Unit tests pass for `analyze.py`.

---

## Phase 6 — Publish & Store (Step 8)

### 6.1 Publish Script (`scripts/publish.py`)

**Goal:** Finalize and distribute results.

Actions:
- Upload final report to artifact storage (PVC/S3).
- Redact PII/secrets from published logs and reports.
- If evaluation passed thresholds:
  - Re-tag and push images to **Quay.io** with TTL metadata (single promotion point — not in Phase 3).
  - **Commit to agentic-collections repo** (ADR Decision #4): open a PR to `agentic-collections` with the submission files (`instruction.md`, `tests/`, `metadata.yaml`, `skills/`). Target directory derived from `persona` field + normalized skill slug + version. PR includes a standardized title and body with evaluation summary and report link. Requires deploy key/token stored as OpenShift Secret.
- Post summary as a commit status or PR comment on the submissions repo.
- Clean up ephemeral resources (temporary PVCs, intermediate images).

### 6.2 Publish Tekton Task (`pipeline/tasks/publish-store.yaml`)

- Runs `scripts/publish.py`.
- Configurable: notification targets (Slack, email, GitHub status).

### 6.3 Definition of Done

- [ ] Report uploaded to artifact storage.
- [ ] Passing skills committed to agentic-collections via PR.
- [ ] Images promoted to Quay.io on pass.
- [ ] Ephemeral resources cleaned up.

---

## Phase 7 — AI-Assisted Skill Evaluation (Decision #2 & #3)

> This phase is **optional for MVP**. Use feature flags `ENABLE_AI_TEST_GENERATION` and `ENABLE_AI_QUALITY_REVIEW` to toggle.

### 7.1 Dual Submission Paths

Support two modes per ADR Decision #2. The source of truth is `metadata.yaml` field `generation_mode: manual | ai`:

1. **Manual (`generation_mode: manual`):** User provides skill + instruction + tests.
2. **AI-Generated (`generation_mode: ai`):** User provides skill only; instruction and tests are generated by an LLM.

Pipeline task order when AI features are enabled:

`[generate_tests if generation_mode=ai] → validate → [ai_review if ENABLE_AI_QUALITY_REVIEW] → scaffold → ...`

- [ ] Create `scripts/generate_tests.py` that uses the configured LLM to generate `instruction.md`, `test_outputs.py`, and optionally `llm_judge.py` from a skill definition.
- [ ] Add a conditional step in the pipeline: if `generation_mode: ai`, invoke generation before validation.
- [ ] Safeguard: if `generation_mode: ai`, ALL of `instruction.md` and `test_outputs.py` must be produced by `generate_tests.py` or the run fails. Partial manual + partial AI is not supported.

### 7.2 Independent AI Evaluation (Decision #3)

- [ ] For both paths, run an independent AI review of the skill, test, and task quality before proceeding to Harbor evaluation.
- [ ] Create `scripts/ai_review.py` that uses the LLM to assess coherence, coverage, and potential issues.
- [ ] Add as a pipeline task between validation and scaffolding.

### 7.3 Definition of Done

- [ ] AI generation produces valid submission structure when enabled.
- [ ] AI review produces structured quality assessment.
- [ ] Feature flags correctly toggle these steps.

---

## Phase 8 — Infrastructure & Operations

### 8.1 LiteLLM Deployment (Optional — Vertex AI mode only)

> **Ordering note:** If `LLM_MODE=vertex`, Phase 8.1 must complete before Phase 4 (Harbor evaluation) can run trials. For other modes, this phase is skipped entirely.

- [ ] Create Deployment manifest (HA: 2+ replicas).
- [ ] Create Service for in-cluster access.
- [ ] Mount Vertex AI credentials from OpenShift Secret.
- [ ] Configure model routing (Claude via Vertex AI + optionally self-hosted models per Decision #8).

### 8.2 RBAC & Security

- [ ] Create dedicated `ServiceAccount` for pipeline.
- [ ] Define `Role` with permissions per Phase 4.5 RBAC table.
- [ ] Create `RoleBinding`.
- [ ] **Mode-specific network egress policies** for trial Pods:
  - `vertex+litellm` mode: allow only in-cluster LiteLLM Service; deny all external egress.
  - `direct-api` mode: allowlist only provider domains (`api.anthropic.com`, `api.openai.com`, etc.) + DNS; deny all else.
  - `self-hosted` mode: allow only internal model endpoint; deny external egress.
- [ ] Pod security context for trial Pods:
  - `runAsNonRoot: true`
  - `readOnlyRootFilesystem: true` (with `emptyDir` mounts for `/tmp` and agent cache paths)
  - `allowPrivilegeEscalation: false`
  - Drop all Linux capabilities
  - Seccomp profile: `RuntimeDefault`
- [ ] Resource quotas on trial Pods (CPU, memory limits).

### 8.3 Storage & Cleanup

- [ ] Provision PVC for pipeline workspaces and artifacts.
- [ ] Alternatively, set up MinIO for S3-compatible artifact storage.
- [ ] Create a CronJob or TTL-based cleanup for:
  - Stale trial Pods.
  - Old container images (Quay retention: default 30 days).
  - Expired evaluation reports.
  - OpenShift internal registry images (after Quay promotion or on failure).

### 8.4 LLM Model Strategy (Decision #8 — Updated)

Three supported modes, configured per pipeline run:

| Mode | Configuration | Infrastructure |
|---|---|---|
| Direct API key | Set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in trial Pod env | None — agent calls provider directly |
| opencode + self-hosted | Set model endpoint in trial Pod env, use opencode as agent wrapper | Self-hosted model (vLLM, Ollama, etc.) |
| Vertex AI + LiteLLM | Set `LITELLM_BASE_URL` in trial Pod env | LiteLLM Deployment + Vertex AI credentials |

The pipeline and Harbor backend are agnostic — they pass LLM config as environment variables to trial Pods.

### 8.5 Cost Controls & Observability

A single evaluation run consumes N × 2 LLM sessions (default N=20, 40 total). Cost management is a first-class concern:

- [ ] Configure LiteLLM per-key budget limits (when using Vertex mode).
- [ ] Implement pre-flight cost estimate: before launching Harbor, estimate token usage based on skill complexity and configured N. Log the estimate to the run summary to flag potential runaway cost before spend happens.
- [ ] Implement per-run token usage tracking — surface estimated vs. actual spend in the evaluation report.
- [ ] Set up concurrency limits: max parallel `PipelineRun`s and max parallel trial Pods per namespace.
- [ ] Track metrics: pass/fail rate by skill and model, mean/95p trial duration, failure categories (validation, build, runtime, LLM, infra).
- [ ] Alert thresholds for cost spikes and error-rate spikes.

### 8.6 Failure Handling, Retries, and Idempotency

- [ ] Define Tekton `retries` policy per task (safe-to-retry list vs. non-retryable failures).
- [ ] Partial-run recovery: Tekton does not natively support "resume from task T." Options: (a) manual re-run with workspace snapshot from PVC, (b) split into smaller chained Pipelines, or (c) application-level checkpointing in Harbor (persist partial results per trial). Document chosen mechanism before implementation.
- [ ] Timeouts: per-trial Pod timeout and global evaluation timeout.
- [ ] Dead-letter path: failed runs retain artifacts for debugging.
- [ ] Persist partial results to PVC after each trial — prevents full re-run on LiteLLM HA failure.

---

## Phase 9 — Continuous Performance Monitoring

### 9.1 Platform Update Regression Testing

- [ ] Define a dedicated test set for mission-critical Insights API calls and tool-mapping expectations.
- [ ] Create a separate Tekton pipeline/trigger for upstream platform/vendor update events (e.g., Gemini/Insights stack).
- [ ] Reuse the evaluation infrastructure (containerized runs) for regression tests.
- [ ] **MCP LightSpeed regression pack** (ADR footnote [e]): Identify the source location of MCP LightSpeed tests (specific repo/path — to be inventoried during this phase). Bundle as committed fixtures in `ABEvalFlow/tests/canary/` or reference via config-driven pointer. The monitoring pipeline task should accept a `canary-test-set` param that defaults to the MCP LightSpeed pack but can be overridden.

### 9.2 Degradation Detection

**9.2a — Simple thresholds (MVP):**
- [ ] Store historical pass rates per skill in PVC/S3.
- [ ] Configure hard-coded alerting thresholds for performance drops (e.g., >10% pass-rate decline).
- [ ] Wire alerts into the notification channel from Phase 6 (Slack/GitHub status).

**9.2b — CUSUM (post-MVP hardening):**
- [ ] Implement CUSUM in `scripts/monitor.py` using configurable drift threshold (starting defaults: `k=0.5`, `h=5`).
- [ ] Input: historical pass-rate time series per skill.
- [ ] Output: boolean alert flag + CUSUM statistic at time of detection.
- [ ] Add `tests/test_monitor.py` with synthetic degradation fixtures.

---

## Implementation Order (Recommended)

| Priority | Phase | Estimated Effort | Dependencies |
|---|---|---|---|
| 1 | Phase 0 — Bootstrap | 1-2 days | None |
| 2 | Phase 8.2 — RBAC & Security | 1-2 days | OpenShift cluster (parallel with Phases 1-3) |
| 3 | Phase 1 — Validation | 2-3 days | Phase 0 |
| 4 | Phase 2 — Scaffolding | 2-3 days | Phase 1 |
| 5 | Phase 3 — Build & Push | 2-3 days | Phase 2, Registry access |
| 6 | Phase 8.1 — LiteLLM (if Vertex mode) | 1-2 days | Vertex AI credentials (must complete before Phase 4 trials if Vertex mode) |
| 7 | Phase 4 — Harbor Backend | 3-5 days | Harbor fork, Phase 3 |
| 8 | Phase 5 — Analysis | 2-3 days | Phase 4 |
| 9 | Phase 6 — Publish | 1-2 days | Phase 5, agentic-collections deploy key |
| 10 | Phase 7 — AI Assist (optional) | 3-4 days | LLM access configured |
| 11 | Phase 8.3-8.6 — Ops & Cost | 2-3 days | Phases 5-6 |
| 12 | Phase 9 — Monitoring | 2-3 days | Phases 5-6 |

**Total estimated effort: ~22-33 days** (parallelizable — RBAC/Security work runs alongside Phases 1-3; LiteLLM setup slots in before Harbor eval if using Vertex mode).

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM API costs (40 sessions per eval) | High cost at scale | Pre-flight cost estimate, per-key budget limits, cost tracking in reports, concurrency caps |
| Harbor fork divergence from upstream | Maintenance burden | Minimize changes, pin fork SHA, review upstream quarterly |
| User-submitted code security | Attack surface via `test_outputs.py`, `llm_judge.py`, mock MCPs | Mode-specific network policies, resource limits, ephemeral unprivileged Pods, PodSecurityContext hardening |
| LiteLLM HA failure during 40-session eval | All results lost, full re-run required | 2+ replicas; persist partial results to PVC after each trial |
| Agentic-collections write conflicts | Two evals passing simultaneously both try to commit | PR-based flow with conflict detection |
| `llm_judge.py` calling external services | Data exfiltration / unexpected cost | Network policy blocks all egress from trial Pods except configured LLM endpoint |
| Harbor upstream API drift | New features unavailable, security patches missed | Pin fork SHA, quarterly upstream review |
| Platform update degrades skills silently | Regression goes undetected | Phase 9 monitoring (thresholds + optional CUSUM) with automated alerting |
| `metadata.yaml` schema evolution | Breaks old submissions | `schema_version` field in Pydantic model |
| `docker buildx` incompatible with OpenShift CRI-O | Build step fails in unprivileged Tekton pods | Reconcile ADR Decision #5 with Buildah; document chosen approach |
