# Failure Handling, Retries, and Idempotency

## Retry Policy (Target — Not Yet Applied)

> **Note:** The per-task retry values below are the target policy. They
> will be added to `pipeline.yaml` once the pipeline assembly PR merges
> and per-task `retries` fields are wired. Currently `pipeline.yaml`
> only sets aggregate `spec.timeouts`.

| Task | Planned Retries | Rationale |
|---|---|---|
| `clone-repo` (ClusterTask) | 2 | Network-dependent, fully idempotent |
| `validate` | 1 | Read-only, deterministic |
| `scaffold` | 1 | Deterministic template rendering |
| `build-push` | 2 | Transient registry/network errors; Buildah is idempotent with layer caching |
| `harbor-eval` | 0 | Long-running (up to 3h), not idempotent — partial trial results would conflict with a fresh run |
| `analyze` | 1 | Reads from workspace, deterministic computation |
| `store-results` | 2 | Database transient errors; upsert logic ensures idempotency via `pipeline_run_id` uniqueness |

## Timeouts (Target — Not Yet Applied)

> **Note:** The per-task timeouts below are the target policy. They will
> be added to `pipeline.yaml` alongside the retry values. Currently only
> aggregate timeouts are set: `pipeline: 4h`, `tasks: 3h`.

| Task | Planned Timeout | Notes |
|---|---|---|
| `clone-repo` | 5m | Large repos may need adjustment |
| `validate` | 10m | Includes py_compile on all test files |
| `scaffold` | 10m | Jinja2 rendering + file copy |
| `build-push` | 30m | Two container builds (treatment + control) |
| `harbor-eval` | 3h | 20 trials x 2 variants; adjust based on task complexity |
| `analyze` | 15m | Statistical computation + report generation |
| `store-results` | 15m | Database writes + observer notifications |
| **Pipeline total** | 4h | Safety net above sum of individual timeouts |

## Non-Retryable Failures

Certain failure categories should not be retried because they indicate
a problem that will not resolve on its own:

- **Validation failures** — malformed submission, missing required files
- **Schema violations** — invalid `metadata.yaml`
- **Build failures** from syntax errors in user code
- **Harbor evaluation failures** from test assertion errors (the skill genuinely fails)

These are distinguished from transient failures (network timeouts,
registry 503s, DB connection drops) by exit code conventions:

| Exit Code | Meaning | Retry? |
|---|---|---|
| 0 | Success | -- |
| 1 | Transient/recoverable error | Yes |
| 2 | Validation/user error (non-retryable) | No |
| 3 | Infrastructure error (retryable) | Yes |

Scripts should use `sys.exit(2)` for user-facing errors to signal
Tekton that a retry would not help.

## Dead-Letter Path

When a PipelineRun fails after exhausting retries:

1. **Artifacts are retained** on the workspace PVC (not cleaned up)
2. The `abevalflow-dead-letter` PVC is provisioned and reserved for
   failed-run artifact storage. Automatic copy logic is **not yet
   implemented** — operators can manually copy artifacts from the
   workspace PVC for post-mortem analysis.
3. PipelineRun metadata remains queryable via `tkn pipelinerun describe`
   until the cleanup CronJob prunes it (keeps the 7 most recent by
   count, configurable via `PIPELINERUN_KEEP_COUNT`)

## Partial-Run Recovery

Tekton does not natively support resuming a pipeline from a specific
task. The recovery strategy is:

1. **Workspace snapshot** — the PVC retains all intermediate artifacts
   from completed tasks. A re-run with the same submission will
   overwrite these, effectively starting fresh.

2. **Harbor checkpointing** — the Harbor fork persists individual trial
   results to the workspace as they complete. If `harbor-eval` fails
   mid-way (e.g., after 15 of 20 trials), the partial `result.json`
   files are available for inspection. However, the analysis step
   expects a complete set, so a re-run of `harbor-eval` is needed.

3. **Manual re-trigger** — use `tkn pipeline start` with the same
   parameters to re-run the full pipeline. Since all tasks before the
   failure point are idempotent, they will complete quickly using
   cached layers (builds) or deterministic outputs (scaffold).

## Concurrency

- **PipelineRuns** — no built-in Tekton limit; use `ResourceQuota` on
  the namespace (`config/security/resource_quota.yaml`) to cap total
  pods, which indirectly limits concurrent runs.
- **Trial Pods** — Harbor's `OpenShiftEnvironment` controls concurrency
  via its `max_concurrent` parameter in the job config.
