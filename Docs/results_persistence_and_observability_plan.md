# Results Persistence & Observability Plan

> **Jira:** APPENG-4985 (Results Persistence & Storage)
> **Branch:** `APPENG-4985/results-persistence`
> **Depends on:** APPENG-4907 (`abevalflow/report.py` — `AnalysisResult` model)

---

## Goal

Persist every A/B evaluation run to PostgreSQL so results are queryable, comparable over time, and available to downstream consumers (dashboards, CI gates, trend alerts). Provide a pluggable observer interface so observability backends (MLflow, Langfuse, OTel, etc.) can be added later without structural changes.

---

## Scope — What This Ticket Covers (ABEvalFlow side)

| Deliverable | Description |
|---|---|
| DB schema | SQLAlchemy 2.0 models for `evaluation_runs` and `trials` tables |
| DB engine | Connection factory, `create_all()` bootstrap |
| `store_results.py` | CLI script: reads `report.json`, persists to PostgreSQL |
| `query_results.py` | CLI script: historical queries (`list`, `latest`, `history`, `compare`) |
| Observer protocol | Pluggable `ResultsObserver` interface for future backends |
| Tekton task | `store-results.yaml` — runs after `analyze-report` |
| OpenShift manifests | PostgreSQL StatefulSet, Service, Secret template, PVC |
| RBAC update | Pipeline SA gets read access to DB Secret |
| Tests | Full coverage using SQLite in-memory (no PostgreSQL needed for CI) |

## Scope — What This Ticket Does NOT Cover

| Out of scope | Where it belongs |
|---|---|
| LLM call tracing inside trial pods | Harbor fork (see "Harbor Side" below) |
| MLflow / Langfuse adapter implementation | Follow-up ticket (observer interface is ready) |
| Alembic migrations | Add when schema evolves |
| Grafana dashboards | Follow-up (reads from same PostgreSQL) |
| OTel collector deployment | Follow-up (infrastructure concern) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Tekton Pipeline                                         │
│                                                         │
│  analyze-report ──► store-results ──► publish (future)  │
│       │                   │                             │
│       ▼                   ▼                             │
│  report.json         PostgreSQL                         │
│                        │    │                           │
│                        │    └──► Observer(s) [optional]  │
│                        │         ├─ MLflow              │
│                        │         ├─ Langfuse            │
│                        │         └─ OTel exporter       │
│                        ▼                                │
│                   query_results.py                      │
│                   Grafana (future)                      │
└─────────────────────────────────────────────────────────┘
```

---

## 1. Database Schema

### Table: `evaluation_runs`

One row per pipeline run. Flattened summary for fast queries.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default gen | |
| `submission_name` | VARCHAR(255) | NOT NULL, indexed | |
| `pipeline_run_id` | VARCHAR(255) | UNIQUE, NOT NULL | Tekton PipelineRun name (or content hash for CLI) |
| `commit_sha` | VARCHAR(64) | nullable | |
| `treatment_image_ref` | TEXT | nullable | Digest-based ref |
| `control_image_ref` | TEXT | nullable | |
| `harbor_fork_revision` | VARCHAR(64) | nullable | |
| `recommendation` | VARCHAR(10) | NOT NULL | `pass` / `fail` |
| `uplift` | FLOAT | NOT NULL | treatment - control pass rate |
| `mean_reward_gap` | FLOAT | nullable | |
| `ttest_p_value` | FLOAT | nullable | |
| `fisher_p_value` | FLOAT | nullable | |
| `treatment_n_trials` | INT | NOT NULL | |
| `treatment_n_passed` | INT | NOT NULL | |
| `treatment_n_failed` | INT | NOT NULL | |
| `treatment_n_errors` | INT | NOT NULL | |
| `treatment_pass_rate` | FLOAT | NOT NULL | |
| `treatment_mean_reward` | FLOAT | nullable | |
| `treatment_median_reward` | FLOAT | nullable | |
| `treatment_std_reward` | FLOAT | nullable | |
| `control_n_trials` | INT | NOT NULL | |
| `control_n_passed` | INT | NOT NULL | |
| `control_n_failed` | INT | NOT NULL | |
| `control_n_errors` | INT | NOT NULL | |
| `control_pass_rate` | FLOAT | NOT NULL | |
| `control_mean_reward` | FLOAT | nullable | |
| `control_median_reward` | FLOAT | nullable | |
| `control_std_reward` | FLOAT | nullable | |
| `report_json` | JSON | NOT NULL | Full `AnalysisResult`; uses `JSONB` on PostgreSQL via `with_variant` |
| `created_at` | TIMESTAMP TZ | NOT NULL, default now | |

**Indexes:** `submission_name`, composite `(submission_name, created_at)`.

### Table: `trials`

One row per trial. Enables drill-down queries.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default gen | |
| `run_id` | UUID | FK → `evaluation_runs.id`, CASCADE DELETE | |
| `variant` | VARCHAR(20) | NOT NULL | `treatment` / `control` |
| `trial_name` | VARCHAR(255) | NOT NULL | |
| `reward` | FLOAT | nullable | None = error/unparseable |
| `passed` | BOOLEAN | NOT NULL | |
| `created_at` | TIMESTAMP TZ | NOT NULL, default now | |

**Indexes:** `run_id`, composite `(run_id, variant)`.

---

## 2. DB Engine — `abevalflow/db/engine.py`

- `get_engine(url: str | None = None) -> Engine` — creates SQLAlchemy engine from URL
- `init_db(engine: Engine)` — `Base.metadata.create_all(engine)`, retries up to 5 times with exponential backoff on `OperationalError` (via `tenacity`) for cold-start resilience; non-transient errors (auth, SSL) fail immediately
- `make_session(engine: Engine) -> sessionmaker[Session]` — returns a session factory bound to the given engine
- Connection URL from `DATABASE_URL` env var
- Format: `postgresql+psycopg://user:pass@host:5432/abevalflow`
- Falls back to SQLite for local dev / testing

---

## 3. Observer Protocol — `abevalflow/db/observer.py`

```python
from typing import Protocol
import uuid
from abevalflow.report import AnalysisResult

class ResultsObserver(Protocol):
    """Pluggable hook called after results are persisted to the DB."""

    def on_evaluation_stored(
        self,
        result: AnalysisResult,
        run_id: uuid.UUID,
    ) -> None:
        """Called after a successful DB commit.

        Implementations may log to MLflow, push to Langfuse, emit OTel
        spans, post a Slack notification, etc. Failures in observers
        are logged but do not fail the pipeline.
        """
        ...
```

**Loading:** Observers are discovered via env vars. All matching observers run
(not first-match):
- `MLFLOW_TRACKING_URI` set → load `MLflowObserver` (future)
- `LANGFUSE_PUBLIC_KEY` set → load `LangfuseObserver` (future)
- No env vars → no observers, PostgreSQL only

**Error isolation:** Each observer runs in a `try/except` — observer failures are logged as warnings, never fail the pipeline.

---

## 4. Store Script — `scripts/store_results.py`

```
python scripts/store_results.py \
  --report-dir /path/to/report-dir \
  --database-url postgresql+psycopg://user:pass@host:5432/abevalflow
```

The script reads `{report-dir}/report.json`. This matches `analyze-report`'s output
which emits a directory containing `report.json` and `report.md`.

Logic:
1. Load and validate `{report-dir}/report.json` via `AnalysisResult.model_validate_json()`
2. Map `AnalysisResult` → `EvaluationRun` row (flatten provenance + summary)
3. Map each trial → `Trial` row (persist from `TrialResult.model_dump()` which includes the computed `passed` field — same rule as the Pydantic model)
4. Single transaction — all-or-nothing
5. **Idempotent:** if `pipeline_run_id` already exists, log warning and skip
6. After commit, invoke all registered `ResultsObserver` instances
7. Exit 0 on success, exit 1 on failure

**Idempotency key:** `pipeline_run_id` is required in the Tekton task (always
available). For standalone CLI usage, provide `--run-id` explicitly; if omitted,
the script computes a content hash of the report JSON as a deterministic fallback
key. This guarantees deduplication even without Tekton.

**Security:** `DATABASE_URL` and credentials are never logged. The script logs
the host/db name only (masked connection string).

---

## 5. Query Script — `scripts/query_results.py`

```
python scripts/query_results.py list
python scripts/query_results.py latest my-submission
python scripts/query_results.py history my-submission
python scripts/query_results.py compare my-submission
```

| Subcommand | Output |
|---|---|
| `list` | All submissions with latest result (name, recommendation, uplift, date) |
| `latest <name>` | Detailed latest run for a submission |
| `history <name>` | All runs for a submission (date, recommendation, uplift, pass rates, p-values) |
| `compare <name>` | Trend view: pass_rate and uplift over time (regression highlighting deferred to v2) |

Output: formatted table to stdout (simple column alignment, no heavy dependency).

---

## 6. Tekton Task — `pipeline/tasks/store-results.yaml`

- Runs after `analyze-report` in the pipeline
- Params: `report-dir`, `submission-name`, `pipeline-run-id`, `pipeline-repo-url`, `pipeline-repo-revision`
- DB credentials from Kubernetes Secret (`ab-eval-db-credentials`) mounted as env vars
- Sets `export PYTHONPATH="$PIPELINE_DIR"` before running scripts (same as `analyze-report`)
- Uses the same `git clone / fetch / checkout` pattern as `analyze-report` to avoid stale code
- Installs `sqlalchemy psycopg[binary] pydantic tenacity`
- Runs `scripts/store_results.py --report-dir ... --run-id $(params.pipeline-run-id)`
- `pipeline-run-id` is required in the Tekton task (always available from `$(context.pipelineRun.name)`)

---

## 7. OpenShift Manifests — `config/postgres/`

| File | Content |
|---|---|
| `config/postgres/statefulset.yaml` | PostgreSQL 16 StatefulSet (single replica MVP) with inline `volumeClaimTemplates` (10Gi) |
| `config/postgres/service.yaml` | ClusterIP Service (`ab-eval-db:5432`) |
| `config/postgres/secret.yaml` | Secret template (placeholder values — never real credentials) |

Update `config/rbac.yaml`: grant pipeline SA read access to `ab-eval-db-credentials` Secret.

**CrunchyData PGO:** The StatefulSet is designed so it can be replaced by a CrunchyData PostgresCluster CRD later (APPENG-4910) without schema changes.

**MVP limitations:** Single replica, no HA, no automated backups. HA/backup hardening is tracked in APPENG-4910 (Infrastructure & Operations).

**Pull secret:** The `registry.redhat.io` PostgreSQL image requires a pull secret on the cluster. Ensure `imagePullSecrets` is configured on the namespace or ServiceAccount.

---

## 8. Dependencies

Add to `pyproject.toml`:
- `sqlalchemy>=2.0`
- `psycopg[binary]>=3.1`

---

## 9. Tests

| Test file | Coverage |
|---|---|
| `tests/test_db_models.py` | Model creation, relationships, constraints, cascade delete |
| `tests/test_store_results.py` | Store, idempotency, observer invocation, error handling |
| `tests/test_query_results.py` | All subcommands against seeded data |

All unit tests use **SQLite in-memory / file-based** — no PostgreSQL required for CI.
An optional PostgreSQL integration test (deferred — add `@pytest.mark.skipif`
when PostgreSQL is deployed) would validate dialect-specific behavior (JSONB, UUID).

---

## 10. APPENG-4907 Dependency

`abevalflow/report.py` (defines `AnalysisResult`) lives on `APPENG-4907/analysis-reporting` (not yet merged to main). Strategy:

- **Prefer taking from `main`** once APPENG-4907 merges
- **Cherry-picked** onto this branch as commit `9567429` (from `APPENG-4907/analysis-reporting`)
- Both branches produce the identical file — git auto-merges cleanly when both land on main
- The store script imports `AnalysisResult` directly for type-safe validation at ingest time

---

## Commit Plan (4 commits)

1. **`feat: add report models (cherry-pick from APPENG-4907)`**
   - `abevalflow/report.py`

2. **`feat: add DB schema, engine, and observer protocol`**
   - `abevalflow/db/__init__.py`
   - `abevalflow/db/models.py`
   - `abevalflow/db/engine.py`
   - `abevalflow/db/observer.py`
   - `pyproject.toml` (dependencies)
   - `tests/test_db_models.py`

3. **`feat: add store and query scripts with tests`**
   - `scripts/store_results.py`
   - `scripts/query_results.py`
   - `tests/test_store_results.py`
   - `tests/test_query_results.py`

4. **`feat: add store-results Tekton task and PostgreSQL manifests`**
   - `pipeline/tasks/store-results.yaml`
   - `config/postgres/statefulset.yaml` (includes `volumeClaimTemplates` for PVC)
   - `config/postgres/service.yaml`
   - `config/postgres/secret.yaml`
   - `config/rbac.yaml` (update)

---

## Harbor Side — Observability Integration Points

> This section documents what the Harbor fork needs for full observability.
> These changes are **not** part of APPENG-4985 — they belong in the Harbor fork repo.

### Layer 1: Trial-Level LLM Tracing (Harbor Fork)

Each Harbor trial pod runs an LLM agent. To trace those calls:

1. **OTel SDK in trial pods** — add `opentelemetry-sdk` + `opentelemetry-exporter-otlp` to the trial container's dependencies
2. **Configure via env vars** — pass `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME` through Harbor's `environment_kwargs` or task config
3. **Auto-instrumentation** — LiteLLM (if used) has OTel auto-instrumentation; otherwise manual spans around agent calls
4. **Backend routing** — the OTel collector routes traces to whichever backend is deployed:
   - Langfuse (supports OTel ingestion)
   - MLflow (uses OTel internally for tracing)
   - Grafana Tempo / Jaeger (native OTel backends)

### Layer 2: Job-Level Metrics (Harbor Fork)

Harbor already writes `result.json` per trial with `verifier_result.rewards.reward`, timing, and token usage. To surface aggregate metrics:

1. **Post-job callback** — after all trials complete, Harbor could call a webhook or write a summary JSON
2. **MLflow integration** — the existing `log_to_mlflow.py` pattern (currently a standalone script in the local clone) could be integrated into Harbor's CLI as `harbor log --backend mlflow`
3. **OTel metrics** — emit gauge/counter metrics for pass rate, mean reward, trial count using OTel Metrics API

### Layer 3: Connecting Both Layers

The `pipeline_run_id` (Tekton run name) is the join key:
- ABEvalFlow stores it in `evaluation_runs.pipeline_run_id` (PostgreSQL)
- Harbor trial pods can set it as an OTel resource attribute (`pipeline.run.id`)
- Any observability backend can correlate evaluation results with individual LLM traces using this key

### Recommended Sequence

1. **Now (APPENG-4985):** PostgreSQL + observer protocol in ABEvalFlow ← this ticket
2. **Next:** Pick an observability backend (MLflow or Langfuse) and implement one `ResultsObserver` adapter (~50 lines)
3. **Then:** Add OTel SDK to Harbor trial pods for LLM call tracing
4. **Finally:** Grafana dashboards over PostgreSQL for trend monitoring
