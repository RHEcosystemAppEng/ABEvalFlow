# Observability Tables Reference

Database tables added by the observability layer (APPENG-5370, Phases 0 and A) to enable SQL queries over evaluation results that previously lived only as JSON files on MinIO.

## Schema Diagram

```
evaluation_runs (pre-existing)
    │
    ├── trials (pre-existing, 1:many)
    │
    ├──→ scorecards (1:1 via pipeline_run_id)
    │        │
    │        ├── gate_results (1:many, CASCADE delete)
    │        │
    │        └── certifications (1:3 max, CASCADE delete)
    │
    └──→ observability_metrics (1:1 via pipeline_run_id)
```

## Pipeline Integration

These tables are populated automatically in the **store** step of the Tekton pipeline:

1. `store_results.py` reads `scorecard.json` → writes to `scorecards`, `gate_results`, `certifications`
2. `store_results.py` reads `_metrics_checkpoint.json` → writes to `observability_metrics`
3. The `_metrics_checkpoint.json` is created in the **test** phase finalize step from quality review token data

No configuration needed — the `enable-scorecard` pipeline flag defaults to `true`.

---

## Table: `scorecards`

One row per pipeline run. Stores the unified gate verdict combining all evaluation gates (engine, security, quality) with the applied policy.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key, auto-generated |
| `pipeline_run_id` | VARCHAR(255) | No | Tekton PipelineRun name. Unique — one scorecard per run |
| `submission_name` | VARCHAR(255) | No | Name of the evaluated skill/agent submission |
| `eval_engine` | VARCHAR(50) | No | Evaluation engine used: harbor, ase, mcpchecker, a2a, both |
| `recommendation` | VARCHAR(20) | No | Unified verdict: pass, warn, or fail |
| `recommendation_reason` | TEXT | No | Human-readable explanation of why this recommendation was given |
| `combination_mode` | VARCHAR(20) | No | How gates were combined: all_pass, any_pass, or weighted |
| `gates_passed` | INTEGER | No | Count of gates that passed |
| `gates_failed` | INTEGER | No | Count of gates that failed |
| `blocking_gates_passed` | INTEGER | No | Count of blocking (mode=block) gates that passed |
| `blocking_gates_failed` | INTEGER | No | Count of blocking gates that failed |
| `highest_certification` | VARCHAR(20) | No | Highest certification level achieved: none, foundational, trusted, or certified |
| `scorecard_json` | JSONB | No | Full Scorecard object for fast single-row retrieval and debugging |
| `created_at` | TIMESTAMP TZ | No | When the scorecard was persisted |

**Indexes:** `submission_name`, `(submission_name, created_at)`, `highest_certification`

**Relationships:** Has many `gate_results` and `certifications` (CASCADE delete).

---

## Table: `gate_results`

One row per gate per pipeline run. Stores normalized results for each evaluation gate (engine, security, quality), enabling queries like "gate pass rate by type" without parsing JSON.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key, auto-generated |
| `scorecard_id` | UUID | No | Foreign key to `scorecards.id` (CASCADE delete) |
| `gate_name` | VARCHAR(100) | No | Category name of the gate: evaluation, security, or quality |
| `gate_type` | VARCHAR(20) | No | Type of gate: engine, security, or quality |
| `policy_key` | VARCHAR(100) | Yes | Implementation name for policy lookup: harbor, cisco, llm-review, etc. |
| `passed` | BOOLEAN | No | Whether the gate passed based on its threshold and mode |
| `score` | FLOAT | No | Normalized score from 0.0 (worst) to 1.0 (best) |
| `mode` | VARCHAR(20) | No | Enforcement mode applied: disabled, warn, or block |
| `threshold` | FLOAT | Yes | Threshold used to determine pass/fail, if applicable |
| `findings_count` | INTEGER | No | Number of findings (issues) discovered by this gate |
| `message` | TEXT | Yes | Human-readable summary of the gate result |
| `details_json` | JSONB | Yes | Engine-specific payload (e.g. AnalysisResult for engine gates) |
| `duration_ms` | INTEGER | Yes | Gate execution time in milliseconds (populated by Phase A when available) |
| `prompt_tokens` | INTEGER | Yes | LLM prompt tokens used by this gate (e.g. quality review) |
| `completion_tokens` | INTEGER | Yes | LLM completion tokens used by this gate |
| `evaluated_at` | TIMESTAMP TZ | Yes | When the gate was evaluated |

**Indexes:** `scorecard_id`, `(scorecard_id, policy_key)`, `(gate_name, passed)`, `policy_key`

**Relationships:** Belongs to one `scorecard` (CASCADE delete on parent).

---

## Table: `certifications`

One row per certification level per pipeline run. Maximum 3 rows per scorecard (foundational, trusted, certified). Tracks which certification checks passed or failed.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key, auto-generated |
| `scorecard_id` | UUID | No | Foreign key to `scorecards.id` (CASCADE delete) |
| `level` | VARCHAR(20) | No | Certification level: foundational, trusted, or certified |
| `passed` | BOOLEAN | No | Whether all required checks for this level passed |
| `checks_total` | INTEGER | No | Total number of checks required for this level |
| `checks_passed` | INTEGER | No | Number of checks that passed |
| `checks_failed` | INTEGER | No | Number of checks that failed |
| `failed_checks` | JSONB | Yes | List of check IDs that failed (e.g. ["basic_security_validation"]) |
| `details_json` | JSONB | Yes | Full per-check results for this level |

**Indexes:** `scorecard_id`, `(level, passed)`

**Constraints:** Unique on `(scorecard_id, level)` — prevents duplicate certification rows for the same level.

**Relationships:** Belongs to one `scorecard` (CASCADE delete on parent).

---

## Table: `observability_metrics`

One row per pipeline run with aggregated LLM token usage and phase timing. Enables cost tracking and performance monitoring. No foreign key to scorecards — metrics may outlive scorecard lifecycle.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key, auto-generated |
| `pipeline_run_id` | VARCHAR(255) | No | Tekton PipelineRun name |
| `submission_name` | VARCHAR(255) | No | Name of the evaluated submission |
| `model_name` | VARCHAR(100) | Yes | Primary LLM model used (e.g. claude-sonnet) |
| `pipeline_duration_ms` | INTEGER | Yes | Total pipeline wall clock time in milliseconds |
| `prepare_duration_ms` | INTEGER | Yes | Duration of the prepare phase |
| `test_duration_ms` | INTEGER | Yes | Duration of the test phase |
| `evaluate_duration_ms` | INTEGER | Yes | Duration of the evaluate phase |
| `analyze_duration_ms` | INTEGER | Yes | Duration of the analyze phase |
| `store_duration_ms` | INTEGER | Yes | Duration of the store phase |
| `total_prompt_tokens` | BIGINT | Yes | Total prompt tokens sent to the LLM across all calls |
| `total_completion_tokens` | BIGINT | Yes | Total completion tokens received from the LLM |
| `total_tokens` | BIGINT | Yes | Sum of prompt + completion tokens |
| `estimated_cost_usd` | NUMERIC(12,6) | Yes | Estimated cost in USD based on per-model token rates |
| `llm_calls_count` | INTEGER | Yes | Total number of LLM API calls made during the run |
| `trials_count` | INTEGER | Yes | Number of evaluation trials in the run |
| `attempt_number` | INTEGER | No | 1 for first run, increments on retry. Enables idempotent upserts |
| `created_at` | TIMESTAMP TZ | No | When the metrics were persisted |

**Indexes:** `submission_name`, `created_at`, `model_name`

**Constraints:** Unique on `(pipeline_run_id, attempt_number)` — prevents duplicate metrics for the same run attempt.

