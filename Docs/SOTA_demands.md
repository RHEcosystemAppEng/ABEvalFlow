# SOTA Demands: CI and Monitoring Pipeline

Assessment of ABEvalFlow's CI (`abevalflow-pipeline`) and monitoring (`abevalflow-monitoring-pipeline`) against state-of-the-art expectations for agent, tool, skill, and MCP evaluation platforms.

**Scope:** Gap analysis and recommended roadmap. Does not replace existing ADRs or implementation plans.

---

## Executive Summary

ABEvalFlow is **SOTA on breadth**: multi-engine evaluation (Harbor, ASE, MCPChecker, A2A) on OpenShift with persistence, statistical A/B analysis, and a separate monitoring path.

It is **not yet SOTA on enforcement, observability depth, or lifecycle closure**:

- CI does not fully **block** bad submissions before expensive evaluation in all cases.
- Monitoring does not yet **detect** regressions at production ML observability maturity.
- The platform does not yet **close the loop** from eval results → optimization → certified marketplace artifact.

The single highest-impact gap: a **formal gate policy + merge-blocking GitHub integration + unified scorecard** that turns "we ran eval" into "this artifact is certified for the marketplace."

> **Update (2026-06-16):** Unified scorecard implemented in `unified-scorecard-factory` branch.
> Engine, security, and quality gates now combine into a single `scorecard.json` with
> configurable `gate_policy` in `metadata.yaml`. See `abevalflow/scorecard.py` and
> `scripts/aggregate_scorecard.py`.

---

## What Is Already Strong (SOTA-Aligned)

| Area | Current capability |
|------|-------------------|
| **Multi-engine routing** | Harbor, ASE, `both`, MCPChecker, A2A in one Tekton shell |
| **Layered pre-eval gates** | Schema validation, Cisco security scan (warn/block), LLM quality review |
| **High-fidelity path** | Harbor A/B on OpenShift, prebuilt images, statistical analyze (uplift + p-values) |
| **Fast path** | ASE for PR-scale feedback |
| **MCP path** | MCPChecker for tool-use correctness (distinct from skill A/B) |
| **Persistence** | PostgreSQL + MinIO + Tekton results + PR comment publish |
| **Monitoring skeleton** | Separate pipeline, canary ConfigMap, degradation ratio + Slack |
| **AI-assisted authoring** | Generate instruction, tests, or evals in prepare phase |

This combination is ahead of most approaches that only lint or unit-test skill files.

---

## Critical Gaps

### 1. CI Gates Are Not Strict Enough for Marketplace Sign-Off

| Issue | Detail |
|-------|--------|
| **Quality review is advisory** | `test_quality_review.py` always exits 0. A `fail` recommendation does not stop the pipeline. |
| **`tests-passed` is unused** | `evaluate` runs whenever `test` completes. Only a failing **security-scan step** (block mode) reliably blocks downstream work. |
| **No GitHub required check** | Feedback is PR comments via `publish.py`, not merge-blocking commit statuses. |
| **No unified marketplace policy** | No single certified/rejected verdict combining security, quality, uplift, significance, and engine-specific criteria. |

**SOTA expectation:** Tiered gates (PR fast path vs marketplace hard gate) with enforced skip of expensive evaluate on soft failures where configured.

---

### 2. No Shared Eval Contract Across Engines

- Harbor, ASE, MCPChecker, and A2A use different artifact shapes.
- No **`eval.yaml`-style unified contract** (ODH agent-eval-harness / WG #8 direction).
- **`both` mode** lacks a unified cross-engine scorecard with one pass/fail semantics.

**SOTA expectation:** One submission contract with engine plugins, not four parallel formats.

---

### 3. Observability Is Storage-Heavy, Insight-Light

**Implemented:** Postgres rows, MinIO artifacts, Tekton pipeline results.

**Missing:**

| Gap | Why it matters |
|-----|----------------|
| **Grafana dashboards** | Trend lines, fleet view, engine comparison |
| **MLflow / Langfuse / OTel** | Observers stubbed in `abevalflow/db/observer.py` |
| **Token/cost/latency per run** | Required for agent eval at scale (see APPENG-5370) |
| **Trial-level trace UI** | Harbor trajectories exist in artifacts but lack a first-class explorer |
| **Reproducibility bundle** | Pin model ID, prompt hash, harness version, dataset version in one provenance record |

**SOTA expectation:** Every run is queryable, comparable, and attributable to model, infra, and harness version.

---

### 4. Monitoring Is MVP, Not Production-Grade Regression Detection

| Topic | Current behavior |
|-------|------------------|
| **Test phase** | Skipped in monitoring pipeline (no security re-scan on canary runs) |
| **Degradation logic** | Last-run vs previous-run ratio (`monitor.py`, default 0.85) |
| **Statistics** | No CUSUM, rolling baseline, or seasonality (noted as post-MVP in APPENG-4911) |
| **Schedule** | Cron every 10 days — coarse |
| **Triggers** | No webhook on model upgrade, prompt change, or agent deploy |
| **Canary config** | ConfigMap — not a versioned baseline registry or dedicated canary repo |
| **CI linkage** | CI has `enable-degradation-check: false`; monitoring and CI use disconnected policies |

**SOTA expectation:** Event-driven canaries (model/prompt/infra change) + statistical process control + fleet dashboards + auto-ticket on regression.

---

### 5. Eval Depth Gaps for Agent / Tool / MCP SOTA

| Gap | Detail |
|-----|--------|
| **Static security only** | Cisco scan on skill content — no dynamic red-team or adversarial eval during agent runs |
| **No skills quality linter in CI** | skillsaw-style structural lint not integrated |
| **Snyk / dependency scan** | Designed (APPENG-5305) but not wired like Cisco in the current test phase |
| **Single model per run** | No matrix (model × agent × engine) |
| **Flake handling** | No pass@k reporting, retry-aware stats, or infra-error exclusion in marketplace gate |
| **Multi-task benchmarks** | One submission ≈ one task; no curated benchmark suite execution |
| **MCPChecker bypasses test phase** | MCP submissions skip security and quality gates |
| **A2A pass criteria** | Heuristic threshold — not comparable to Harbor A/B semantics |

**SOTA expectation:** Layered eval — lint → static security → functional eval → adversarial probe → monitoring canary.

---

### 6. Closed Loop Missing (Authoring → CI → Optimization → Marketplace)

The pipeline narrative includes GEPA, SkillOps, and skillberry-style authoring, but the platform does not yet:

- Ingest optimized skill iterations automatically after authoring frameworks run.
- Run **eval-driven optimization** (ODH `/eval-optimize` pattern).
- Promote certified artifacts to marketplace/agentic-collections with signed provenance.
- Apply different policies for **published** skills vs **candidate** PR skills.

**SOTA expectation:** CI is the enforcement layer in a larger skill lifecycle, not the terminal step.

---

## CI vs Monitoring Asymmetries

| Topic | CI pipeline | Monitoring pipeline | SOTA alignment |
|-------|-------------|---------------------|----------------|
| Test / security | Yes (MCPChecker skipped) | **No** | Monitoring should run lightweight checks or trust pinned baselines explicitly |
| Degradation check | **Off** | **On** | CI could warn; monitoring should alert |
| AI generation | Optional | Off | Appropriate |
| PR publish | Yes | No | Appropriate |
| Trigger | PR webhook (`eval/<name>`) | CronJob | Add event triggers for monitoring |

---

## Recommended Roadmap

### P0 — Marketplace Trust (3–6 weeks)

1. **Enforced gate matrix** — Skip `evaluate` when `tests-passed=false`. Add `marketplace-mode` with stricter thresholds (uplift + p-value + security block).
2. **GitHub commit status / required check** — Not only PR comments.
3. ✅ **Unified scorecard schema** — `scorecard.json` with `gates[]` breakdown combining engine, security, and quality gates. Configurable via `gate_policy` in `metadata.yaml`. *(Implemented 2026-06-16)*
4. ✅ **Quality review blocking mode** — Gates support `disabled`, `warn`, or `block` modes. *(Implemented 2026-06-16)*

### P1 — Observability and Monitoring (6–10 weeks)

5. **Grafana dashboards** on PostgreSQL (pass rate, uplift, p-value, degradation, cost).
6. **Token and time tracking** per run and per trial.
7. **Monitoring v2** — Rolling baseline, CUSUM, triggers on model/prompt/agent change, Jira or Slack ticket on regression.
8. **MLflow or Langfuse observer** — Implement at least one observer behind `ResultsObserver`.

### P2 — SOTA Eval Depth (ongoing)

9. **ODH `eval.yaml` engine adapter** — Shared contract without replacing the Tekton shell.
10. **Benchmark registry** — Versioned datasets (skills, MCP tasks, agent scenarios), not only ad hoc submissions.
11. **skillsaw + Snyk** in test phase alongside Cisco.
12. **Adversarial eval pack** — Runtime prompt-injection probes, not only static scan.
13. **Multi-model matrix** — Optional PipelineRun matrix parameter.
14. **skillberry / GEPA hook** — Optional post-fail optimization pipeline branch.

---

## Gap Summary Table

| Category | Current state | SOTA target | Priority |
|----------|---------------|-------------|----------|
| Merge gating | PR comments | Required GitHub checks + tiered modes | P0 |
| Quality gate | ✅ Configurable block | Configurable block for marketplace | ✅ Done |
| Eval contract | Per-engine artifacts | Unified `eval.yaml` + plugins | P0–P1 |
| Scorecard | ✅ Unified `scorecard.json` | Unified gates + recommendation | ✅ Done |
| Observability | Postgres + MinIO | Grafana + traces + cost | P1 |
| Monitoring | Ratio vs previous run | SPC, event triggers, fleet view | P1 |
| Security depth | Static Cisco | + Snyk, skillsaw, adversarial runtime | P2 |
| Optimization loop | None in pipeline | skillberry / GEPA / ODH optimize | P2 |
| Benchmarks | Per submission | Versioned benchmark registry | P2 |

---

## Related Documentation

- [implementation_plan.md](implementation_plan.md) — Phased pipeline build-out
- [results_persistence_and_observability_plan.md](results_persistence_and_observability_plan.md) — DB, observers, Grafana follow-ups
- [trigger_models_and_experiment_types.md](trigger_models_and_experiment_types.md) — PR vs submissions-repo models
- [failure_handling.md](failure_handling.md) — Pipeline failure behavior

---

## References (Ecosystem)

| System | Relevant SOTA patterns |
|--------|------------------------|
| [agent-eval-harness](https://github.com/opendatahub-io/agent-eval-harness) | `eval.yaml`, MLflow, optimization loop, execution hooks |
| [Cisco skill-scanner](https://github.com/cisco-ai-defense/skill-scanner) | Static skill security (integrated) |
| [MCPChecker](https://github.com/mcpchecker/mcpchecker) | MCP task-based validation (integrated) |
| [skillberry-ai](https://github.com/skillberry-ai) | Skill optimization candidate (POC) |
