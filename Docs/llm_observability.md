# LLM Observability

ABEvalFlow routes all pipeline LLM calls through a shared LiteLLM proxy
(`http://litellm.ab-eval-flow.svc:4000`). Each PipelineRun uses a unique
correlation API key so operators can trace every LLM request for a single
evaluation run in Grafana Tempo, Jaeger, or any OTLP-compatible backend.

## Correlation Model

Each PipelineRun gets a deterministic API key:

```text
sk-run-<PipelineRun.name>
```

Example: PipelineRun `abevalflow-pipeline-run-abc12` uses key
`sk-run-abevalflow-pipeline-run-abc12`.

Clients pass this key in the standard Authorization header (or provider-specific
env vars such as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `LLM_API_KEY`). LiteLLM
accepts any key when `master_key` auth is configured; the key is used for
correlation only, not for real authentication.

### Key Resolution

Pipeline parameter `llm-api-key` controls behavior:

| Value | Behavior |
|-------|----------|
| Empty (`""`, default) | Auto-generate `sk-run-$(context.pipelineRun.name)` |
| Explicit value | Use the provided key (local dev, overrides) |

Resolution is implemented in `scripts/resolve_llm_api_key.sh` and invoked from
Tekton tasks in prepare, test, and evaluate phases.

### Components Using the Correlation Key

| Component | How the key is passed |
|-----------|----------------------|
| Harbor eval | `--llm-api-key` / `ANTHROPIC_API_KEY` |
| ASE eval | `OPENAI_API_KEY` |
| Cisco skill-scanner | `SKILL_SCANNER_LLM_API_KEY`, `OPENAI_API_KEY` |
| Test quality review | `LLM_API_KEY` |
| Test generation (prepare) | `LLM_API_KEY` |

## LiteLLM OpenTelemetry Configuration

LiteLLM exports spans via the `otel` callback. Configuration lives in
`config/litellm/litellm_config.yaml` and is deployed through
`config/litellm/configmap.yaml`.

```yaml
litellm_settings:
  success_callback: ["otel"]
  failure_callback: ["otel"]

general_settings:
  otel_exporter: otlp_http
  otel_endpoint: "http://otel-collector.monitoring.svc:4318"
  otel_headers: ""
```

The Deployment also supports environment overrides:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP HTTP endpoint | `http://otel-collector.monitoring.svc:4318` |
| `OTEL_EXPORTER_OTLP_HEADERS` | Optional auth headers for the collector | empty |

Apply config changes:

```bash
oc apply -f config/litellm/
oc rollout restart deployment/litellm -n ab-eval-flow
```

## OpenTelemetry Collector

A reference collector config is in `config/otel/collector-config.yaml`. Most
clusters already run a collector in a monitoring namespace; point LiteLLM at
that endpoint instead of deploying a second collector.

## Querying Traces

### Jaeger

1. Open Jaeger UI and select the LiteLLM service (often `litellm` or
   `litellm-proxy`).
2. Search by tag `metadata.user_api_key` or HTTP header metadata containing
   your correlation key.
3. Filter by time range matching the PipelineRun start/end.

Example Jaeger tag query:

```text
metadata.user_api_key=sk-run-abevalflow-pipeline-run-abc12
```

### Grafana (Tempo)

In Explore, use TraceQL or search:

```text
{ span.http.request.header.authorization =~ ".*sk-run-abevalflow-pipeline-run-abc12.*" }
```

Or search logs/metrics that include the LiteLLM request metadata field
`user_api_key`.

### kubectl / PipelineRun Lookup

Find the PipelineRun name, then derive the key:

```bash
RUN=abevalflow-pipeline-run-abc12
echo "sk-run-${RUN}"
```

Cross-reference with Tekton:

```bash
oc get pipelinerun -n ab-eval-flow
```

## Troubleshooting

| Symptom | Check |
|---------|-------|
| No traces in backend | LiteLLM pod logs; collector endpoint reachable from `ab-eval-flow` namespace |
| Wrong run grouped together | Confirm `llm-api-key` param is empty (auto-gen) or unique per run |
| Missing spans from one gate | Verify that task exports `LLM_API_KEY` / provider key env vars |
| Local runs use `mock` | Set `llm-api-key` param explicitly or leave empty for auto-generation |

## Related Files

- `scripts/resolve_llm_api_key.sh` — key resolution helper
- `config/litellm/litellm_config.yaml` — source LiteLLM config with OTel
- `config/litellm/configmap.yaml` — deployed ConfigMap
- `config/litellm/deployment.yaml` — LiteLLM Deployment with OTEL env vars
- `config/otel/collector-config.yaml` — reference collector pipeline
- `scripts/generate_eval_config.py` — passes `--llm-api-key` into Harbor configs
