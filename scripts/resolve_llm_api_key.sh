#!/usr/bin/env bash
# Resolve LLM API key for pipeline-run correlation via LiteLLM proxy.
#
# When the pipeline param is empty, generates sk-run-<pipeline-run-name>.
# When a non-empty key is provided, uses it as-is (backward compatible).
#
# Usage:
#   LLM_API_KEY=$(resolve_llm_api_key.sh "$PARAM_KEY" "$PIPELINE_RUN_NAME")

set -euo pipefail

param_key="${1:-}"
run_name="${2:-${PIPELINE_RUN_NAME:-${PIPELINE_RUN_ID:-unknown}}}"

if [ -z "$param_key" ]; then
  printf 'sk-run-%s' "$run_name"
else
  printf '%s' "$param_key"
fi
