#!/usr/bin/env bash
set -euo pipefail

# Trigger ABEvalFlow test PipelineRuns.
#
# Usage:
#   ./scripts/misc/trigger_test_runs.sh [branch-name] [all|aeh|legacy]
#
#   branch-name  Pipeline repo revision (default: current git branch)
#   mode         all     — Harbor/A2A/ASE monitoring+CI plus AEH single+pairwise (default)
#                aeh     — AEH single + pairwise only (verified smoke samples)
#                legacy  — original 6 Harbor/A2A/ASE runs only
#
# Env overrides:
#   NAMESPACE          OpenShift namespace (default: guy-ziv-evalflow)
#   PIPELINE_CI        CI pipeline name (default: abevalflow-pipeline-dev)
#   PIPELINE_MONITOR   Monitoring pipeline name (default: abevalflow-monitoring-pipeline)
#   LLM_API_BASE       LiteLLM base URL
#   AEH_IMAGE          AEH/Harbor trial image
#   SKILL_REPO         skill-submissions git URL
#
# Verified AEH samples (skill-submissions):
#   single:   revision=eval/aeh-hello-world-single   submission-dir=aeh-hello-world-single
#   pairwise: revision=eval/aeh-hello-world-pairwise submission-dir=aeh-hello-world-pairwise

BRANCH="${1:-}"
if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
fi
MODE="${2:-all}"

NAMESPACE="${NAMESPACE:-guy-ziv-evalflow}"
PIPELINE_CI="${PIPELINE_CI:-abevalflow-pipeline-dev}"
PIPELINE_MONITOR="${PIPELINE_MONITOR:-abevalflow-monitoring-pipeline}"
LLM_API_BASE="${LLM_API_BASE:-http://litellm.ab-eval-flow.svc.cluster.local:4000}"
AEH_IMAGE="${AEH_IMAGE:-quay.io/ecosystem-appeng/agent-eval-harness:v1.0.3}"
SKILL_REPO="${SKILL_REPO:-https://github.com/RHEcosystemAppEng/skill-submissions.git}"
LLM_MODEL="${LLM_MODEL:-claude-sonnet}"

echo "pipeline-repo-revision: $BRANCH"
echo "namespace:              $NAMESPACE"
echo "mode:                   $MODE"
echo "ci pipeline:            $PIPELINE_CI"
echo "llm-api-base:           $LLM_API_BASE"
echo ""

trigger_aeh() {
  echo "=== Triggering AEH single + pairwise (verified smoke samples) ==="
  cat <<EOF | oc create -n "$NAMESPACE" -f -
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: aeh-single-
spec:
  pipelineRef:
    name: $PIPELINE_CI
  params:
    - name: repo-url
      value: "$SKILL_REPO"
    - name: revision
      value: "eval/aeh-hello-world-single"
    - name: submission-dir
      value: "aeh-hello-world-single"
    - name: eval-engine
      value: "aeh"
    - name: aeh-mode
      value: "single"
    - name: pipeline-repo-revision
      value: "$BRANCH"
    - name: llm-model
      value: "$LLM_MODEL"
    - name: llm-api-base
      value: "$LLM_API_BASE"
    - name: llm-api-key
      value: "mock"
    - name: aeh-model-override
      value: "$LLM_MODEL"
    - name: aeh-image
      value: "$AEH_IMAGE"
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: [ReadWriteOnce]
          resources:
            requests:
              storage: 5Gi
---
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: aeh-pairwise-
spec:
  pipelineRef:
    name: $PIPELINE_CI
  params:
    - name: repo-url
      value: "$SKILL_REPO"
    - name: revision
      value: "eval/aeh-hello-world-pairwise"
    - name: submission-dir
      value: "aeh-hello-world-pairwise"
    - name: eval-engine
      value: "aeh"
    - name: aeh-mode
      value: "pairwise"
    - name: pipeline-repo-revision
      value: "$BRANCH"
    - name: llm-model
      value: "$LLM_MODEL"
    - name: llm-api-base
      value: "$LLM_API_BASE"
    - name: llm-api-key
      value: "mock"
    - name: aeh-model-override
      value: "$LLM_MODEL"
    - name: aeh-image
      value: "$AEH_IMAGE"
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: [ReadWriteOnce]
          resources:
            requests:
              storage: 5Gi
EOF
}

trigger_legacy() {
  echo "=== Triggering Harbor / A2A / ASE monitoring + CI (6 runs) ==="
  # Short litellm hostname for ab-eval-flow namespace; override LLM_API_BASE if needed.
  local LITELLM_SHORT="${LLM_API_BASE}"
  cat <<EOF | oc create -n "$NAMESPACE" -f -
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: harbor-test-
spec:
  pipelineRef:
    name: $PIPELINE_MONITOR
  params:
    - name: repo-url
      value: "$SKILL_REPO"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "hello-world"
    - name: eval-engine
      value: "harbor"
    - name: pipeline-repo-revision
      value: "$BRANCH"
    - name: llm-model
      value: "$LLM_MODEL"
    - name: llm-api-base
      value: "$LITELLM_SHORT"
    - name: llm-api-key
      value: "mock"
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: [ReadWriteOnce]
          resources:
            requests:
              storage: 1Gi
---
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: a2a-verify-
spec:
  pipelineRef:
    name: $PIPELINE_MONITOR
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/ABEvalFlow.git"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "a2a-agent-eval"
    - name: eval-engine
      value: "a2a"
    - name: pipeline-repo-revision
      value: "$BRANCH"
    - name: agent-endpoint
      value: "http://lightspeed-agent.ab-eval-flow.svc:8000"
    - name: llm-model
      value: "$LLM_MODEL"
    - name: llm-api-base
      value: "$LITELLM_SHORT"
    - name: llm-api-key
      value: "mock"
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: [ReadWriteOnce]
          resources:
            requests:
              storage: 1Gi
---
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ase-verify-
spec:
  pipelineRef:
    name: $PIPELINE_MONITOR
  params:
    - name: repo-url
      value: "$SKILL_REPO"
    - name: revision
      value: "eval/hello-world-full"
    - name: submission-dir
      value: "hello-world-full"
    - name: eval-engine
      value: "ase"
    - name: pipeline-repo-revision
      value: "$BRANCH"
    - name: llm-model
      value: "$LLM_MODEL"
    - name: llm-api-base
      value: "$LITELLM_SHORT"
    - name: llm-api-key
      value: "mock"
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: [ReadWriteOnce]
          resources:
            requests:
              storage: 1Gi
---
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ci-harbor-
spec:
  pipelineRef:
    name: $PIPELINE_CI
  params:
    - name: repo-url
      value: "$SKILL_REPO"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "hello-world"
    - name: eval-engine
      value: "harbor"
    - name: pipeline-repo-revision
      value: "$BRANCH"
    - name: llm-model
      value: "$LLM_MODEL"
    - name: llm-api-base
      value: "$LITELLM_SHORT"
    - name: llm-api-key
      value: "mock"
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: [ReadWriteOnce]
          resources:
            requests:
              storage: 1Gi
---
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ci-a2a-
spec:
  pipelineRef:
    name: $PIPELINE_CI
  params:
    - name: repo-url
      value: "https://github.com/RHEcosystemAppEng/ABEvalFlow.git"
    - name: revision
      value: "main"
    - name: submission-dir
      value: "a2a-agent-eval"
    - name: eval-engine
      value: "a2a"
    - name: pipeline-repo-revision
      value: "$BRANCH"
    - name: agent-endpoint
      value: "http://lightspeed-agent.ab-eval-flow.svc:8000"
    - name: llm-model
      value: "$LLM_MODEL"
    - name: llm-api-base
      value: "$LITELLM_SHORT"
    - name: llm-api-key
      value: "mock"
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: [ReadWriteOnce]
          resources:
            requests:
              storage: 1Gi
---
apiVersion: tekton.dev/v1
kind: PipelineRun
metadata:
  generateName: ci-ase-
spec:
  pipelineRef:
    name: $PIPELINE_CI
  params:
    - name: repo-url
      value: "$SKILL_REPO"
    - name: revision
      value: "eval/hello-world-full"
    - name: submission-dir
      value: "hello-world-full"
    - name: eval-engine
      value: "ase"
    - name: pipeline-repo-revision
      value: "$BRANCH"
    - name: llm-model
      value: "$LLM_MODEL"
    - name: llm-api-base
      value: "$LITELLM_SHORT"
    - name: llm-api-key
      value: "mock"
  timeouts:
    pipeline: "2h"
    tasks: "1h30m"
  taskRunTemplate:
    serviceAccountName: pipeline
  workspaces:
    - name: shared-workspace
      volumeClaimTemplate:
        spec:
          accessModes: [ReadWriteOnce]
          resources:
            requests:
              storage: 1Gi
EOF
}

case "$MODE" in
  aeh)
    trigger_aeh
    ;;
  legacy)
    trigger_legacy
    ;;
  all)
    trigger_legacy
    trigger_aeh
    ;;
  *)
    echo "Unknown mode: $MODE (use all|aeh|legacy)" >&2
    exit 1
    ;;
esac

echo ""
echo "Done! Monitor with:"
echo "  oc get pipelineruns -n $NAMESPACE --sort-by=.metadata.creationTimestamp | tail -15"
