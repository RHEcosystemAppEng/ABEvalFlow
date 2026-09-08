#!/usr/bin/env bash
set -euo pipefail
#
# Build and push the eval base image to the OpenShift internal registry.
# Uses podman locally (fast) and pushes via the external registry route.
#
# Prerequisites:
#   - podman installed
#   - logged in to OpenShift (oc whoami)
#
# Usage:
#   ./scripts/build_base_image.sh                  # with claude-code (default)
#   ./scripts/build_base_image.sh --no-claude       # without claude-code
#   ./scripts/build_base_image.sh --tag v2          # custom tag
#   ./scripts/build_base_image.sh --tag local-env   # harbor-eval / a2a runner tag
#
# When to rebuild:
#   - Harbor / AEH pin bump (edit templates/Dockerfile.base ARG defaults)
#   - claude-code version update (npm will pull latest)
#   - uv version bump (edit templates/Dockerfile.base)
#   - base OS image update (ubi9/python-312)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NAMESPACE="${NAMESPACE:-ab-eval-flow}"
IMAGE_NAME="${IMAGE_NAME:-eval-base}"
TAG="latest"
INSTALL_CLAUDE="true"
# AEH v1.20.0 — keep in sync with templates/Dockerfile.base default
AEH_SHA="${AEH_SHA:-ff8b8301d861d2cc83a46ec12c394aafb846171b}"
HARBOR_VERSION="${HARBOR_VERSION:-0.20.0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-claude)   INSTALL_CLAUDE="false"; shift ;;
    --tag)         TAG="$2"; shift 2 ;;
    --aeh-sha)     AEH_SHA="$2"; shift 2 ;;
    --harbor-version) HARBOR_VERSION="$2"; shift 2 ;;
    *)             echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ROUTE=$(oc get route -n openshift-image-registry default-route -o jsonpath='{.spec.host}')
INTERNAL_REF="image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}:${TAG}"
EXTERNAL_REF="${ROUTE}/${NAMESPACE}/${IMAGE_NAME}:${TAG}"
DOCKERFILE="${REPO_ROOT}/templates/Dockerfile.base"

echo "=== Building eval base image ==="
echo "  Tag:                 ${TAG}"
echo "  INSTALL_CLAUDE_CODE: ${INSTALL_CLAUDE}"
echo "  Harbor:              ${HARBOR_VERSION}"
echo "  AEH_SHA:             ${AEH_SHA}"
echo "  Dockerfile:          ${DOCKERFILE}"
echo "  Context:             ${REPO_ROOT}"
echo "  Push to:             ${EXTERNAL_REF}"
echo "  Pipeline ref:        ${INTERNAL_REF}"
echo ""

podman login --tls-verify=false \
  -u "$(oc whoami)" \
  -p "$(oc whoami -t)" \
  "$ROUTE"

podman build \
  --platform linux/amd64 \
  --build-arg "INSTALL_CLAUDE_CODE=${INSTALL_CLAUDE}" \
  --build-arg "AEH_SHA=${AEH_SHA}" \
  --build-arg "HARBOR_VERSION=${HARBOR_VERSION}" \
  -f "$DOCKERFILE" \
  -t "$EXTERNAL_REF" \
  "$REPO_ROOT"

podman push --tls-verify=false "$EXTERNAL_REF"

echo ""
echo "=== Done ==="
echo "Image pushed to registry. Pipeline will use:"
echo "  ${INTERNAL_REF}"
echo ""
echo "Rebuild both tags used by CI when deps change:"
echo "  ./scripts/build_base_image.sh --tag latest"
echo "  ./scripts/build_base_image.sh --tag local-env"
