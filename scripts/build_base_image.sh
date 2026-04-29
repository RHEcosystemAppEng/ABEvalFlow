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
#
# When to rebuild:
#   - claude-code version update (npm will pull latest)
#   - uv version bump (edit templates/Dockerfile.base)
#   - base OS image update (ubi9/python-311)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NAMESPACE="${NAMESPACE:-ab-eval-flow}"
IMAGE_NAME="${IMAGE_NAME:-eval-base}"
TAG="latest"
INSTALL_CLAUDE="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-claude)   INSTALL_CLAUDE="false"; shift ;;
    --tag)         TAG="$2"; shift 2 ;;
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
echo "  Dockerfile:          ${DOCKERFILE}"
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
  -f "$DOCKERFILE" \
  -t "$EXTERNAL_REF" \
  "$REPO_ROOT/templates"

podman push --tls-verify=false "$EXTERNAL_REF"

echo ""
echo "=== Done ==="
echo "Image pushed to registry. Pipeline will use:"
echo "  ${INTERNAL_REF}"
