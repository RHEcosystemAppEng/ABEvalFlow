#!/usr/bin/env bash
set -euo pipefail
#
# Build and push the eval base image to the OpenShift internal registry.
# Run this from any machine logged in to the cluster (oc whoami).
#
# Usage:
#   ./scripts/build_base_image.sh                         # with claude-code
#   ./scripts/build_base_image.sh --no-claude              # without claude-code
#   ./scripts/build_base_image.sh --tag v2                 # custom tag
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REGISTRY="${REGISTRY:-image-registry.openshift-image-registry.svc:5000}"
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

FULL_TAG="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${TAG}"
DOCKERFILE="${REPO_ROOT}/templates/Dockerfile.base"

echo "=== Building base image ==="
echo "  Image:              ${FULL_TAG}"
echo "  INSTALL_CLAUDE_CODE: ${INSTALL_CLAUDE}"
echo "  Dockerfile:         ${DOCKERFILE}"
echo ""

oc -n "${NAMESPACE}" new-build --name base-image-builder \
  --binary --strategy=docker --to="${NAMESPACE}/${IMAGE_NAME}:${TAG}" \
  --build-arg="INSTALL_CLAUDE_CODE=${INSTALL_CLAUDE}" \
  --dry-run -o yaml 2>/dev/null | head -1 > /dev/null || true

oc -n "${NAMESPACE}" start-build base-image-builder \
  --from-dir="${REPO_ROOT}/templates" \
  --build-arg="INSTALL_CLAUDE_CODE=${INSTALL_CLAUDE}" \
  --follow 2>/dev/null && exit 0 || true

echo "BuildConfig not found — falling back to pod-based buildah build..."

POD_NAME="base-image-build-$(date +%s)"

cat <<YAML | oc -n "${NAMESPACE}" create -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  labels:
    app: base-image-builder
spec:
  restartPolicy: Never
  serviceAccountName: pipeline
  containers:
    - name: buildah
      image: registry.access.redhat.com/ubi9/buildah:9.6
      securityContext:
        capabilities:
          add: ["SETFCAP"]
      command:
        - /bin/bash
        - -c
        - |
          set -euo pipefail
          buildah login \
            --tls-verify=false \
            -u serviceaccount \
            -p "\$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)" \
            "${REGISTRY}"

          cd /workspace
          buildah build \
            --storage-driver=vfs \
            --build-arg "INSTALL_CLAUDE_CODE=${INSTALL_CLAUDE}" \
            -f Dockerfile.base \
            -t "${FULL_TAG}" \
            .

          buildah push \
            --storage-driver=vfs \
            --tls-verify=false \
            "${FULL_TAG}" \
            "docker://${FULL_TAG}"

          echo "=== Done: pushed ${FULL_TAG} ==="
      volumeMounts:
        - name: dockerfile
          mountPath: /workspace
  volumes:
    - name: dockerfile
      configMap:
        name: base-image-dockerfile
YAML

echo ""
echo "Waiting for Dockerfile ConfigMap — creating it now..."
oc -n "${NAMESPACE}" create configmap base-image-dockerfile \
  --from-file=Dockerfile.base="${DOCKERFILE}" \
  --dry-run=client -o yaml | oc -n "${NAMESPACE}" apply -f -

echo "Waiting for pod ${POD_NAME} to complete..."
oc -n "${NAMESPACE}" wait pod "${POD_NAME}" --for=condition=Ready --timeout=30s 2>/dev/null || true
oc -n "${NAMESPACE}" logs -f "${POD_NAME}"
oc -n "${NAMESPACE}" wait pod "${POD_NAME}" --for=jsonpath='{.status.phase}'=Succeeded --timeout=600s

echo ""
echo "=== Base image pushed: ${FULL_TAG} ==="
echo "You can now reference it in the pipeline as:"
echo "  base-image: ${FULL_TAG}"

oc -n "${NAMESPACE}" delete pod "${POD_NAME}" --ignore-not-found
