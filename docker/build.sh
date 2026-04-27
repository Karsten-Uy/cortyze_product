#!/usr/bin/env bash
# Build + push the GPU worker image to a registry RunPod can pull from.
#
# Usage:
#   GITHUB_USER=yourname ./docker/build.sh         # build + push to GHCR
#   GITHUB_USER=yourname TAG=v1 ./docker/build.sh  # tagged build
#
# Prerequisites:
#   - Docker / OrbStack / Colima running locally (Apple Silicon: --platform
#     linux/amd64 cross-compiles to RunPod's x86 GPUs).
#   - Logged in to the registry: `echo $GITHUB_PAT | docker login ghcr.io -u $GITHUB_USER --password-stdin`

set -euo pipefail

if [[ -z "${GITHUB_USER:-}" ]]; then
  echo "ERROR: set GITHUB_USER (your GitHub username for ghcr.io)" >&2
  exit 1
fi

# Docker registries require all-lowercase repository names; GitHub usernames
# are case-insensitive at the auth layer, so we lowercase here transparently.
GITHUB_USER_LC="$(echo "${GITHUB_USER}" | tr '[:upper:]' '[:lower:]')"

REGISTRY="${REGISTRY:-ghcr.io/${GITHUB_USER_LC}}"
IMAGE="${IMAGE:-cortyze-gpu-worker}"
TAG="${TAG:-latest}"
FULL="${REGISTRY}/${IMAGE}:${TAG}"

echo "Building ${FULL}..."
docker buildx build \
  --platform linux/amd64 \
  -t "${FULL}" \
  -f docker/runpod.Dockerfile \
  --push \
  .

echo
echo "Pushed: ${FULL}"
echo
echo "Use this image URL when configuring the RunPod template:"
echo "  ${FULL}"
