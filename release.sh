#!/usr/bin/env bash
set -euo pipefail

IMAGE="gwesterrunpod/rp-gwestr-vllm"
VERSION="$(cat VERSION)"
TAG="${IMAGE}:${VERSION}"

echo "Building ${TAG}..."
docker build --network=host -t "${TAG}" .

echo "Pushing ${TAG}..."
docker push "${TAG}"

echo "Done: ${TAG}"
echo "Update your RunPod endpoint template to use: ${TAG}"
