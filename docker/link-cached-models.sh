#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${HF_HUB_CACHE:-/runpod-volume/huggingface-cache/hub}"
MODEL_ID="${CACHED_MODEL_ID:-${MODEL_NAME:-}}"

if [[ -z "${MODEL_ID}" ]]; then
  echo "CACHED_MODEL_ID or MODEL_NAME must identify the cached Hugging Face repository" >&2
  exit 1
fi

REPO_DIR="${CACHE_ROOT}/models--${MODEL_ID//\//--}"
REF_FILE="${REPO_DIR}/refs/main"

if [[ -f "${REF_FILE}" ]]; then
  REVISION="$(tr -d '[:space:]' < "${REF_FILE}")"
  SNAPSHOT="${REPO_DIR}/snapshots/${REVISION}"
else
  SNAPSHOT="$(find "${REPO_DIR}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${SNAPSHOT:-}" || ! -d "${SNAPSHOT}" ]]; then
  echo "Cached model snapshot not found for ${MODEL_ID} under ${REPO_DIR}" >&2
  exit 1
fi

if [[ -f "${SNAPSHOT}/flux1-dev-fp8.safetensors" ]]; then
  SOURCE_PATH="${SNAPSHOT}/flux1-dev-fp8.safetensors"
elif [[ -f "${SNAPSHOT}/checkpoints/flux1-dev-fp8.safetensors" ]]; then
  SOURCE_PATH="${SNAPSHOT}/checkpoints/flux1-dev-fp8.safetensors"
else
  echo "Required cached model file flux1-dev-fp8.safetensors is missing in snapshot ${SNAPSHOT}" >&2
  exit 1
fi

TARGET_PATH="/comfyui/models/checkpoints/flux1-dev-fp8.safetensors"
mkdir -p "$(dirname "${TARGET_PATH}")"
ln -sfn "${SOURCE_PATH}" "${TARGET_PATH}"
echo "Linked flux1-dev-fp8.safetensors"

exec /start.sh
