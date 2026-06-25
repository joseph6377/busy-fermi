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

declare -a REQUIRED_FILES=(
  "diffusion_models/flux-2-klein-9b-fp8.safetensors"
  "text_encoders/qwen_3_8b_fp8mixed.safetensors"
  "vae/flux2-vae.safetensors"
)

for relative_path in "${REQUIRED_FILES[@]}"; do
  source_path="${SNAPSHOT}/${relative_path}"
  target_path="/comfyui/models/${relative_path}"
  if [[ ! -f "${source_path}" ]]; then
    echo "Required cached model file is missing: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_path}")"
  ln -sfn "${source_path}" "${target_path}"
  echo "Linked ${relative_path}"
done

exec /start.sh
