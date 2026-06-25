#!/usr/bin/env bash
set -euo pipefail

# 1. First, check if there is a models/ folder in the network volume and link those
if [[ -d "/runpod-volume/models" ]]; then
  echo "Found models folder in network volume. Linking dynamically..."
  find /runpod-volume/models -type f | while read -r source_path; do
    relative_path="${source_path#/runpod-volume/models/}"
    target_path="/comfyui/models/${relative_path}"
    mkdir -p "$(dirname "${target_path}")"
    ln -sfn "${source_path}" "${target_path}"
    echo "Linked volume model: ${relative_path}"
  done
fi

# 2. For the default FLUX files, link from HF cache if they are not already provided by the volume
CACHE_ROOT="${HF_HUB_CACHE:-/runpod-volume/huggingface-cache/hub}"
MODEL_ID="${CACHED_MODEL_ID:-${MODEL_NAME:-}}"

if [[ -n "${MODEL_ID}" ]]; then
  REPO_DIR="${CACHE_ROOT}/models--${MODEL_ID//\//--}"
  REF_FILE="${REPO_DIR}/refs/main"

  if [[ -f "${REF_FILE}" ]]; then
    REVISION="$(tr -d '[:space:]' < "${REF_FILE}")"
    SNAPSHOT="${REPO_DIR}/snapshots/${REVISION}"
  else
    SNAPSHOT="$(find "${REPO_DIR}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1 || true)"
  fi

  if [[ -n "${SNAPSHOT:-}" && -d "${SNAPSHOT}" ]]; then
    declare -a REQUIRED_FILES=(
      "diffusion_models/flux-2-klein-9b-fp8.safetensors"
      "text_encoders/qwen_3_8b_fp8mixed.safetensors"
      "vae/flux2-vae.safetensors"
    )

    for relative_path in "${REQUIRED_FILES[@]}"; do
      target_path="/comfyui/models/${relative_path}"
      # Only link from cache if not already placed by the network volume
      if [[ ! -f "${target_path}" ]]; then
        source_path="${SNAPSHOT}/${relative_path}"
        if [[ -f "${source_path}" ]]; then
          mkdir -p "$(dirname "${target_path}")"
          ln -sfn "${source_path}" "${target_path}"
          echo "Linked cached fallback: ${relative_path}"
        else
          echo "Fallback file missing: ${relative_path}"
        fi
      fi
    done
  fi
fi

exec /start.sh
