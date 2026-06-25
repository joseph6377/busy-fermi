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

echo "=== Debug Cache Mapping ==="
echo "CACHE_ROOT: ${CACHE_ROOT}"
echo "MODEL_ID: ${MODEL_ID}"
echo "REPO_DIR: ${REPO_DIR}"
echo "SNAPSHOT: ${SNAPSHOT}"

# Check for FLUX.2 Klein files
FLUX2_DETECTOR_1="${SNAPSHOT}/diffusion_models/flux-2-klein-9b-fp8.safetensors"
FLUX2_DETECTOR_2="${SNAPSHOT}/flux-2-klein-9b-fp8.safetensors"

if [[ -f "${FLUX2_DETECTOR_1}" || -f "${FLUX2_DETECTOR_2}" ]]; then
  echo "Detected FLUX.2 Klein 9B repository. Linking FLUX.2 Klein files..."
  
  if [[ -f "${FLUX2_DETECTOR_1}" ]]; then
    UNET_SRC="${FLUX2_DETECTOR_1}"
    CLIP_SRC="${SNAPSHOT}/text_encoders/qwen_3_8b_fp8mixed.safetensors"
    VAE_SRC="${SNAPSHOT}/vae/flux2-vae.safetensors"
  else
    UNET_SRC="${FLUX2_DETECTOR_2}"
    CLIP_SRC="${SNAPSHOT}/qwen_3_8b_fp8mixed.safetensors"
    VAE_SRC="${SNAPSHOT}/flux2-vae.safetensors"
  fi

  for f in "${UNET_SRC}" "${CLIP_SRC}" "${VAE_SRC}"; do
    if [[ ! -f "$f" ]]; then
      echo "Error: Required FLUX.2 Klein file is missing: $f" >&2
      exit 1
    fi
  done

  UNET_TARGET="/comfyui/models/diffusion_models/flux-2-klein-9b-fp8.safetensors"
  mkdir -p "$(dirname "${UNET_TARGET}")"
  ln -sfn "$(realpath "${UNET_SRC}")" "${UNET_TARGET}"
  echo "Linked UNET: ${UNET_TARGET}"

  CLIP_TARGET="/comfyui/models/text_encoders/qwen_3_8b_fp8mixed.safetensors"
  mkdir -p "$(dirname "${CLIP_TARGET}")"
  ln -sfn "$(realpath "${CLIP_SRC}")" "${CLIP_TARGET}"
  echo "Linked CLIP: ${CLIP_TARGET}"

  VAE_TARGET="/comfyui/models/vae/flux2-vae.safetensors"
  mkdir -p "$(dirname "${VAE_TARGET}")"
  ln -sfn "$(realpath "${VAE_SRC}")" "${VAE_TARGET}"
  echo "Linked VAE: ${VAE_TARGET}"

else
  # Default check/fallback for FLUX.1 Dev
  FLUX1_DETECTOR_1="${SNAPSHOT}/flux1-dev-fp8.safetensors"
  FLUX1_DETECTOR_2="${SNAPSHOT}/checkpoints/flux1-dev-fp8.safetensors"

  if [[ -f "${FLUX1_DETECTOR_1}" ]]; then
    SOURCE_PATH="${FLUX1_DETECTOR_1}"
  elif [[ -f "${FLUX1_DETECTOR_2}" ]]; then
    SOURCE_PATH="${FLUX1_DETECTOR_2}"
  else
    echo "Could not auto-detect model type (FLUX.1 Dev or FLUX.2 Klein) in snapshot ${SNAPSHOT}." >&2
    echo "Available files in snapshot:"
    find "${SNAPSHOT}" -type f -o -type l | xargs ls -l || true
    exit 1
  fi

  REAL_SOURCE_PATH="$(realpath "${SOURCE_PATH}")"
  TARGET_PATH="/comfyui/models/checkpoints/flux1-dev-fp8.safetensors"
  mkdir -p "$(dirname "${TARGET_PATH}")"
  ln -sfn "${REAL_SOURCE_PATH}" "${TARGET_PATH}"
  echo "Linked flux1-dev-fp8.safetensors to ${TARGET_PATH}"
fi

exec /start.sh
