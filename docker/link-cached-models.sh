#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${HF_HUB_CACHE:-/runpod-volume/huggingface-cache/hub}"
MODEL_ID="${CACHED_MODEL_ID:-${MODEL_NAME:-}}"

if [[ -z "${MODEL_ID}" ]]; then
  echo "CACHED_MODEL_ID or MODEL_NAME must identify the cached Hugging Face repository" >&2
  exit 1
fi

# Extract HF repository ID and optional revision from URL if a URL is provided
CACHED_REVISION=""
if [[ "${MODEL_ID}" =~ ^https?://huggingface.co/ ]]; then
  TEMP_ID="${MODEL_ID#*huggingface.co/}"
  if [[ "${TEMP_ID}" == *":"* ]]; then
    HF_REPO="${TEMP_ID%%:*}"
    HF_REV="${TEMP_ID#*:}"
  else
    HF_REPO="${TEMP_ID}"
    HF_REV=""
  fi
  MODEL_ID="${HF_REPO}"
  if [[ -n "${HF_REV}" ]]; then
    CACHED_REVISION="${HF_REV}"
  fi
fi

# Also check if raw MODEL_ID contains a colon for revision
if [[ "${MODEL_ID}" == *":"* ]]; then
  CACHED_REVISION="${MODEL_ID#*:}"
  MODEL_ID="${MODEL_ID%%:*}"
fi

REPO_DIR="${CACHE_ROOT}/models--${MODEL_ID//\//--}"
REF_FILE="${REPO_DIR}/refs/main"

if [[ -n "${CACHED_REVISION}" ]]; then
  SNAPSHOT="${REPO_DIR}/snapshots/${CACHED_REVISION}"
elif [[ -f "${REF_FILE}" ]]; then
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

# Check for LTX-2.3 files
LTX2_DETECTOR_1="${SNAPSHOT}/diffusion_models/LTX-2/ltx-2-19b-dev-fp8.safetensors"
LTX2_DETECTOR_2="${SNAPSHOT}/diffusion_models/ltx-2-19b-dev-fp8.safetensors"
LTX2_DETECTOR_3="${SNAPSHOT}/ltx-2-19b-dev-fp8.safetensors"

if [[ -f "${LTX2_DETECTOR_1}" || -f "${LTX2_DETECTOR_2}" || -f "${LTX2_DETECTOR_3}" ]]; then
  echo "Detected LTX-2.3 repository. Linking LTX-2.3 files..."
  
  if [[ -f "${LTX2_DETECTOR_1}" ]]; then
    DIFF_SRC="${LTX2_DETECTOR_1}"
    GEMMA_SRC="${SNAPSHOT}/text_encoders/GEMMA-3/gemma-3-12b-it-fp8_e4m3fn.safetensors"
    PROJ_SRC="${SNAPSHOT}/text_encoders/LTX-2/ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
    VVAE_SRC="${SNAPSHOT}/vae/LTX2_video_vae_bf16.safetensors"
    AVAE_SRC="${SNAPSHOT}/vae/LTX2_audio_vae_bf16.safetensors"
    UPSCALER_SRC="${SNAPSHOT}/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
  elif [[ -f "${LTX2_DETECTOR_2}" ]]; then
    DIFF_SRC="${LTX2_DETECTOR_2}"
    GEMMA_SRC="${SNAPSHOT}/text_encoders/gemma-3-12b-it-fp8_e4m3fn.safetensors"
    PROJ_SRC="${SNAPSHOT}/text_encoders/ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
    VVAE_SRC="${SNAPSHOT}/vae/LTX2_video_vae_bf16.safetensors"
    AVAE_SRC="${SNAPSHOT}/vae/LTX2_audio_vae_bf16.safetensors"
    UPSCALER_SRC="${SNAPSHOT}/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
  else
    DIFF_SRC="${LTX2_DETECTOR_3}"
    GEMMA_SRC="${SNAPSHOT}/gemma-3-12b-it-fp8_e4m3fn.safetensors"
    PROJ_SRC="${SNAPSHOT}/ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
    VVAE_SRC="${SNAPSHOT}/LTX2_video_vae_bf16.safetensors"
    AVAE_SRC="${SNAPSHOT}/LTX2_audio_vae_bf16.safetensors"
    UPSCALER_SRC="${SNAPSHOT}/ltx-2-spatial-upscaler-x2-1.0.safetensors"
  fi

  for f in "${DIFF_SRC}" "${GEMMA_SRC}" "${PROJ_SRC}" "${VVAE_SRC}" "${AVAE_SRC}"; do
    if [[ ! -f "$f" ]]; then
      echo "Error: Required LTX-2.3 file is missing: $f" >&2
      exit 1
    fi
  done

  DIFF_TARGET="/comfyui/models/diffusion_models/ltx-2-19b-dev-fp8.safetensors"
  mkdir -p "$(dirname "${DIFF_TARGET}")"
  ln -sfn "$(realpath "${DIFF_SRC}")" "${DIFF_TARGET}"
  echo "Linked Diffusion Model: ${DIFF_TARGET}"

  GEMMA_TARGET="/comfyui/models/text_encoders/gemma-3-12b-it-fp8_e4m3fn.safetensors"
  mkdir -p "$(dirname "${GEMMA_TARGET}")"
  ln -sfn "$(realpath "${GEMMA_SRC}")" "${GEMMA_TARGET}"
  echo "Linked Gemma 3: ${GEMMA_TARGET}"

  PROJ_TARGET="/comfyui/models/text_encoders/ltx-2-19b-embeddings_connector_dev_bf16.safetensors"
  mkdir -p "$(dirname "${PROJ_TARGET}")"
  ln -sfn "$(realpath "${PROJ_SRC}")" "${PROJ_TARGET}"
  echo "Linked Text Projection: ${PROJ_TARGET}"

  VVAE_TARGET="/comfyui/models/vae/LTX2_video_vae_bf16.safetensors"
  mkdir -p "$(dirname "${VVAE_TARGET}")"
  ln -sfn "$(realpath "${VVAE_SRC}")" "${VVAE_TARGET}"
  echo "Linked Video VAE: ${VVAE_TARGET}"

  AVAE_TARGET="/comfyui/models/vae/LTX2_audio_vae_bf16.safetensors"
  mkdir -p "$(dirname "${AVAE_TARGET}")"
  ln -sfn "$(realpath "${AVAE_SRC}")" "${AVAE_TARGET}"
  echo "Linked Audio VAE: ${AVAE_TARGET}"

  if [[ -f "${UPSCALER_SRC}" ]]; then
    UPSCALER_TARGET="/comfyui/models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    mkdir -p "$(dirname "${UPSCALER_TARGET}")"
    ln -sfn "$(realpath "${UPSCALER_SRC}")" "${UPSCALER_TARGET}"
    echo "Linked Spatial Upscaler: ${UPSCALER_TARGET}"
  fi

elif [[ -f "${FLUX2_DETECTOR_1}" || -f "${FLUX2_DETECTOR_2}" ]]; then
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
