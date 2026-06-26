# RunPod Cached Model Setup

RunPod currently supports one cached Hugging Face repository per Serverless endpoint. LTX-2.3 Video+Audio needs multiple ComfyUI files that originate from separate repositories, so this project uses one private consolidation repository with this structure:

```text
diffusion_models/
└── LTX-2/
    └── ltx-2-19b-dev-fp8.safetensors
text_encoders/
├── GEMMA-3/
│   └── gemma-3-12b-it-fp8_e4m3fn.safetensors
└── LTX-2/
    └── ltx-2-19b-embeddings_connector_dev_bf16.safetensors
vae/
├── LTX2_video_vae_bf16.safetensors
└── LTX2_audio_vae_bf16.safetensors
latent_upscale_models/
└── ltx-2-spatial-upscaler-x2-1.0.safetensors  (optional)
```

The custom worker image contains no model weights. At startup, `docker/link-cached-models.sh` finds the RunPod cached Hugging Face snapshot and symlinks its files into `/comfyui/models/`.

## Endpoint configuration

- Container image: `ghcr.io/joseph6377/busy-fermi:latest`
- Model: `joseph6377/ltx-2.3-consolidated`
- Hugging Face token: a RunPod secret with read access to that repository
- Environment variable:

```text
CACHED_MODEL_ID=joseph6377/ltx-2.3-consolidated
```

- Active workers: `0`
- Max workers: `1`
- GPU count: `1`
- Idle timeout: `5`
- Execution timeout: `600`
- FlashBoot: enabled
- Network volume: none

RunPod downloads cached models before starting billable worker compute. The endpoint may wait for a compatible cached host on its first request, but the download itself is not billed as GPU worker time.

## Updating models

Upload a new revision to the same private Hugging Face repository, then redeploy or restart the endpoint so RunPod resolves the new cached revision. The Docker image only needs rebuilding when startup mapping or custom nodes change.
