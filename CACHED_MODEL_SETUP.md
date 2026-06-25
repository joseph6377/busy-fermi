# RunPod Cached Model Setup

RunPod currently supports one cached Hugging Face repository per Serverless endpoint. FLUX.2 Klein 9B needs three ComfyUI files that originate from separate repositories, so this project uses one private consolidation repository with this structure:

```text
diffusion_models/
└── flux-2-klein-9b-fp8.safetensors
text_encoders/
└── qwen_3_8b_fp8mixed.safetensors
vae/
└── flux2-vae.safetensors
```

The custom worker image contains no model weights. At startup, `docker/link-cached-models.sh` finds the RunPod cached Hugging Face snapshot and symlinks its files into `/comfyui/models/`.

## Endpoint configuration

- Container image: the custom image built from this repository
- Model: the private consolidated Hugging Face repository ID
- Hugging Face token: a RunPod secret with read access to that repository
- Environment variable:

```text
CACHED_MODEL_ID=<hugging-face-user>/<private-repository>
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
