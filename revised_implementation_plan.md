# ComfyUI Local UI + RunPod Cached Model Execution

## Budget-First Implementation Plan

The Mac runs ComfyUI for workflow design and a local dashboard for submissions. RunPod Serverless performs GPU inference. Model weights are delivered through **RunPod Cached Models**, so the GPU is not billed while large Hugging Face files download.

For FLUX.2 Klein 9B, RunPod's current one-repository cache limit requires one private Hugging Face repository containing the three ComfyUI-ready files.

---

## 1. Architecture

```text
Civitai / Hugging Face
        │
        ▼
Private consolidated Hugging Face repository
        │
        │ RunPod Cached Models (download is not billed as worker time)
        ▼
RunPod cached snapshot
        │
        │ custom startup symlinks
        ▼
/comfyui/models/{diffusion_models,text_encoders,vae,...}
        │
        ▼
RunPod ComfyUI Serverless worker
        │
        ▼
Local Node.js proxy → outputs/ → browser gallery
```

The browser never receives API keys or Hugging Face tokens.

---

## 2. FLUX.2 Klein 9B Model Repository

Create one private Hugging Face model repository with:

```text
diffusion_models/
└── flux-2-klein-9b-fp8.safetensors
text_encoders/
└── qwen_3_8b_fp8mixed.safetensors
vae/
└── flux2-vae.safetensors
```

RunPod currently supports one cached repository per endpoint. Keeping these files together makes the full ComfyUI model set available before billable worker startup.

The model repository must remain private because redistribution rights depend on the upstream licenses. The repository README must identify each source and license.

---

## 3. Custom Worker Image

Build a small image from:

```dockerfile
FROM runpod/worker-comfyui:5.8.6-base
```

The image contains:

- no model weights;
- no access tokens;
- the cached-model linking script;
- custom nodes only when a workflow requires them.

At startup, the script:

1. Reads `CACHED_MODEL_ID` or `MODEL_NAME`.
2. Resolves the Hugging Face cache snapshot under:

   ```text
   /runpod-volume/huggingface-cache/hub/
   ```

3. Verifies all required files exist.
4. Symlinks them into `/comfyui/models/`.
5. Executes the official worker `/start.sh`.

Missing files must fail startup clearly rather than trigger an internet download on paid GPU time.

---

## 4. RunPod Endpoint Configuration

| Setting | Value |
|---|---|
| Endpoint type | Queue |
| GPU tier | 24 GB standard |
| Active workers | `0` |
| Max workers | `1` |
| GPUs per worker | `1` |
| Idle timeout | `5` seconds |
| Execution timeout | `600` seconds |
| FlashBoot | Enabled, Standard |
| Worker image | Custom image built from this repository |
| Cached Model | Private consolidated Hugging Face repository |
| Network volume | None |

Add the Hugging Face token as a RunPod encrypted secret with read access to the private model repository.

Add:

```text
CACHED_MODEL_ID=<hugging-face-user>/<repository>
```

Keep endpoint active workers at zero. Creating or updating endpoint configuration does not submit a generation job.

---

## 5. Local Application

```text
.
├── Dockerfile
├── docker/
│   └── link-cached-models.sh
├── proxy.js
├── lib/
├── public/
├── samples/
│   └── flux2-klein-9b-text-to-image-api.json
├── outputs/
├── local_models/
├── model_manifest.json
├── CACHED_MODEL_SETUP.md
└── .env
```

The local proxy provides:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/config` | Safe readiness information |
| `GET` | `/api/models` | Local model manifest |
| `POST` | `/api/workflows/inspect` | Validate API-format workflow |
| `POST` | `/api/jobs` | Submit asynchronous RunPod job |
| `GET` | `/api/jobs/:id` | Poll and normalize job status |
| `POST` | `/api/jobs/:id/cancel` | Cancel job |
| `GET` | `/outputs/...` | Serve locally saved outputs |

The real RunPod API key remains in `.env` on the Mac.

---

## 6. Workflow

The first supported workflow is:

```text
samples/flux2-klein-9b-text-to-image-api.json
```

It uses:

- FLUX.2 Klein 9B FP8 distilled model;
- Qwen 3 8B FP8 mixed encoder;
- FLUX.2 VAE;
- Euler sampler;
- four steps;
- CFG `1`;
- 1024×1024 output;
- batch size `1`.

The dashboard exposes prompt, seed, dimensions, steps, CFG, sampler and recognized model filenames.

---

## 7. Cost Controls

- Cached model download occurs before billable worker time.
- Active workers remain `0`.
- Maximum workers remains `1`.
- One local submission is allowed at a time.
- Batch size defaults to `1`.
- No automatic generation retries.
- No persistent network-volume storage bill.
- No model download fallback inside the running GPU worker.
- Job execution timeout is `600` seconds.
- Output polling uses bounded intervals.
- Automated tests never submit paid jobs.

FlashBoot may preserve paused workers for faster starts without active-worker billing, subject to RunPod availability and current product behavior.

---

## 8. Implementation Phases

### Phase 1 — Local control plane

- Local Express proxy and dashboard.
- Workflow validation and parameter inspection.
- Mock cloud sessions and mock jobs.
- Safe configuration handling.

**Status:** Complete.

### Phase 2 — Model preparation

- Download three FLUX.2 Klein 9B files.
- Verify and register SHA-256 hashes.
- Create matching API-format workflow.

**Status:** Complete.

### Phase 3 — Cached model package

- Create private Hugging Face consolidation repository.
- Upload the three verified files.
- Add source/license README.
- Verify repository file sizes.

**Completion gate:** All three files are accessible through one private repository using the configured token.

### Phase 4 — Custom worker

- Build `Dockerfile`.
- Test cached snapshot resolution.
- Test symlink creation.
- Push image to a container registry.

**Completion gate:** Worker startup finds the cached repository and ComfyUI recognizes all three model filenames.

### Phase 5 — Endpoint update

- Change endpoint image to the custom worker.
- Set the cached Model repository.
- Add encrypted Hugging Face token.
- Set `CACHED_MODEL_ID`.
- Reconfirm zero active workers and one max worker.

**Completion gate:** Endpoint is ready with zero running workers and no configuration error.

### Phase 6 — Paid end-to-end test

1. Submit one 1024×1024, four-step job.
2. Observe queue, initialization and execution.
3. Save all returned images locally.
4. Record delay and execution time.
5. Confirm the worker scales down.
6. Confirm there is no network volume.

Only this phase intentionally incurs GPU compute cost.

---

## 9. Fallbacks

Use a temporary network volume only when:

- a model is not hosted on Hugging Face;
- licensing prevents creating a private consolidated repository;
- the endpoint needs frequently changing arbitrary Civitai files;
- cached-model limitations prevent the required layout.

The existing multipart S3 and guarded volume-management modules remain available as a fallback, not the primary architecture.

---

## 10. Verification

Automated tests cover:

- API workflow validation;
- known-field detection;
- generation limits;
- output decoding and safe filenames;
- preservation of unknown workflow inputs.

Worker-image verification must additionally cover:

- missing `CACHED_MODEL_ID`;
- missing cache repository;
- missing revision reference;
- missing required model file;
- correct symlink destinations;
- successful handoff to `/start.sh`.

---

## 11. Security

- Never commit `.env`.
- Never bake Hugging Face or RunPod tokens into Docker layers.
- Use a read-only Hugging Face token.
- Store the Hugging Face token as a RunPod encrypted secret.
- Keep the consolidated model repository private.
- Do not log authorization headers or model tokens.
- Bind the local proxy to `127.0.0.1`.

---

## 12. References

- [RunPod Cached Models](https://docs.runpod.io/serverless/endpoints/model-caching)
- [RunPod ComfyUI worker customization](https://github.com/runpod-workers/worker-comfyui/blob/main/docs/customization.md)
- [Official ComfyUI FLUX.2 Klein guide](https://docs.comfy.org/tutorials/flux/flux-2-klein)
- [Official RunPod ComfyUI worker](https://github.com/runpod-workers/worker-comfyui)

Review product limits, pricing and worker releases before deployment because they can change.
