# Project Structure Map

## High-Level Anatomy

| Directory/File | Description |
|----------------|-------------|
| `__init__.py` | **ENTRY POINT**. Registers nodes with ComfyUI and handles theme/settings injection. |
| `modules/` | **CORE LOGIC**. Refactored modular node implementations. |
| `web/` | Javascript files for UI extensions (styling, colors, and Nodes 2.0 enforcements). |
| `docs/` | Internal architectural documentation and code maps. |
| `AGENTS.md` | Developer guide for AI Agents. |

## Architecture: Hub-and-Spoke Pipeline

The toolkit uses a **hub-and-spoke** architecture centered on the `BlockSampler`:

```
Loader (UME_BUNDLE) ──┐
Settings (UME_SETTINGS)──┤
Prompts ──────────────────┤──▶ BlockSampler ──▶ UME_PIPELINE (GenerationContext)
LoRAs ────────────────────┤                        │
Source Image (UME_IMAGE) ─┘                        ├──▶ Post-Process (Upscale/Detail/Daemon)
                                                   └──▶ ImageSaver
```

- **`UME_BUNDLE`**: `UmeBundle` dataclass `{model, model_low_noise, clip, vae, model_name, bundle_type, loader_type, shift, overrides, clip_vision, audio_vae, latent_upscale_model}` — produced by Loaders.
- **`UME_SETTINGS`**: `UmeSettings` dataclass `{width, height, steps, cfg, sampler_name, scheduler, seed}` — produced by GenerationSettings.
- **`UME_PIPELINE`**: `GenerationContext` object — created by BlockSampler or PackPipeline, carries image + all context through post-processing chain.
- **`UME_VIDEO_PIPELINE`**: `VideoGenerationContext` object — created by Video Generator/Extender/Enhancer/Keyframe/PromptDirector nodes.

## Interoperability (Pack/Unpack Nodes)

Pack and Unpack nodes enable bidirectional compatibility with native/community ComfyUI nodes:

| Node | Direction | Description |
|------|-----------|-------------|
| **Pack Models Bundle** | Native → UME | Packs MODEL + CLIP + VAE into `UME_BUNDLE` |
| **Unpack Models Bundle** | UME → Native | Extracts MODEL, CLIP, VAE, model_name from `UME_BUNDLE` |
| **Unpack Pipeline** | UME → Native | Extracts IMAGE + all 15 fields from `UME_PIPELINE` |
| **Unpack Settings Bundle** | UME → Native | Extracts all settings from `UME_SETTINGS` |
| **Unpack Image Bundle** | UME → Native | Extracts IMAGE, MASK, mode, denoise, auto_resize from `UME_IMAGE` |

## Sub-Modules (`modules/`)

- `common.py`: `GenerationContext` class, `PipelineParams`, `extract_pipeline_params()`, `validate_bundle()`, shared constants (`KNOWN_DIT_MODELS`), and core utilities.
- `logger.py`: Standardized colorized logging utility.
- `optimization_utils.py`: Environment checks (SageAttention, Triton, etc.).
- `logic_nodes.py`: Re-export shim — imports from `upscale_nodes`, `seedvr2_nodes`, `face_nodes`, `detail_daemon_nodes`, `detail_refiner`.
- `block_nodes.py`: Re-export shim for backward compatibility — imports from sub-modules.
- `block_inputs.py`: LoRA blocks, ControlNet, GenerationSettings (→ `UME_SETTINGS`), VideoSettings, LTXVideoSettings, Image Loader/Process, Prompt Inputs.
- `block_loaders.py`: Model Loaders (→ `UME_BUNDLE`), BundleAutoLoader.
- `block_sampler.py`: BlockSampler hub (→ `UME_PIPELINE`).
- `block_passthrough.py`: PackPipeline — alternative to BlockSampler for upscale-only workflows.
- `image_nodes.py`: Pipeline Image Saver.
- `utils_nodes.py`: Pack/Unpack interoperability nodes, Bundle Downloader, Signature.
- `manifest.py`: Model manifest management (HuggingFace cache + download).
- `download_utils.py`: Download engine (aria2c + urllib fallback).

## Core Directories (Vendored/Integrated)

- `facedetailer_core/`: Logic for face detection and enhancement.
- `seedvr2_core/`: Ported tiling upscaler for high-VRAM efficiency.
- `usdu_core/`: Integrated Ultimate SD Upscale logic.
- `image_saver_core/`: Robust image saving with metadata.
- `vendor/comfyui_gguf/`: GGUF model loading support.
- `vendor/ltxvideo/`: Vendored ComfyUI-LTXVideo samplers and latent helpers.
- `vendor/aria2/`: Bundled aria2c binary for accelerated model downloads.

## Registration Workflow

1. Node classes are defined in `modules/`.
2. `__init__.py` imports necessary classes.
3. `NODE_CLASS_MAPPINGS` links ComfyUI internal keys to Python classes.
4. `NODE_DISPLAY_NAME_MAPPINGS` provides user-friendly titles.
5. `WEB_DIRECTORY` exposes the `web/` folder for frontend styling.
