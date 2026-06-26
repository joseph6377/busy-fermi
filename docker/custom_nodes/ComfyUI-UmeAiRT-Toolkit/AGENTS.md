# UmeAiRT Toolkit - Agent Development Guide

> Instructions for AI coding agents working on this project.
> For architecture details, see `/docs/codemaps/`.
>
> This file follows the [AGENTS.md](https://agents.md) standard.

## Project Overview

ComfyUI Custom Nodes toolkit (57 nodes) organized into 11 menu categories:
1. **Loaders** (`UmeAiRT/Loaders`): Model loading nodes (Checkpoint, FLUX, Z-IMG, QWEN, HiDream, WAN, LTX, Bundle) + Lightning Accelerator.
2. **LoRA** (`UmeAiRT/Loaders/LoRA`): LoRA stack blocks (1x, 3x, 5x, 10x).
3. **Inputs** (`UmeAiRT/Inputs`): Generation Settings.
4. **Image** (`UmeAiRT/Image`): Image loading, processing (Img2Img/Inpaint/Outpaint/Kontext), and ControlNet.
5. **Sampler** (`UmeAiRT/Sampler`): Central hub (`BlockSampler`) using hub-and-spoke architecture.
6. **Post-Process** (`UmeAiRT/Post-Process`): Upscalers (USDU, SeedVR2), Detailers (Subject, Daemon, Refiner), Video Interpolation/Upscale.
7. **Output** (`UmeAiRT/Output`): Image saving.
8. **Video** (`UmeAiRT/Video`): WAN/LTX video generators, extender, enhancer, keyframe, prompt director, audio replacer, slicer, optimization, lightning, output.
9. **Interop** (`UmeAiRT/Interop`): Pack/Unpack nodes for native ComfyUI compatibility.
10. **Utils** (`UmeAiRT/Utils`): Signature, Bundle Downloader.
11. **Tools** (`UmeAiRT/Tools`): Image to Prompt (VLM-powered).

## Ecosystem (Sibling Projects on `Y:\`)

This project is part of a 6-project ecosystem. **Direct** relationships:

| Project | Relationship |
|---------|-------------|
| `ComfyUI-Workflows` | Workflows will be migrated to use this Toolkit's block pipeline nodes (Toolkit still in development — not yet integrated) |
| `ComfyUI-Auto_installer` | The installer auto-installs this Toolkit as a custom node via `custom_nodes.json` |
| `ComfyUI-UmeAiRT-Sync` | The Sync node distributes workflows that depend on this Toolkit |
| `UmeAiRT-NAS-Utils` | Orchestration hub — may run consistency checks against this project |

> ⚠️ **Impact awareness**: Renaming or removing a node class will break existing workflows in `ComfyUI-Workflows`. Always check workflow compatibility before modifying `NODE_CLASS_MAPPINGS`.

## Critical Conventions

### Block Architecture (Hub-and-Spoke)

The **BlockSampler** is the central hub. Side-input nodes feed into it, and the generated image flows through post-processing via the `UME_PIPELINE`.

```
Loader ──▶ UME_BUNDLE {model, clip, vae, model_name}
                │
Settings ──▶ UME_SETTINGS {width, height, steps, cfg, ...}
                │
Prompts ────────┤
LoRAs ──────────┤──▶ BlockSampler ──▶ UME_PIPELINE (GenerationContext)
Source Image ───┘                          │
                                           ├──▶ Post-Process nodes (read/write gen_pipe.image)
                                           └──▶ ImageSaver (reads gen_pipe.image)
```

**Key types:**

| Type | Content (dataclass) | Produced by |
|------|---------|-------------|
| `UME_BUNDLE` | `UmeBundle` — model, model_low_noise, clip, vae, model_name, bundle_type, loader_type, shift, overrides, clip_vision, audio_vae, latent_upscale_model | All Loader nodes |
| `UME_SETTINGS` | `UmeSettings` — width, height, steps, cfg, sampler_name, scheduler, seed | GenerationSettings |
| `UME_VIDEO_SETTINGS` | `UmeVideoSettings` — width, height, duration, steps, cfg, shift, sampler_name, scheduler, seed, frame_rate, audio_enabled, sigmas_preset, custom_sigmas | VideoSettings / LTXVideoSettings |
| `UME_PIPELINE` | `GenerationContext` object (image + all context) | BlockSampler, PackPipeline |
| `UME_VIDEO_PIPELINE` | `VideoGenerationContext` object (frames + audio + context) | Video Generator/Extender/Enhancer/Keyframe/PromptDirector nodes |
| `UME_IMAGE` | `UmeImage` — image, mask, reference_image, mode, denoise, auto_resize, controlnets, outpaint_target_w/h, outpaint_h_align/v_align, outpaint_mask_blur | BlockImageLoader → ImageProcess nodes |
| `UME_LORA_STACK` | list of (name, model_strength, clip_strength[, target]) — target is `"both"`, `"high"`, or `"low"` (optional, defaults to `"both"`) | LoRA Block / WAN LoRA Block nodes |
| `UME_PROMPT_SCHEDULE` | list of {start_time, prompt} segments | Prompt Segment (chainable) |
| `UME_VACE_FRAMES` | `UmeVaceFrames` — start_image, end_image | Video VACE Prep |

**Rules:**
- Post-process nodes receive `UME_PIPELINE`, read `gen_pipe.image`, process, update `gen_pipe.image`, return `UME_PIPELINE`.
- Only `BlockSampler` and `PackPipeline` create `GenerationContext`. Only Video Generator/Extender/Enhancer/Keyframe/PromptDirector nodes create `VideoGenerationContext`.
- All pipeline/generation parameters are named `gen_pipe` (not `pipeline` or `generation`).

### Manifest & Auto-Download Architecture
- The toolkit relies on `modules/manifest.py` which manages a cached copy of the remote HuggingFace `model_manifest.json` to map models (Upscale, BBox, SegM) to their download URLs.
- **Dynamic UI**: Nodes like `⬡ UltimateSD Upscale` merge local scanned models with remote manifest entries. Remote files are auto-downloaded on execution via `download_bundle_files` before processing.
- **Outpaint Logic**: The passive `⬡ Image Process (Outpaint)` node sets target dimensions in `UME_IMAGE`. The `BlockSampler` uses this to apply **Replicate Padding + Moderate Gaussian Blur** prior to VAE encoding. This preserves local textures at the boundaries and totally eliminates "box" or "barcode" stretching artifacts!



### Coding Standards

**Naming:**

- Class Names: `UmeAiRT_` prefix (e.g., `UmeAiRT_BlockSampler`).
- Display Names: Prefixed with `⬡` for visual identification (e.g., `⬡ Image Generator`, `⬡ Checkpoint Loader`).
- Output names: `model_bundle` for loaders, `gen_pipe` for sampler/post-process.
- Pipeline parameters: Always use `gen_pipe` (not `pipeline` or `generation`).

**Registration:**

- All new nodes **MUST** be registered in `__init__.py` in two places:
    1. `NODE_CLASS_MAPPINGS`
    2. `NODE_DISPLAY_NAME_MAPPINGS`

### Advanced Inputs (Vue 2.0 Engine)

ComfyUI's Nodes 2.0 (Vue) layout engine natively suffers from a "ghost padding" bug when using `"advanced": True`. Visually hidden inputs still mathematically reserve their height in the node's `min-height`, causing massive empty gaps.
**The UmeAiRT Solution**: The `web/umeairt_colors.js` script contains an automated `computeSize` interceptor that subtracts the mathematical height of advanced widgets for all nodes prefixed with `UmeAiRT_`. 
**Rule**: You are free to use `"advanced": True` directly in your Python `INPUT_TYPES` for optional or heavy sliders. You do **not** need to split nodes into `_Simple` and `_Advanced` variants just to fix spacing gaps, as the Javascript layer will dynamically handle the CSS flex constraints for any `UmeAiRT_` node.

### File Structure

**Core:**
- `modules/common.py`: `GenerationContext`, `VideoGenerationContext`, `@dataclass` bundle types (`UmeBundle`, `UmeSettings`, `UmeVideoSettings`, `UmeImage`), pipeline helpers (`extract_pipeline_params`, `validate_bundle`, `PipelineParams`), shared helpers (`resize_tensor`, `encode_prompts`, `apply_outpaint_padding`), `KNOWN_DIT_MODELS` constant.
- `modules/logger.py`: Standard logging utility.
- `modules/optimization_utils.py`: Environment and optimization checks (SageAttention, Triton).
- `modules/monitor_hardware.py`: Hardware monitoring backend — `GPUBackend` cascade (NVIDIA/AMD/MPS/CUDA fallback), `HardwareMonitor` aggregator, `MonitorService` daemon thread (WebSocket broadcast).
- `modules/manifest.py`: Model manifest management — fetch, cache, parse HuggingFace manifest, `download_bundle_files()`, `get_bundle_dropdowns()`.
- `modules/download_utils.py`: Download engine (aria2c + urllib fallback), SHA256 verification.

**Image Pipeline (Hub-and-Spoke):**
- `modules/block_inputs.py`: LoRA blocks, ControlNet, GenerationSettings (→ `UME_SETTINGS`), VideoSettings, LTXVideoSettings, Image Loader/Process, Prompt Inputs.
- `modules/block_loaders.py`: Model Loaders (→ `UME_BUNDLE`), BundleAutoLoader.
- `modules/block_lightning.py`: Lightning Accelerator middleware — auto-loads acceleration LoRAs and injects bundle overrides.
- `modules/block_sampler.py`: BlockSampler hub (→ `UME_PIPELINE`). Reads `UmeBundle.overrides` from middleware nodes.
- `modules/block_passthrough.py`: PackPipeline — alternative to BlockSampler for upscale-only workflows (creates `GenerationContext` without sampling).
- `modules/flux_sampler.py`: FLUX-specific sampling pipelines.
- `modules/sampler_tasks.py`: Mode-specific image preparation (txt2img, img2img, inpaint, outpaint).
- `modules/sampler_cache.py`: Prompt encoding cache.
- `modules/sampler_controlnet.py`: ControlNet loading/caching.
- `modules/extra_samplers.py`: Custom KSampler algorithms (SA-Solver, RES Multistep).

**Post-Processing:**
- `modules/logic_nodes.py`: Re-export shim — imports from `upscale_nodes`, `seedvr2_nodes`, `face_nodes`, `detail_daemon_nodes`, `detail_refiner`.
- `modules/upscale_nodes.py`: UltimateSD Upscale node.
- `modules/seedvr2_nodes.py`: SeedVR2 Upscale node.
- `modules/face_nodes.py`: Subject Detailer (FaceDetailer).
- `modules/detail_daemon_nodes.py`: Detailer Daemon.
- `modules/detail_refiner.py`: Detail Refiner.
- `modules/image_nodes.py`: Pipeline Image Saver.
- `modules/image_analyze.py`: Image to Prompt (VLM-powered, Qwen3-VL).

**Video Pipeline (Unified WAN + LTX):**
- `modules/video_sampler.py`: Unified Video Generator orchestrator (dispatches WAN/LTX based on `loader_type`).
- `modules/wan_sampler.py`: WAN video generation logic (T2V, I2V, VACE, FunControl, MoE).
- `modules/ltx_sampler.py`: LTX-2.3 video generation logic (dual-pass, AV, ManualSigmas) — free function, no class.
- `modules/video_utils.py`: Shared video utilities (`patch_wan_model()`, `apply_color_match()`).
- `modules/video_extender.py`: Unified Video Extender orchestrator (dispatches WAN/LTX).
- `modules/wan_extender.py`: WAN video extension logic (VACE continuation).
- `modules/ltx_extender.py`: LTX video extension logic (reference frames + AV latents) — free function, no class.
- `modules/ltx_enhancer.py`: LTX Video Enhancer (LoopingSampler).
- `modules/ltx_keyframe_generator.py`: LTX Keyframe Generator (multi-keyframe conditioning).
- `modules/ltx_prompt_director.py`: Prompt Segment + Prompt Director (temporal scheduling).
- `modules/ltx_audio_replacer.py`: LTX Audio Replacer.
- `modules/ltx_utils.py`: LTX spatio-temporal tiled VAE decode.
- `modules/video_slicer.py`: Video Slicer (trim/segment).
- `modules/video_output.py`: Video Output with audio muxing.
- `modules/video_postprod.py`: Video Frame Interpolation, Smart Upscale.
- `modules/video_looper.py`: WAN Video Looper (seamless loop via VACE, uses `video_utils`).
- `modules/video_vace_prep.py`: VACE Prep node.
- `modules/video_funcontrol.py`: Video ControlNet Apply (FunControl prep).
- `modules/video_lightning.py`: Video Lightning Accelerator.
- `modules/video_optimization.py`: Video VRAM optimization (CFGZeroStar, EasyCache, NAG).

**Utilities:**
- `modules/utils_nodes.py`: Pack/Unpack interoperability nodes, Bundle Downloader, Signature.
- `modules/block_nodes.py`: Re-export shim for backward compatibility.

**Infrastructure:**
- `__init__.py`: Registration and exposing nodes to ComfyUI.
- `web/`: Javascript/CSS extensions (UI tweaks, colors, Nodes 2.0 enforcements, hardware monitor).
- `*/core/`: Integrated libraries (e.g., `usdu_core`, `seedvr2_core`, `facedetailer_core`, `image_saver_core`).
- `vendor/comfyui_gguf/`: Vendored `ComfyUI-GGUF` for `.gguf` weight loading.
- `vendor/ltxvideo/`: Vendored `ComfyUI-LTXVideo` samplers and latent helpers.
- `vendor/aria2/`: Bundled aria2c binary for accelerated downloads.
- `tests/`: Unit tests (mock: `python run_tests.py`) + GPU runtime tests (`tests/test_comfyui_runtime.py` — 45 tests).
- `.gitlab-ci.yml`: GitLab CI — lint, test matrix (Python 3.10-3.13), **test-comfyui** (GPU integration: real ComfyUI container + RTX 4080 + SDXL end-to-end sampling), MkDocs, GitLab release, Comfy Registry, Codeberg mirror.

## UI & Styling (Node Colors)

Nodes are color-coded by category in `web/umeairt_colors.js`:

| Category | Color Family | Hex (Bg/Fg) | Menu Location |
|----------|--------------|-------------|---------------|
| **Settings / Controls**   | Amber / Bronze | `#4A290B` / `#935116` | `UmeAiRT/Inputs`, `UmeAiRT/Image` |
| **Model / Files**         | Deep Blue      | `#0A2130` / `#154360` | `UmeAiRT/Loaders` |
| **Prompts**               | Dark Green     | `#0A2D19` / `#145A32` | `UmeAiRT/Inputs` |
| **LoRA**                  | Violet         | `#25122D` / `#4A235A` | `UmeAiRT/Loaders/LoRA` |
| **Samplers (Processors)** | Slate Gray     | `#1A252F` / `#2C3E50` | `UmeAiRT/Sampler` |
| **Post-Processing**       | Pale Blue / Teal | `#123851` / `#2471A3` | `UmeAiRT/Post-Process` |
| **Utilities**             | Dark Gray      | `#1A252F` / `#34495E` | `UmeAiRT/Utils` |
| **Image Inputs**          | Rust Red       | `#35160D` / `#6B2D1A` | `UmeAiRT/Image` |

**Connection colors:**

| Type | Color | Hex |
|------|-------|-----|
| `UME_BUNDLE` | Bright Blue | `#3498DB` |
| `UME_PIPELINE` | Teal | `#1ABC9C` |
| `UME_SETTINGS` | Amber/Copper | `#CD8B62` |
| `UME_IMAGE` | Orange/Brown | `#DC7633` |
| `UME_LORA_STACK` | Purple | `#9B59B6` |
| `UME_VACE_FRAMES` | Amber/Copper | `#CD8B62` |

## Project Maintenance & Stability Rules

To avoid regressions and maintain a stable, production-ready codebase, adhere strictly to the following rules:

1. **Dependency Synchronization**: Always update `pyproject.toml` instantly when adding a new package to `requirements.txt`. They must mirror each other to guarantee seamless node installation for users. The `test_registration` suite validates this.
2. **Proper Exception Handling**: **NEVER** use bare exceptions (`except:` or `except: pass`). Always catch specific exceptions or use `except Exception as e:` and log the error via `log_node()` so failures are visible during debugging.
3. **Changelog Maintenance**: All notable modifications, bug fixes, or additions must be immediately documented in `CHANGELOG.md` following the *Keep a Changelog* format.
4. **Tooltip Requirement**: Every `INPUT_TYPES` parameter **MUST** have a `"tooltip"` key with a beginner-friendly description. The `test_tooltips` suite enforces this.
5. **Test Coverage**: Run `python run_tests.py` locally (mock tests) before submitting. CI runs 45+ GPU runtime tests automatically on push via the `test-comfyui` job (real ComfyUI container with RTX 4080). Coverage target: ≥65%.

## Critical Files

| File | Notes |
|------|-------|
| `modules/common.py` | Contains `GenerationContext`, `VideoGenerationContext`, `@dataclass` bundle types, pipeline helpers, and shared utilities. |
| `__init__.py` | Entry point. **Must be updated** when adding nodes via import from modules. |
| `modules/manifest.py` | Model manifest management — fetch, cache, download. Core of the auto-download system. |
| `modules/block_passthrough.py` | Alternative to BlockSampler — creates `GenerationContext` for upscale-only workflows. |
| `web/umeairt_colors.js` | Node color mapping. **Must be synced** with NODE_CLASS_MAPPINGS when adding/removing nodes. |
| `docs/codemaps/structure.md` | Overview of the modular organization. |
| `TODO.md` | Technical backlog (remaining items from critical analysis). |
| `tests/test_registration.py` | Validates NODE_CLASS_MAPPINGS ↔ NODE_DISPLAY_NAME_MAPPINGS sync, dep sync. |
| `tests/test_tooltips.py` | Regression test: every input must have a tooltip. |

## Common Pitfalls

| Don't | Do Instead |
|-------|-----------|
| Add separate image input/output to post-process nodes | Read/write `gen_pipe.image` from/to `UME_PIPELINE` |
| Create `GenerationContext` in a loader | Only `BlockSampler` and `PackPipeline` create `GenerationContext` |
| Return `MODEL`, `CLIP`, `VAE` separately from loaders | Return a single `UME_BUNDLE` dataclass |
| Forget `__init__.py` | Double-check registration after creating a new node class |
| Take native types as input without interop | Use `Pack Models Bundle` to convert native → UME, or `Unpack *` for UME → native |
| Name pipeline param `pipeline` or `generation` | Use `gen_pipe` everywhere |
| Add input without tooltip | Add `"tooltip": "description"` to every input dict |
| Skip tests | Run `python tests/test_*.py -v` before committing |
| Duplicate pipeline extraction | Use `extract_pipeline_params()` from `common.py` |
| Duplicate KNOWN_DIT_MODELS | Import from `common.py` |
| Add a node without a color entry | Sync `web/umeairt_colors.js` with `NODE_CLASS_MAPPINGS` |
| Use `reflect` padding for outpaint | Use `replicate` padding (see `apply_outpaint_padding()`) |
| Describe bundle types as `TypedDict` or `dict` | They are `@dataclass` instances (see `common.py`) |

## 🚨 Mandatory Verification Checklist

**Before marking any task as complete, you MUST verify:**

1. [ ] **`__init__.py` Updated**: Did you add the new node class to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py`?
2. [ ] **Web Directory**: If the node has frontend code, is it in `web/` and registered?
3. [ ] **Tooltips**: Does every input have a beginner-friendly `"tooltip"` key?
4. [ ] **Tests Pass**: Did you run `python run_tests.py` locally and all tests pass? The CI `test-comfyui` job will additionally run 45 GPU runtime tests with real ComfyUI + SDXL end-to-end sampling.
5. [ ] **Syntax Check**: Did you do a final syntax check on the files you edited (especially big lists like mappings)?
6. [ ] **User Notification**: Did you tell the user *exactly* where to find the new node (Category/Name)?
