# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.12.9] — 2026-06-24

### Added
- **WAN 2.2 Frame-to-Frame (FLF2V)**: Implemented support for Frame-to-Frame generation using WAN 2.2 I2V models with start and end frames anchoring, replicating ComfyUI's native mask-reshaping and CLIP Vision dual-image encoding.

### Fixed
- **⬡ WAN Loader**: Fixed a bug where the manual WAN loader was searching for CLIP Vision models in the `clip` directory instead of the `clip_vision` directory, causing silent load failures and falling back incorrectly to VACE mode.

## [1.12.8] — 2026-06-22

### Added
- **⬡ Video Loader**: Added a new native video loader node (`UmeAiRT_BlockVideoLoader`) that uses PyAV to decode video files from the `input` directory, supporting average framerate detection (with optional `force_fps` override) and audio stream extraction, packaged into the unified `UME_VIDEO_PIPELINE` format.
- **⬡ Video Concatenate**: Added a new post-processing node (`UmeAiRT_VideoConcat`) to merge two videos together, supporting automatic resolution matching, FPS selection, and robust audio stream resampling/mono-stereo/silence-padding logic.
- **Pack Video Pipeline**: Updated `UmeAiRT_Pack_VideoPipeline` (`Pack Video Pipeline`) with an optional `audio` input port to support embedding audio streams in manual pipelines.

### Fixed
- **Block Nodes Re-export**: Added the missing `UmeAiRT_BlockVideoLoader` class to `modules/block_nodes.py` to ensure it is properly re-exported and can be imported at startup without causing an `ImportError`.
- **Web Node Styling**: Added color style entries in `web/umeairt_colors.js` for `⬡ Video Loader`, `⬡ Video Slicer`, and `⬡ Video Concatenate` nodes.

## [1.12.7] — 2026-06-16
### Fixed
- Fixed a bug in `UmeAiRT_Unpack_Pipeline` node where the input `pipeline` didn't match the Python argument name.

## [1.12.4] — 2026-06-07

### Fixed
- **Video Output Metadata**: Fixed metadata never being persisted in generated video files. Root cause: MP4 containers silently discard custom metadata keys — only `comment` and `description` survive. All metadata is now packed into the `comment` field as JSON, ensuring cross-format (MP4/WebM) compatibility.
- **Drag & Drop Workflow Reload**: Added `PROMPT` and `EXTRA_PNGINFO` hidden inputs to the `⬡ Video Output` node, enabling ComfyUI workflow JSON embedding. Users can now drag generated videos onto the canvas to reload the full workflow.
- **Generation Metadata**: The `⬡ Video Output` node now embeds a complete A1111-compatible `parameters` string (prompt, negative prompt, steps, sampler, scheduler, CFG, seed, model, LoRAs) and full generation context (duration, fps, frame count, shift, denoise) in every video file. WebM additionally writes native Matroska tags for direct tool access.

## [1.12.3] — 2026-06-07

### Fixed
- Fixed missing UI color mapping for the `⬡ ANIMA Loader` node (`UmeAiRT_FilesSettings_ANIMA`).

## [1.12.2] — Fix Test Dependencies Again

### Fixed
- Fixed UTF-16 encoding corruption in `requirements.txt` that broke test dependency sync verification.

## [1.12.1] — Fix Test Dependencies

### Fixed
- Fixed a missing `requests` dependency in `requirements.txt` causing GitLab CI test suite to fail.

## [1.12.0] — Anima Support & Progress Bar Fixes

### Added
- **Anima Model Support**: Added native integration and dynamic VRAM staging for the new Anima Base/fp16 model and its variants.
- **Anima Loader/Paths**: Updated `model_manifest.json` parsing, `block_loaders.py`, and `manifest.py` for correct resolution of `anima_diff` folder paths and safe-tensor loading.

### Fixed
- **Detail Refiner Progress Bar**: Re-injected `comfy.utils.ProgressBar` inside the `detail_refiner` node so the global hardware monitor top bar properly tracks refinement progress without generating intermediate preview frames.

## [1.11.1] — Hardware Monitor & Progress Bar
- **Built-In Hardware Monitor**: Integrated hardware monitoring directly into the top bar, replacing the need for Crystools.
  - **Multi-Platform GPU Support**: Cascade detection — NVIDIA (`pynvml`), AMD ROCm (`pyamdgpuinfo`), macOS Apple Silicon (`torch.mps`), and `torch.cuda` fallback for any CUDA-compatible device.
  - **Multi-GPU Support**: Full support for multi-GPU setups (RunPod, etc.) with per-GPU metrics.
  - **3 Switchable Styles**: Glassmorphism Pills, Accent Strip, and Micro Gauges — selectable in Settings → UmeAiRT → Monitor.
  - **Contextual Progress Bar**: Displays node-specific labels during pipeline execution (Generating, Upscaling T2, Detailing, Refining, Encoding, etc.) with percentage and per-tile tracking.
  - **Rich Tooltips**: Hover for GPU name, VRAM used/total, peak VRAM (double-click to reset), temperature.
  - **Warning States**: Visual alerts at 80% (warning) and 95% (critical) utilization.
  - **Settings**: Enable/disable monitoring, select style, adjust refresh rate (0.5s–5s).
- **New Files**: `modules/monitor_hardware.py` (backend), `web/umeairt_monitor.js` (frontend), `web/umeairt_monitor.css` (styles).
- **New API Routes**: `GET /umeairt/monitor/gpu-info`, `PATCH /umeairt/monitor/settings`.
- **New Dependency**: `pynvml` (NVIDIA monitoring, platform-conditional).

### Changed
- **UltimateSD Upscale Progress**: Tile sampling now sends WebSocket progress events via a progress-only callback (no preview images in node UI). Previously, `suppress_preview=True` disabled all progress feedback.

## [1.11.0] — Unified Video Pipeline & T2I Adapter

### Added
- **Unified Video Orchestrator**: The `⬡ Video Generator` and `⬡ Video Extender` now support both WAN and LTX models transparently based on the loaded bundle type. LTX-specific nodes are now backward-compatible aliases to the unified nodes.
- **⬡ Video Pipe to Image Pipe**: New adapter node to convert a `UME_VIDEO_PIPELINE` into a standard `UME_PIPELINE`, allowing single-frame video generations (T2I) to be directly saved with standard Image Savers.
- **T2I Support for WAN**: The `⬡ Video Settings` duration slider now accepts `0.0`, triggering the generation of a single frame for pure Text-to-Image / Image-to-Image using WAN models.

### Changed
- **Video Settings UI**: Consolidated all video settings (WAN + LTX) into the single `⬡ Video Settings` node. LTX-specific fields (`frame_rate`, `audio_enabled`, `sigmas_preset`) and `shift` are now grouped as advanced parameters. `frame_rate` is now a slider.
- **Modular Video Engine**: Split monolithic pipeline logic into focused sub-modules: `wan_sampler.py`, `wan_extender.py`, `video_utils.py` while simplifying the orchestrator nodes.

## [1.10.0] — VACE (Video All-in-One Creation and Editing) Support

### Added
- **⬡ Video VACE Prep**: New preparation node for VACE workflows. Takes a `start_image` and an optional `end_image` to package them into a `UME_VACE_FRAMES` bundle.
- **VACE Conditioning**: The `⬡ Video Generator` now automatically detects `vace_frames` and builds the native WanVaceToVideo conditioning (split inactive/reactive VAE encoding, 8x8 block-pixelized masks).
- **VACE Color Matching**: Added an automatic Reinhard color transfer (`color_match=True`) to the Video Generator to correct the color drift commonly introduced during VACE frame generation.
- **VACE Lightning Fallback**: The `⬡ Video Lightning` accelerator now gracefully falls back to the T2V Lightning LoRA when a VACE model is loaded, since no dedicated VACE-distilled models exist yet.
- **New Type `UME_VACE_FRAMES`**: Dedicated data bundle type for VACE conditioning with Amber/Copper connection color.

## [1.9.0] — WAN 2.2 LoRA Block with High/Low Noise Targeting

### Added
- **⬡ WAN LoRA 1x / 3x / 5x / 10x**: New LoRA Block nodes with per-slot High/Low noise expert targeting for WAN 2.2 MoE pipelines. Each slot includes a `target` dropdown (`Both` / `High-Noise Only` / `Low-Noise Only`) to direct the LoRA to the correct expert model. Available under `UmeAiRT/Loaders/LoRA/WAN`.
- **4-Tuple LoRA Stack Format**: The `UME_LORA_STACK` type now supports an optional 4th element `(name, model_str, clip_str, target)`. Legacy 3-tuple entries default to `"both"` for full backward compatibility.
- **Targeted LoRA Routing in Video Generator**: The `⬡ Video Generator` now routes each LoRA to the correct MoE expert based on the target field. LoRAs targeting `"high"` skip the low-noise model, and vice versa.

## [1.8.1] — Critical Analysis Cleanup

### Fixed
- **Exception Handling**: Standardized bare `except Exception: pass` blocks across 6 files. Critical failures (MoE LoRA loading in `video_sampler.py`) now log warnings via `log_node()`. Non-critical fallbacks annotated with clarifying comments.
- **`OUTPUT_NODE` on Middleware**: Removed `OUTPUT_NODE = True` from `VideoLightningAccelerator` — middleware nodes returning `UME_BUNDLE` should not force re-execution.
- **Redundant Exception Catch**: Simplified `except (AttributeError, Exception)` to `except Exception` in `flux_sampler.py` (AttributeError is a subclass of Exception).

### Changed
- **Loader Deduplication**: Unified `_load_single_model()` with `_load_diffusion_model()` in `block_loaders.py` — 20 lines of duplicated GGUF/dtype logic replaced by delegation via a new `model_path` parameter.
- **Type Hints**: Fixed incorrect `Dict[str, Any]` type annotations on `BlockSampler.process()` to use actual dataclass types (`UmeBundle`, `UmeSettings`, `UmeImage`). Cleaned unused `Dict`/`Any` imports from `block_sampler.py` and `block_loaders.py`.
- **Tooltip Test Coverage**: Expanded `test_tooltips.py` from 6 to 27 node module files. Fixed false positives caused by a 2-line context window (now 5 lines) and added `result` to skip keys.

## [1.8.0] — LTX-2.3 Keyframe, Prompt Director, Audio & Slicer (Phase 3)

### Added
- **⬡ LTX Keyframe Generator**: New node that generates video guided by 2 or 3 keyframe images. Auto-detects mode: connect `first_frame` + `last_frame` (2-keyframe) or add `middle_frame` (3-keyframe with auto-calculated midpoint). Uses `LTXVBaseSampler.optional_cond_images` for frame-accurate conditioning.
- **⬡ Prompt Segment**: Chainable prompt block that defines a temporal segment (`start_time` + `prompt`). Chain multiple segments to build a `UME_PROMPT_SCHEDULE` — same pattern as LoRA Blocks.
- **⬡ LTX Prompt Director**: Consumes a chain of Prompt Segments and generates video with per-chunk prompt conditioning. Each temporal chunk gets the conditioning from the matching time window via `LTXVLoopingSampler.optional_positive_conditionings`.
- **⬡ LTX Audio Replacer**: Two modes — "Replace from File" (swap in external AUDIO with auto trim/pad) and "Regenerate from Video" (re-run diffusion for audio only with video latent fully masked).
- **⬡ Video Slicer**: Generic post-processing node that trims video to a time range (start/end in seconds). Slices both frames and audio proportionally. Works with both LTX and WAN pipelines.
- **New Type `UME_PROMPT_SCHEDULE`**: Chainable prompt schedule type with Soft Green (#52BE80) connection color.
- **New Tests**: Test suites for all 5 new nodes — Video Slicer (slicing logic, audio trim, edge cases), Audio Replacer (mode validation, trim/pad), Keyframe Generator (index calculation), Prompt Director (schedule sorting, chaining, validation).

## [1.7.0] — LTX-2.3 Video Extender & Enhancer (Phase 2)

### Added
- **⬡ LTX Video Extender**: New node that extends an existing video by generating new frames conditioned on the last N seconds. Supports dual-pass pipeline (half-res → upscale → full-res), audio extension with time-based AV masking, reference frame injection via I2V, and configurable extension/reference durations.
- **⬡ LTX Video Enhancer**: New node that enhances/upscales video quality using the LTXVLoopingSampler with overlapping temporal chunks. Processes video in configurable chunk sizes with overlap blending for seamless quality enhancement. Supports guided re-sampling from original frames.
- **Vendored LTXVideo Utilities**: Zero external dependencies — all required classes from ComfyUI-LTXVideo (Apache 2.0) vendored into `vendor/ltxvideo/`: LTXVLoopingSampler, LTXVBaseSampler, LTXVExtendSampler, LTXVInContextSampler, LinearOverlapLatentTransition, LTXVSelectLatents, LTXVDilateLatent, LTXVAddLatentGuide, LTXVSetAudioVideoMaskByTime, LTXVAdainLatent, blur_internal, and IC-LoRA attention helpers.
- **New Tests**: Test suites for Video Extender and Video Enhancer node definitions, input validation, and tooltip coverage.

## [1.6.0] — LTX-2.3 Video+Audio Integration

### Added
- **LTX-2.3 Support**: Full architectural integration for Lightricks LTX-2.3 video+audio generation.
- **⬡ LTX Loader**: New loader node for LTX-2.3 models with Gemma 3 + LTX dual CLIP text encoders, video VAE (bf16), audio VAE (bf16), and optional spatial 2x latent upscaler. Supports GGUF quantized formats for both diffusion model and text encoders.
- **⬡ LTX Video Settings**: New settings node with LTX-specific parameters — resolution aligned to 32px, native 25fps frame rate, audio toggle, and ManualSigmas presets (Standard 8-step, Fast 4-step, Custom).
- **⬡ LTX Video Generator**: New dual-pass T2V + I2V pipeline. Orchestrates text encoding, ConditioningZeroOut, LTXVConditioning (frame_rate injection), empty video+audio latents, AV NestedTensor concatenation, dual-pass sampling with ManualSigmas, latent upscaling 2x, spatio-temporal tiled VAE decode, and audio VAE decode — all in a single node.
- **Audio Muxing in Video Output**: The `⬡ Video Output` node now automatically muxes decoded audio (AAC for MP4, Opus for WebM) into the output container when `ctx.audio` is present.
- **Spatio-Temporal Tiled Decode (`ltx_utils.py`)**: Vendored and inlined the minimal tiled VAE decode logic from ComfyUI-LTXVideo (Lightricks, Apache 2.0) — no external dependency required. Supports configurable spatial tiles, temporal chunk length, and overlap blending for low VRAM usage on long/high-res videos.
- **Extended Data Types**: `UmeBundle` gained `audio_vae` and `latent_upscale_model` fields. `UmeVideoSettings` gained `frame_rate`, `audio_enabled`, `sigmas_preset`, and `custom_sigmas` fields. `VideoGenerationContext` gained `audio` and `audio_vae` fields. All defaults are backward-compatible with existing WAN workflows.
- **Bundle Auto-Loader LTX Support**: The `⬡ Bundle Auto-Loader` now handles LTX-2.3 manifests with dual VAE files and latent upscale models.
- **46 New Tests**: Comprehensive test suite (`test_ltx_nodes.py`) covering LTXVideoSettings, LTXLoader, LTXVideoGenerator, ManualSigmas presets, UmeBundle/VideoSettings/VideoContext LTX fields, and ltx_utils tiled decode helpers.

### Fixed
- **Double Evaluation Bug**: Fixed `LTXVCropGuides.execute()` being called twice in a ternary expression in the dual-pass upscaler path.

## [1.5.2] — Cloud Container Optimization

### Added
- **Skip Hash Verification (`UMEAIRT_SKIP_HASH_CHECK`)**: New environment variable to bypass SHA-256 hash verification on model downloads. Set `UMEAIRT_SKIP_HASH_CHECK=1` in container entrypoints (RunPod, Vast.ai, etc.) to eliminate the costly full-file read on multi-GB models over network storage. Accepts `1`, `true`, or `yes` (case-insensitive). A yellow log message is emitted when verification is skipped so the behavior is always visible.

## [1.5.1] — Qwen3-VL Pipeline Overhaul

### Fixed
- **VLM Inference Loops**: Fixed a critical bug where the `⬡ 🔍 Image to Prompt` node produced infinite repetition loops (e.g., "shipwreck, shipwreck, shipwreck..."). Root cause: the `HFModelWrapper` bypassed `generation_config.json`, and `do_sample=False` was explicitly hardcoded, overriding the model's native sampling parameters.
- **8B Model Device Crash**: Replaced ComfyUI's `ModelPatcher` (designed for diffusion UNets) with HuggingFace's native `device_map="auto"` for VLM inference. This properly handles GPU/CPU offloading for models larger than VRAM (e.g., Qwen3-VL-8B on a 16GB GPU).
- **Model Switching Crash**: Fixed `RuntimeError: You can't move a model that has some modules offloaded` when switching between VLM models. Accelerate dispatch hooks prevent `.to("cpu")`; cleanup now uses `del` + `gc.collect()`.
- **EOS Token Handling**: The model's built-in `generation_config.json` now provides the correct dual EOS token IDs (`[151645, 151643]`) automatically, instead of a manually extracted single ID.

### Added
- **Seed Control**: New `seed` input allows reproducible results and varied outputs on the same image. Changing the seed forces ComfyUI to re-execute the node.
- **Anti-Repetition Guard**: `no_repeat_ngram_size=3` for Tags/Mixed modes physically blocks the model from emitting the same 3-token sequence twice. Post-processing deduplication provides an additional safety net.
- **Auto Max Tokens**: `max_tokens` defaults to `0` (auto), which selects mode-appropriate limits: Tags=64, Prompt=256, Mixed=384, Custom=512.

### Changed
- **Simplified Node UI**: Only `image`, `model`, and `mode` are visible by default. `custom_prompt`, `seed`, `max_image_size`, `max_tokens`, and `keep_loaded` are now advanced parameters.
- **Slider Widgets**: `max_image_size` and `max_tokens` now use slider display for easier adjustment.
- **Manifest Cleanup**: Purged legacy FP8 entries and hidden metadata files (`.gitattributes`, `.cache`) from `model_manifest.json`. Registered BF16 Qwen3-VL-2B and 8B models.
- **Removed Version Dropdown**: The redundant model version combobox has been removed; the system now auto-selects the first available version.

## [1.5.0] — WAN 2.2 MoE Architecture

### Added
- **WAN 2.2 14B MoE Support**: Full architectural integration for Mixture-of-Experts dual-stage video generation.
- **Dual-Pass Sampling**: `video_sampler.py` now splits the denoising schedule, automatically routing the High-Noise expert for the first half of the steps, and switching to the Low-Noise expert (with noise disabled) for texture and detail refinement.
- **Intelligent Bundle Loader**: The `⬡ Bundle Loader` dynamically detects MoE models in the manifest, simultaneously downloading and loading both experts into a unified `UmeBundle`.
- **Manual MoE Wiring**: The `⬡ WAN Loader` exposes a new optional `diff_model_low_noise` input port for manual 14B pipeline construction.
- **Optimized Interoperability**: `⬡ Pack Video Pipeline` and `⬡ Unpack Video Pipeline` nodes updated to handle the secondary MoE model.

### Fixed
- **Lightning LoRA Parity**: Resolved a critical MoE routing error in `video_lightning.py` where the High-Noise acceleration LoRA was inadvertently applied to the Low-Noise expert. The pipeline now correctly resolves, downloads, and independently applies the HIGH and LOW LoRAs to their respective models.
- **MoE Optimization Propagation**: Fixed CFGZeroStar and EasyCache middleware to ensure performance patches are uniformly applied to both High-Noise and Low-Noise models.

## [1.4.0] — Video Pipelines & Native Badges

### Added
- **GGUF Native Support**: All model loaders (WAN, FLUX, ZIMG, QWEN, HiDream) now actively scan and merge `.gguf` files into dropdown menus, bypassing ComfyUI's native cache limitations.
- **Pack/Unpack Video Pipelines**: Added `⬡ Pack Video Pipeline` and `⬡ Unpack Video Pipeline` nodes to allow manual construction and decomposition of `UME_VIDEO_PIPELINE` objects for advanced interoperability with community nodes.
- **Save Last Frame**: Added an advanced option `save_last_frame` to the `⬡ Video Output` node. When toggled, it extracts and saves the final frame as a PNG alongside the generated video, enabling seamless chaining for Image-to-Video generation sequences.
- **Premium Native UI Badges**: Deployed a dynamic Javascript CSS injector (`umeairt_badges.js`) that automatically upgrades the default ComfyUI-Manager extension tags into premium `✦ 𝒰𝓂𝑒𝒜𝒾𝑅𝒯-Toolkit` badges directly inside the Node Search menu.
- **Node Descriptions**: Injected explicit class `DESCRIPTION` metadata into all main Loader and Output nodes to provide clear context directly in the ComfyUI frontend search menu.

### Changed
- **Unified Dropdowns**: Merged legacy (`diffusion_models`, `clip`) and modern (`unet`, `text_encoders`) model directories in the UI dropdowns, ensuring compatibility with all ComfyUI file structure variations.
- **`clip_vision` Visibility**: Removed the `advanced` flag from the `clip_vision` input in the `⬡ WAN Loader`, making it visible by default since it is critically required for I2V processing.
- **Unclamped Video Output Node**: Removed hardcoded max-height constraints from the `⬡ Video Output` UI panel, preventing the node from arbitrarily shrinking user interfaces.

### Fixed
- **Post-Production Runtime Crash**: Resolved an `UnboundLocalError` in `video_postprod.py` where `comfy.model_management` was imported inside a conditional block but referenced globally.
- **Video Output Filenames**: Overhauled the filename parsing mechanism in `video_output.py` to match the robust string replacement logic used by the standard Image Saver (supporting `%date`, `%time`, `%seed`), fixing previous Regex failures.

## [1.3.9] — Lightning Acceleration LoRAs
- **⬡ ⚡ Lightning Accelerator**: New middleware node that sits on the `UME_BUNDLE` wire between a Loader and the Image Generator. Automatically loads the correct Lightning LoRA for the selected QWEN model variant and overrides generation settings for accelerated sampling.
  - **Combo selector**: Off / 4 Steps / 8 Steps — pure pass-through when Off.
  - **Auto-mapping**: Resolves the correct Lightning LoRA based on the model filename (Image → V2.0, Image_Edit → V1.0, Image_Edit_2509 → V1.0-fp32). Distill and unknown variants gracefully skip with a warning.
  - **Bundle overrides**: Embeds `cfg=1.0`, `steps`, `sampler`, `scheduler` into `UmeBundle.overrides` — the BlockSampler reads and applies them silently on top of user settings.
  - **Advanced options**: `sampler_name` (default: euler) and `scheduler` (default: sgm_uniform) exposed as advanced inputs.
  - **Auto-download**: Missing LoRA files are automatically downloaded from the `_ACCELERATION_LORAS` manifest section via aria2c.
- **`UmeBundle.overrides`**: New optional `dict` field on the `UmeBundle` dataclass, enabling any middleware node to inject settings overrides that the BlockSampler applies transparently.
- **Manifest: `_ACCELERATION_LORAS`**: New extensible manifest section (prefixed with `_` to stay hidden from Bundle Loader dropdown) containing Lightning LoRA entries per model family (QWEN, extensible to WAN and future families).
- **Manifest: `loras` path_type**: Added `loras` to `PATH_TYPE_TO_FOLDERS` mapping for proper LoRA auto-download destination resolution.

## [1.3.8] — Detail Refiner & Standalone Pack Pipeline

### Added
- **⬡ Detail Refiner**: New pipeline-aware post-processing node that performs a secondary sampling pass to enhance micro-details and textures without altering composition.
  - **Progressive Denoise**: Supports multi-pass refinement where denoise is automatically halved on each pass.
  - **Architecture-Aware**: Automatically detects and applies appropriate guidance for FLUX, Z-IMG (Lumina 2), and standard SDXL architectures.
  - **Lossless Interop**: Bypasses VAE Encode degradation on the first pass by leveraging the raw latent from the generation context.
  - **Silent Mode**: Executes seamlessly in the background without polluting the ComfyUI interface with intermediate live previews, while still logging progress.
- **⬡ Pack Pipeline**: New standalone node replacing the BlockSampler to enable direct image upscaling pipelines without generating an initial image.

## [1.3.7] — Lumina 2, HiDream & QWEN Integration

### Added
- **Z-IMG (Lumina 2) Support**: Full architectural support for Z-IMG/Lumina 2 models. Integrated dynamic shift handling in the `BlockSampler` (`shift = steps / 6.0`) using a custom `ModelSamplingDiscreteFlow` patch that preserves the native `multiplier=1.0` to prevent noise artifacts.
- **HiDream Support**: Added a dedicated `⬡ HiDream Loader`. The sampler now dynamically injects `ModelSamplingSD3` with `shift=2.0` (configurable via manifest) for HiDream models.
- **QWEN Support**: Added a dedicated `⬡ QWEN Loader` with full multi-text-encoder loading support.
- **Manifest Properties**: `UmeBundle` now exposes a `shift` float property inherited from the manifest's `_meta` block to allow dynamic sampling parameter injection without code changes.

### Changed
- **Manifest-Driven Architecture**: Refactored `block_loaders.py` and `block_sampler.py` to route model behaviors via explicit `loader_type` definitions from `model_manifest.json` instead of relying on fragile string heuristics (e.g., `is_flux_model`).
- **Pack Bundle Upgrades**: The `⬡ Pack Models Bundle` node now includes an optional `loader_type` dropdown to let users explicitly declare the model's architecture when manually packing custom models.
- **SHA Validation**: Added exact SHA256 hashes and file sizes for all Z-IMG Normal, QWEN, and HiDream variants into both Assets and Toolkit manifests, ensuring strict integrity checks.

### Fixed
- **Latent Channel Bug**: Removed a hardcoded `latent_channels == 16` heuristic in the sampler that was falsely triggering FLUX logic for non-FLUX models and breaking generation for Lumina 2 and HiDream.


## [1.3.6] — FLUX ControlNet Union Pro Integration

### Added
- **FLUX ControlNet Support**: Full native support for the `Shakker-LabsFLUX1-dev-ControlNet-Union-Pro` model via the `⬡ ControlNet Apply` node.
- **Dedicated FLUX Pipeline**: The BlockSampler now automatically routes FLUX ControlNet requests through a specialized `BasicGuider` + `SamplerCustomAdvanced` pipeline, as the standard `KSampler` (CFGGuider) is incompatible with FLUX ControlNet patches.
- **Control Image Resizing**: Control images are now automatically resized to match the generation dimensions before the ControlNet is applied, ensuring high-fidelity guidance.
- **UI Enhancements**: `start_percent` and `end_percent` inputs on the `⬡ ControlNet Apply` node now use slider widgets with `0.01` steps for better precision.

### Fixed
- **FLUX Union Type Mapping**: Corrected an issue where SDXL Union type numbers were being sent to the FLUX Union model. The toolkit now dynamically applies the correct type hint (`canny=0, tile=1, depth=2, blur=3, pose=4, gray=5, low_quality=6`) when a FLUX model is active.
- **VAE Pass-through**: Fixed an issue where the VAE was missing during ControlNet encoding. FLUX ControlNets operate in latent space and now properly receive the VAE instance.
- **SoftEdge Support**: Logs a helpful yellow warning when attempting to use the SoftEdge preprocessor with FLUX Union, as the model lacks a dedicated HED/SoftEdge type hint (generation continues without the hint).

## [1.3.5] — FLUX Kontext Integration

### Added
- **FLUX Kontext Support**: Integrated full support for the FLUX Kontext model (multi-image style/composition transfer).
- **⬡ Image Process (Kontext)**: New dedicated pre-processor node that accepts an optional secondary reference image and exposes the `auto_resize` toggle (Resize to Settings vs Keep Original).
- **Kontext Pipeline (`flux_sampler.py`)**: Added a specialized pipeline (`sample_flux_kontext`) that automatically stitches input images, scales them for the Kontext model, encodes them via VAE, and injects them into the conditioning using `ReferenceLatent`. 
- **Auto-routing**: The `⬡ Image Generator` node now detects `mode="kontext"` and correctly bypasses the standard Inpaint/Outpaint paths to route directly into the Kontext pipeline.

### Changed
- **UmeImage Dataclass**: Expanded internal `UmeImage` state to carry an optional `reference_image`.
- **Node UI Colors**: The `⬡ Image Process (Kontext)` node is now styled with the toolkit's signature Amber/Bronze color for the Image Process family.

## [1.3.4] — Cross-Platform GPU Hardening

### Fixed
- **DWPose MPS Memory Leak**: `torch_gc()` now calls `torch.mps.empty_cache()` on Apple Silicon, preventing GPU memory buildup during batch preprocessing.
- **DWPose Legacy OpenPose**: Removed dead `data.cuda()` calls in `body.py` and `hand.py` that were redundant with the existing `data.to(cn_device)`.

### Changed
- **DWPose TorchScript Auto-Detect**: `Wholebody`, `AnimalPoseImage`, `DwposeDetector`, and `AnimalposeDetector` now auto-detect the GPU backend (CUDA/ROCm → MPS → CPU) instead of defaulting to `"cuda"`.
- **FP8 MPS Warning**: Model loaders now log a warning when loading FP8 models on MPS (unsupported by PyTorch MPS backend).
- **Startup Logs**: SageAttention/Triton check shows "N/A (CUDA only)" on MPS instead of misleading ❌ marks.
- **Log Consistency**: Remaining "VRAM" reference in VAE warmup replaced with "GPU memory".

### Removed
- Dead commented-out `.cuda()` code in `body.py`, `hand.py`, `face.py` (inherited from OpenPose upstream).

## [1.3.3] — MPS & Cross-Platform GPU Support

### Fixed
- **SeedVR2 MPS Fallback** ([#1](https://github.com/UmeAiRT-Studio/ComfyUI-UmeAiRT-Toolkit/issues/1)): Fixed the SeedVR2 Upscale node silently falling back to CPU on macOS Apple Silicon (MPS). The `_build_configs()` method hardcoded `"cuda:0"` as the device string, ignoring MPS. Now delegates to ComfyUI's `model_management.get_torch_device()` which handles CUDA, MPS, ROCm, and DirectML automatically.
- **SeedVR2 MPS Offload**: On MPS (unified memory architecture), the offload device is now set to `"none"` instead of `"cpu"`, avoiding unnecessary CPU↔GPU sync overhead.

### Changed
- **GPU Memory Logging**: Renamed `get_cuda_memory()` to `get_gpu_memory()` with MPS support (reports unified memory via `psutil`). Old name preserved as alias for backward compatibility.
- **Backend-Agnostic Logs**: All SeedVR2 GPU memory check messages now use "GPU Memory" instead of "VRAM" for cross-platform accuracy.
- **Startup Memory Log**: The startup optimization check now identifies the active backend ("CUDA" vs "MPS (unified)") in the initial memory report.

## [1.3.2] - FLUX Fill Outpaint Quality Optimization

### Fixed
- **FLUX Fill Edge Grain**: Eliminated high-frequency TV static grain on the extreme edges of large FLUX Fill outpaint generations. The structural base mirror padding now uses `sharp_mirror=True` specifically for FLUX Fill, skipping the aggressive Gaussian blur. This prevents the flow-matching model from attempting to construct high-resolution details over a low-resolution blurry base.
- **Denoise Constraints**: Removed the forced `denoise=1.0` override for outpaint tasks, allowing users to leverage `denoise=0.85` or `0.90` with the new sharp mirror base for significantly higher structural coherence.
- **DifferentialDiffusion Reintegration**: Restored `DifferentialDiffusion` and the soft mask gradient for FLUX Fill outpaint to ensure the pristine center image is preserved with perfect blending.

## [1.3.1] — FLUX Fill Inpaint Hotfix

### Fixed
- **FLUX Fill Inpaint Regression**: Fixed a critical regression introduced in v1.2.0/v1.3.0 where FLUX Fill inpaint mode was completely broken. The outpaint refactoring added a `is_outpaint` gate on the FLUX Fill delegation path, which blocked inpaint (where `is_outpaint=False`). Meanwhile, the latent setup was skipped via `pass` for detected FLUX Fill models, causing the pipeline to fall through to an empty txt2img latent — effectively ignoring the source image entirely.
- **Dead Code in Mode Branching**: Removed unreachable `"outpaint"` from the `elif mode_str in ["inpaint", "outpaint"]` branch — `"outpaint"` was already captured by the preceding `if`.

### Added
- **Dedicated FLUX Fill Inpaint Pipeline (`flux_sampler.sample_flux_inpaint`)**: New function that mirrors the outpaint pipeline but operates directly on the source image and mask without padding. Uses `InpaintModelConditioning` + `DifferentialDiffusion` + `BasicGuider` + `SamplerCustomAdvanced` — the same real ComfyUI nodes used for outpaint, adapted for inpaint.

## [1.3.0] — LoRA Info API & Architecture Improvements

### Added
- **LoRA Metadata Info (`/umeairt/lora-info`)**: New server-side API endpoint that reads safetensors headers (no GPU memory) to extract LoRA metadata: base model, trigger words (from `ss_tag_frequency`, sorted by frequency), network dim/alpha, resolution, training comment, and file size.
- **LoRA Info Frontend (`umeairt_lora_info.js`)**: New Nodes 2.0-compatible JS extension that adds context menu entries on all `⬡ LoRA` nodes:
  - `ℹ️ LoRA Info` — Displays full metadata with auto-copy of trigger words to clipboard.
  - `📋 Copy Trigger Words` — Quick clipboard copy of trigger words only.
  - Background pre-fetching on workflow load for instant response. In-memory cache avoids repeated API calls.
  - Tooltip populated with metadata (visible after canvas re-render with R key).

### Changed
- **Renamed `⬡ KSampler` → `⬡ Image Generator`**: The display name now reflects the node's full pipeline capabilities (prompt encoding, LoRA application, caching, ControlNet, image prep, sampling, VAE decode, inpaint compositing). No workflow breakage — the internal `NODE_CLASS_MAPPINGS` key (`UmeAiRT_BlockSampler`) is unchanged.
- **LoRA Strength Range**: Extended slider range from `[0.0, 2.0]` to `[-5.0, 5.0]` with step `0.05`. Negative values invert the LoRA effect (useful for advanced workflows). Fine values can be typed directly into the field.
- **Refactored `block_sampler.py`**: Decomposed the monolithic 446-line file into focused sub-modules for improved maintainability:
  - `sampler_tasks.py` — Mode-specific image preparation (txt2img, img2img, inpaint, outpaint) and inpaint compositing.
  - `sampler_cache.py` — Prompt caching with PromptCache class, ControlNet stack comparison, and zero-conditioning builder.
  - `sampler_controlnet.py` — ControlNet loading, caching, and application.
  - `flux_sampler.py` — Added `is_flux_model()` and `apply_flux_guidance()` helpers (previously inline in block_sampler).
  - `block_sampler.py` is now a thin orchestrator (~210 lines) that delegates to these sub-modules.

## [1.2.0] — FLUX Fill Pipeline & Architecture Improvements

### Added
- **Dedicated FLUX Fill Sampler (`flux_sampler.py`)**: New modular sampling pipeline for FLUX Fill outpainting that uses real ComfyUI core nodes directly (`ImagePadForOutpaint`, `InpaintModelConditioning`, `DifferentialDiffusion`, `BasicGuider`, `BasicScheduler`, `SamplerCustomAdvanced`, `ImageCompositeMasked`). Eliminates KSampler for FLUX Fill and ensures architectural parity with proven legacy workflows.
- **Manual Install Guide on Download Failure**: When model downloads fail, the manifest system now displays a clear per-file guide with exact destination paths and download URLs for offline installation.

### Changed
- **FLUX Fill Delegation**: `BlockSampler` now delegates entirely to `flux_sampler.sample_flux_outpaint()` when a FLUX Fill model is detected, reducing ~80 lines of inline FLUX logic.
- **Outpaint Padding**: Switched from `reflect` to `replicate` padding to prevent mirroring of central subjects (e.g., a bright sun) into padding zones. Added `skip_noise` parameter for FLUX Fill compatibility.
- **Loader Metadata**: FLUX and SDXL/Illustrious loaders now inject `loader_type` and `bundle_type` into `UmeBundle` for reliable Fill/Inpaint detection downstream.

### Fixed
- **Colorama Windows**: Fixed `colorama.init(convert=is_windows)` to properly handle ANSI color codes on Windows terminals.
- **Import Paths**: Fixed relative import paths in `image_saver_core/logic.py`, `seedvr2_core/seedvr2_adapter.py`, and `seedvr2_core/progress.py` with proper fallback chains.

## [1.1.3] — ControlNet & DWPose Hotfix

### Fixed
- **DWPose Blank Output**: Fixed a critical bug in `UmeAiRT_DWPose` where complex poses resulted in empty (black) skeleton maps. The OpenMMLab YOLOx internal detector expects `BGR` format but ComfyUI supplies `RGB` tensors. Injected an explicit channel conversion to resolve the issue.
- **Auto-Download Strictness**: Restored strict cryptographic (64-char SHA-256) validation for remote model downloads and removed silent failure bypasses. The toolkit will now properly halt generation if a `.safetensors` model is corrupted or missing, preventing users from generating unguided noise.

### Added
- **SDXL Union Support**: The `KSampler` block now natively injects the dynamic `control_type` argument required by `controlnet-union-sdxl-1.0` models based on the selected UmeAiRT preprocessor.
- **Illustrious/Pony Support**: Added official CDN integration for Illustrious-XL ControlNets (`canny`, `depth`, `openpose`) allowing them to be automatically downloaded and selected via the `⬡ ControlNet Apply` dropdown.
## [1.1.2] — FLUX Fill & Quality Fixes

### Added
- **FLUX Fill Inpaint Support**: The `KSampler` now automatically detects `FLUX/Fill` models via bundle metadata (`bundle_type: "image_inpaint"` and `loader_type: "flux"`). When detected, it replicates the native `InpaintModelConditioning` logic internally (encoding masked pixels with VAE and injecting `concat_latent_image` and `concat_mask` into the positive and negative conditioning), allowing FLUX Fill to natively perform highly coherent inpainting within the toolkit pipeline.

### Fixed
- **FLUX Guidance Bug**: Fixed a critical issue where the UI `cfg` slider was applied as standard Classifier-Free Guidance to FLUX models, heavily degrading details and causing a "fried" plastic look. The sampler now intercepts the `cfg` value for all FLUX models, maps it directly to the FLUX-specific internal `guidance` conditioning embedding, and safely forces the core sampler CFG to `1.0` (BasicGuider equivalent), preserving photorealism and microscopic details.
- **Aria2c Download Verifications**: Fixed a bug in `download_utils.py` where Windows path normalization issues caused `aria2c` downloads to be incorrectly flagged as failed. Added robust fallbacks checking direct file existence and aggressive cleanup of zombie `.aria2` control files.

## [1.1.1] — Outpaint Quality Fix

### Fixed
- **Outpaint Mirror Artifacts**: Fixed a critical bug where outpainted areas displayed a visible mirror/reflection of the source image instead of generating fresh content. Root cause was a triple failure: (1) the Gaussian blur kernel on reflect-padded pixels was capped at 21px — far too weak for large padding zones (160-320px), (2) the post-sampling compositing was blending back the padded (mirrored) image instead of the pristine original, and (3) there was no mechanism to break the symmetry pattern the model would latch onto.

### Changed
- **Outpaint Padding Blur**: Increased the blur kernel from a fixed cap of 21px to a dynamic `~2/3 × padding` formula (capped at 99px). For a 160px padding, the kernel now reaches 99px instead of 21px, effectively destroying any recognizable mirror pattern.
- **Outpaint Noise Injection**: Added 12% Gaussian noise injection into padded areas after blurring, giving the diffusion model a randomized starting point instead of a symmetric reflected pattern.
- **Outpaint Compositing**: Replaced the generic inpaint compositing path with a dedicated outpaint compositor that saves the pristine source image **before** padding and stamps it back into the generated output with a 48px feathered blend — guaranteeing zero visual drift in the original image area.
- **Outpaint Denoise Warning**: Instead of silently allowing low denoise values that cause mirror artifacts, the sampler now logs a yellow warning when `denoise < 0.8` in outpaint mode, recommending higher values for best results. The user retains full control.

## [1.1.0] — E2E GPU Testing & UX Improvements

### Added
- **CI/CD**: Massive expansion of the testing suite (from 2 to 171+ tests), including real GPU end-to-end sampling tests with an SDXL model.

### Changed
- **UX**: Changed the `clip_skip` parameter in the Checkpoint Loader to use a positive slider (1 to 24) instead of negative values, aligning with industry standards for a more intuitive user experience.

### Fixed
- **Logic**: Corrected parameter names for `apply_outpaint_padding` to prevent outpainting issues.
- **CI/CD**: Replaced `wget` with native Python `urllib` to ensure robust model downloads in the pipeline.
- **CI/CD**: Fixed torchaudio ABI mismatches and YAML syntax issues.

## [1.0.8] — Security & Perf Fixes

### Fixed
- **CLI Arg Extraction (CVE Prevention)**: Hid aria2 token injection from system process arguments logging to prevent credentials exposure via `ps`/`htop`.
- **Gemnasium CI Memory Fix**: Bounded the torch wheel dependencies lookup to prevent GitLab Runner Gemnasium Out-Of-Memory crashes.
- **Node Hash bypass**: Hardened node hash verification to prevent potential bad-actor tampering with the custom node imports.

## [1.0.7] — LoRA Slider Accuracy
### Fixed
- **LoRA Strength Bounds**: Fixed an issue where the LoRA `strength` sliders allowed absurdly high ranges (-10 to 10). The bounds have been narrowed back down to standard logical limits (`0.0` to `2.0`), restoring slider accuracy for users.

## [1.0.6] — Upscale Model Dropdown Fix
### Fixed
- **UltimateSD Upscale Dropdown Breakage**: Eliminated the `[⬇️] ` UI prefix for auto-downloadable upscale models. Resolves a critical issue where saved workflows using remote models would spontaneously break upon reloading ComfyUI because the UI prefix vanished after successful download, violating native ComfyUI strict frontend validation. Remote models now blend seamlessly into the dropdown and automatically download flawlessly in the background while fully preserving workflow continuity.

## [1.0.5] — Text Encoder Alignment
### Fixed
- **Text Encoders Path Extraction**: Modified the Bundle Loader manifest mapping so that text encoders (`T5`, `Qwen`, `LLaMa`, `Gemma`) download prioritize the `text_encoders` folder over the fallback `clip` folder, preventing unwanted cross-pollution of text encoders within the `clip` root.

### Changed

- **Menu Restructuring**: Reorganized all node categories from 12 fragmented submenus (3 levels deep) to 8 clean top-level categories (2 levels max). Old `UmeAiRT/Block/*` and `UmeAiRT/Pipeline/*` hierarchy replaced with user-friendly names: `Loaders`, `Loaders/LoRA`, `Inputs`, `Image`, `Sampler`, `Post-Process`, `Output`, `Interop`, `Utils`. No workflow breakage — CATEGORY changes are cosmetic and do not affect serialized workflows.

### Fixed

- **SeedVR2 Blackwell Crash**: Fixed stack overflow / access violation at ComfyUI startup on RTX 50xx GPUs. The `_probe_bfloat16_support()` function in `seedvr2_core` was creating CUDA tensors at import time before CUDA was fully initialized. Replaced with `torch.cuda.is_bf16_supported()` which queries GPU compute capability natively without touching CUDA memory. ([#3](https://gitlab.com/UmeAiRT-Studio/ComfyUI-UmeAiRT-Toolkit/-/issues/3))
- **Colorama Linux Crash**: Fixed `AttributeError: 'NoneType' object has no attribute 'erase_line'` on Linux caused by `colorama.init(convert=True)` forcing Win32 ANSI conversion on non-Windows platforms. Removed forced conversion to let colorama auto-detect the platform. ([#4](https://gitlab.com/UmeAiRT-Studio/ComfyUI-UmeAiRT-Toolkit/-/issues/4))

### Added

- **⬡ Image Process (Img2Img)**: New dedicated node for img2img workflows — denoise + optional auto-resize.
- **⬡ Image Process (Inpaint)**: New dedicated node for inpainting — denoise + mask_blur + optional auto-resize.
- **⬡ Image Process (Outpaint)**: New dedicated node using **target dimensions** (`target_width`, `target_height`) instead of raw padding pixels. Alignment (`center`/`left`/`right`/`top`/`bottom`) available in advanced settings. The node is passive — actual resize + padding is executed by the KSampler.
- **Outpaint logic in KSampler**: When receiving `mode="outpaint"`, the sampler resizes the source to fit within target dimensions (maintaining aspect ratio), computes padding from alignment, applies padding + mask generation + blur, then encodes and samples.
- **Upscale Models Auto-Download**: The `⬡ UltimateSD Upscale` node now dynamically populates its dropdown with auto-downloadable upscale models (`[⬇️]`) defined in the `_UPSCALE_MODELS` manifest block, seamlessly integrating with local files.

### Changed

- **Outpaint Padding Mechanism**: Replaced sharp `replicate` stretching with soft `reflect` padding (mirroring internal pixels) combined with a moderate gaussian blur. This utterly eliminates the "box" and "barcode" artifacts, granting the AI highly coherent, texture-rich gradients to outpaint over seamlessly.
- **Outpaint Overlap**: Increased outpaint mask overlap from 8 to 48 pixels to ensure a flawless blend between the original image and the generated extension.

### Changed

- **⬡ Image Loader**: Now returns only `UME_IMAGE` bundle (removed redundant raw `IMAGE` and `MASK` outputs). Use `⬡ Unpack Image Bundle` to extract raw tensors.
- **Outpaint node UX**: Replaced 4 raw padding inputs with 2 intuitive target dimension inputs. Alignment defaults to `center` — no mental math required.
- **UmeImage dataclass**: Added outpaint-specific fields (`outpaint_target_w`, `outpaint_target_h`, `outpaint_h_align`, `outpaint_v_align`, `outpaint_mask_blur`).

### Fixed

- **⬡ Signature node**: Fixed rendering as empty rectangle in ComfyUI Nodes 2.0 (Vue frontend). Restored canvas-based rendering with `computeSize` patch exclusion.
- **⬡ Signature node**: Added periodic DOM watchdog for widget persistence across tab switches and workflow reloads.
- **Image Process node colors**: Applied Ambre/Bronze theme to all Image Process variants.

---

### Changed

- **⬡ Display Names**: All 47 nodes prefixed with `⬡` for instant visual identification. Loader names clarified (e.g., `Model Loader` → `⬡ Checkpoint Loader`, `⬡ FLUX Loader`, `⬡ Z-IMG Loader`, `⬡ Fragmented Loader`).
- **Category Harmonization**: `UmeAiRT/Loaders` → `UmeAiRT/Block/Loaders`, `UmeAiRT/Pipeline/IO` → `UmeAiRT/Pipeline/Output`.
- **DRY: Pipeline Helpers**: Extracted `extract_pipeline_params()`, `validate_bundle()`, `PipelineParams` namedtuple, and `KNOWN_DIT_MODELS` constant into `common.py` — eliminated ~80 lines of duplicated code across 8 methods in `logic_nodes.py`.
- **Node Instance Caching**: `BlockSampler` now caches `VAEEncode`, `KSampler`, `VAEDecode` instances in `__init__()` instead of creating new objects per execution.
- **ControlNet Caching**: Added `_controlnet_cache` to `BlockSampler` for ControlNet model reuse across runs.
- **Bundle Validation**: `BlockSampler.process()` now validates model_bundle input via `validate_bundle()` before unpacking.
- **Latent Channel Detection**: Improved fallback with a YELLOW warning log instead of silent `pass` when `latent_format.latent_channels` is unavailable.

### Fixed

- **Import Hygiene**: Removed duplicate `import nodes as comfy_nodes` and dead seedvr2 top-level imports (`logic_nodes.py`). Moved inline imports (`weakref`, `warmup_vae`, `random`, `string`, `torchvision`) to module-level.
- **Silent Exception**: `bbox` folder registration in `__init__.py` now logs a message instead of silently passing.
- **Smoke Test Mocks**: Added missing `comfy.sd`, `comfy.utils`, `comfy.samplers`, `comfy_extras` mocks to `test_smoke.py`.

### Security

- **aria2c Header Fix**: Separated `--header` flag from its value in `_download_with_aria2()` (`block_loaders.py`) to prevent argument injection.

### Removed

- **Dead Classes**: Deleted `UmeAiRT_PipelineImageLoader` and `UmeAiRT_PipelineImageProcess` from `image_nodes.py` (never registered, leftover from refactoring).

### Added

- `TODO.md` for tracking remaining technical backlog items.
- `test_smoke.py` added to CI pipeline (`.github/workflows/ci.yml`).
- `*.bak` added to `.gitignore`.


### Fixed

- **PERF-04**: Fixed a VRAM memory leak in `BlockSampler` by removing `self._cnet_cache` and using `weakref` for `self._last_clip` allowing ComfyUI's VRAM manager to clear unused models correctly (`block_sampler.py`).
- **CORE-01**: Fixed Python global namespace pollution caused by `UltimateUpscale_Base` by ensuring `usdu_core` path is safely removed from `sys.path` via `finally` block (`logic_nodes.py`).
- **TEST-01**: Hardened the `test_smoke.py` mock strategy for `folder_paths` so the test suite passes consistently in isolated CI environments without failing on unexpected UI-specific attribute lookups.

### Fixed (Previous)

- **BUG-01**: Fixed 3 `NameError` crashes where `generation.width`/`height` was referenced while parameter was named `pipeline` (`image_nodes.py`, `block_inputs.py`).
- **BUG-02**: Fixed `NameError` in `PipelineFaceDetailer` — `super().face_detail()` received undefined `pipeline` instead of correct parameter (`logic_nodes.py`).
- **BUG-03**: Removed duplicate `_get_hf_token()` function definition that silently shadowed the first (`block_loaders.py`).
- **LOGIC-03**: Fixed `Detailer_Daemon_Simple` returning a raw tensor instead of `gen_pipe` on error (`logic_nodes.py`).
- **LOGIC-05**: Fixed `HealthCheck` report showing literal `\n` instead of newlines (`utils_nodes.py`).
- **UX-02**: `BboxDetectorLoader` now raises `RuntimeError` instead of silently returning `None` (`logic_nodes.py`).
- **UX-03**: Fixed `Log_Viewer` trigger max value (`utils_nodes.py`).

### Security

- **SEC-01**: Fixed path traversal bypass via `....` → `..` by using a `while` loop sanitizer in `ImageSaver` (`image_nodes.py`).
- **SEC-02**: Added `timeout=30/60` to all `urllib.request.urlopen` calls (`block_loaders.py`).

### Changed

- **Naming Unification**: Unified all `pipeline`/`generation` parameter names to `gen_pipe` across 6 files (~80 occurrences).
- **PERF-01**: Removed thread-unsafe global `scaled_dot_product_attention` monkey-patch from `SamplerContext`. Optimizations should be activated at ComfyUI startup level (`optimization_utils.py`).
- **PERF-02**: `warmup_vae` now uses a singleton `VAEDecode` instance instead of creating disposable objects (`optimization_utils.py`).
- **LOGIC-02**: Added ControlNet model caching to `BlockSampler` — models are loaded once and reused across runs (`block_sampler.py`).
- **CODE-03**: Refactored 4 identical LoRA Block classes into a single factory function (`block_inputs.py`).
- **JS-02**: Merged double `onNodeCreated` override into a single unified handler for both colors and sizing (`umeairt_colors.js`).
- **JS-03**: Replaced global `LGraphCanvas.prototype.drawNode` monkey-patch with per-node `onDrawForeground`/`onDrawBackground` callbacks (`umeairt_signature.js`).
- **JS-01**: Normalized JS import paths to relative (`umeairt_log_viewer.js`).
- **CODE-01**: Removed duplicate import lines (`block_inputs.py`, `logic_nodes.py`).
- **CODE-02**: Removed redundant local `log_node` import (`block_loaders.py`).
- **CODE-04**: Removed duplicate `colorama.init()` call (`__init__.py`).
- **CODE-05**: Moved `torchvision.transforms.functional` import to module level (`block_inputs.py`, `image_nodes.py`).
- **PERF-03**: Added module-level cache for `_load_bundles_json()` (`block_loaders.py`).
- **UX-04**: Wrote/improved 117 tooltips across all node inputs with beginner-friendly language (6 files).

### Added

- **Tests**: 4 new test files — `test_optimization.py` (8 tests), `test_block_inputs.py` (9 tests), `test_tooltips.py` (1 regression test), `test_registration.py` (7 tests). Total: 42 tests across 7 suites.
- **CI**: GitLab CI pipeline (`.gitlab-ci.yml`) running all tests on Python 3.10-3.13 with CPU-only PyTorch.

### Removed

- Removed deprecated `PipelineImageLoader` and `PipelineImageProcess` nodes (broken, replaced by Block image nodes).
- Removed unused `requests` and `matplotlib` dependencies from `requirements.txt` and `pyproject.toml`.

### Security

- Removed `hf_token` STRING input from `BundleLoader` to prevent token exposure in workflow JSON files. Token is now read automatically from `HF_TOKEN` env var or `~/.cache/huggingface/token`.
- Logs a helpful message with HuggingFace link when no token is found.

### Changed

- **Prompt Caching**: Enhanced `BlockSampler` "Fast Start" caching to be explicitly LoRA-aware, preserving ~30s performance gains while safely recompiling when upstream modifiers change.
- **Type Hinting**: Added strict Python type hints to core processor methods across `block_sampler.py`, `block_loaders.py`, and `block_inputs.py` to formally enforce `common.py` bundle contracts (`UME_BUNDLE`, `UME_SETTINGS`, `UME_IMAGE`, `UME_PIPELINE`).
- **Code Cleanup**: Removed unused legacy pipeline imports from `block_sampler.py` and `logic_nodes.py`.

### Fixed

- **Silent Exceptions**: Refactored multiple generic `except Exception: pass` blocks in `block_loaders.py` and `logic_nodes.py` to properly surface warnings (via `log_node(color="YELLOW")`) regarding missing HF tokens, missing Text Encoders, GGUF failures, and unavailable internals.

### Changed

- **Wireless → Pipeline Rename**: Renamed all 11 `Wireless*` classes and `NODE_CLASS_MAPPINGS` keys to `Pipeline*` (e.g., `UmeAiRT_WirelessImageSaver` → `UmeAiRT_PipelineImageSaver`).
- **Category Normalization**: Standardized all node categories to `UmeAiRT/Block/*`, `UmeAiRT/Pipeline/*`, `UmeAiRT/Utils/*` hierarchy.
- **DRY: Outpaint code**: Extracted ~40 lines of duplicated outpaint padding logic into `apply_outpaint_padding()` in `common.py`.
- **DRY: Prompt encoding**: Centralized inline CLIP prompt encoding into `encode_prompts()` in `common.py`.
- **Input/Output Consistency**: Renamed `pipeline` input parameter to `generation` across all post-processing nodes for consistency with the BlockSampler output name.
- **Display Name Cleanup**: Removed `(Block)`, `(Simple)`, and `(Pipeline)` suffixes from all node display names. Block Sampler renamed to KSampler.
- **Modular Split**: Split monolithic `block_nodes.py` (1426 lines) into 3 focused sub-modules: `block_inputs.py` (LoRA, ControlNet, Settings, Image, Prompts), `block_loaders.py` (Model Loaders, BundleAutoLoader), `block_sampler.py` (BlockSampler). `block_nodes.py` is now a re-export shim.
- **DRY: Bundle download helpers**: Extracted `_get_bundle_dropdowns()` and `_download_bundle_files()` shared helpers — used by both `BundleLoader` and `Bundle_Downloader`.

### Added

- `TypedDict` type definitions: `UmeBundle`, `UmeSettings`, `UmeImage` in `common.py`.
- `encode_prompts()` and `apply_outpaint_padding()` utility functions in `common.py`.
- `_get_hf_token()` helper for secure HuggingFace token retrieval.
- `tests/test_common.py`: 13 unit tests for core common.py components.
- **Bundle Downloader** (`UmeAiRT_Bundle_Downloader`): Standalone download utility — downloads model bundles to correct ComfyUI folders without loading into VRAM. Ideal for RunPod/cloud pre-downloading.

### Removed

- Deleted `UmeAiRT_BlockUltimateSDUpscale` and `UmeAiRT_BlockFaceDetailer` — duplicate of Pipeline equivalents with identical `UME_PIPELINE` interface.
- Removed legacy "Wireless" aliases from `NODE_CLASS_MAPPINGS`.

### Fixed

- Fixed `test_traversal.py` broken by removed `UME_SHARED_STATE`. Rewritten with 6 security test cases.
- Fixed 4 silent `except Exception: pass` — now log errors via `log_node()`.
- Cleaned `umeairt_colors.js`: removed ~30 phantom entries, fixed duplicate `UME_BUNDLE` slot color.
- Updated `AGENTS.md` and `README.md`: removed outdated references, documented new architecture.
- Fixed `NameError: pipeline` in `ImageSaver.save_images()` — missed reference during `pipeline` → `generation` rename.

### Changed (Architecture Refactoring)

- **Hub-and-Spoke Pipeline**: The `BlockSampler` is now the central hub that creates the `GenerationContext` (`UME_PIPELINE`). Loaders and settings nodes feed into it as side-inputs.
- **Loaders → `UME_BUNDLE`**: All 6 loader nodes (Checkpoint, FLUX, Fragmented, ZIMG, Advanced, BundleLoader) now return a single `UME_BUNDLE` dict `{model, clip, vae, model_name}` instead of three separate outputs.
- **`GenerationSettings` → `UME_SETTINGS`**: Returns a settings dict instead of requiring a pipeline input. No longer creates a `GenerationContext`.
- **`BlockSampler`**: Accepts `model_bundle` (UME_BUNDLE) + `settings` (UME_SETTINGS) as inputs, creates `GenerationContext` internally, stores sampled image within it, returns `UME_PIPELINE`.
- **Post-process nodes → pipeline-only**: All 8 post-processing nodes (UltimateUpscale Simple/Advanced, SeedVR2 Simple/Advanced, FaceDetailer Simple/Advanced, Detail Daemon Simple/Advanced) now read the image from `pipeline.image` and return `UME_PIPELINE` with the updated image. No more separate image input/output.
- **`ImageSaver` → pipeline-only**: Reads image from `pipeline.image` instead of a separate `images` input.
- **`BlockImageProcess`**: Removed `pipeline` input dependency. Added `auto_resize` flag that is stored in the `UME_IMAGE` bundle and acted upon by the `BlockSampler` using generation settings dimensions.
- **Display names**: Loader outputs renamed to `model_bundle`, sampler/post-process outputs renamed to `generation`.
- **`GenerationContext`**: Added `image` field, renamed `sampler` → `sampler_name`, `positive` → `positive_prompt`, `negative` → `negative_prompt`.

### Added

- Custom connection colors for `UME_BUNDLE` (#3498DB, bright blue) and `UME_PIPELINE` (#1ABC9C, teal) in `web/umeairt_colors.js`.
- **New node: `Unpack Pipeline`** — Decomposes a `UME_PIPELINE` into 15 native ComfyUI outputs (IMAGE, MODEL, CLIP, VAE, prompts, settings, denoise) for full interoperability with native and community nodes.
- Updated `Unpack Image Bundle` to output all 5 fields: image, mask, mode, denoise, auto_resize (previously only image and mask).
- **New node: `Pack Models Bundle`** — Packs native MODEL, CLIP, VAE into a `UME_BUNDLE` for use with Block nodes. Enables interoperability from any native or community loader into the UmeAiRT pipeline.

### Fixed

- Fixed critical installation issues by synchronizing `pyproject.toml` dependencies with `requirements.txt`.
- Removed duplicated and outdated class definitions (`UmeAiRT_FilesSettings_FLUX`, `UmeAiRT_FilesSettings_Fragmented`) in `modules/block_nodes.py`.
- Fixed manifest loading bug by correcting `bundles.json` reference to `umeairt_bundles.json` in `modules/utils_nodes.py`.
- Replaced numerous bare `except: pass` statements across the codebase with specific or generic exception handling to improve debuggability and stability.
- Restored missing activation switches (`lora_{i}_on`) in all `UmeAiRT_LoraBlock` nodes to properly toggle LoRAs on or off.
- Fixed `UmeAiRT_FilesSettings_Checkpoint_Advanced` incorrectly returning `UME_PIPELINE` instead of `UME_BUNDLE`, making it incompatible with the `BlockSampler`. Now returns a standard `UME_BUNDLE` dict like all other loaders.
- Fixed `UmeAiRT_Unpack_FilesBundle` accepting obsolete `UME_FILES` type; now accepts `UME_BUNDLE`.
- Fixed `UmeAiRT_Unpack_Settings` reading `sampler` key instead of `sampler_name`, causing it to always return the default `"euler"`.

### Removed

- Removed `UmeAiRT_WirelessKSampler` from `__init__.py` registrations (class was already deleted from `logic_nodes.py`, causing a latent `ImportError` at startup).
- Removed orphaned `UME_SHARED_STATE[KEY_LORAS]` write from `MultiLoraLoader` (Block nodes no longer read from global state).

### Added

- Added automated `tests/test_smoke.py` for validating core module imports and node class mappings.
- Implemented a startup "Health Check" node (or process) to validate dependencies and optimizations.
- Added `tests/test_traversal.py` for path traversal security regression testing.

### Security

- Added defense-in-depth path traversal guard in `ImageSaverLogic.save_images()` (`modules/image_saver_core/logic.py`). The output path is now validated with `os.path.abspath()` + `startswith()` to ensure it stays within the output directory, independently of caller-side sanitization.
