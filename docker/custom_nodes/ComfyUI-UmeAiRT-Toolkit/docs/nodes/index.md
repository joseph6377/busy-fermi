# Node Reference

All UmeAiRT nodes use the **⬡** prefix in ComfyUI and are organized into categories in the right-click menu under `UmeAiRT/`.

## Loaders

Load models, VAEs, and text encoders into memory. Menu: `UmeAiRT/Loaders`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Checkpoint Loader](checkpoint-loader.md) | ⬡ Checkpoint Loader | UME_BUNDLE | SD 1.5 / SDXL single-file checkpoints |
| [FLUX Loader](flux-loader.md) | ⬡ FLUX Loader | UME_BUNDLE | FLUX architecture (dual CLIP) |
| [Z-IMG Loader](zimg-loader.md) | ⬡ Z-IMG Loader | UME_BUNDLE | Z-IMAGE / Lumina2 architecture |
| [LTX Loader](ltx-loader.md) | ⬡ LTX Loader | UME_BUNDLE | LTX-2.3 video+audio (Gemma 3 dual CLIP, video/audio VAEs) |
| [Bundle Auto-Loader](bundle-loader.md) | ⬡ 📦 Bundle Auto-Loader | UME_BUNDLE | Auto-download + load from manifest |
| [Lightning Accelerator](lightning-accelerator.md) | ⬡ ⚡ Lightning Accelerator | UME_BUNDLE | Auto-apply QWEN Lightning LoRAs (4/8 steps) |

### LoRA (Loaders submenu)

Stack LoRA models for style/character modifications. Menu: `UmeAiRT/Loaders/LoRA`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [LoRA Blocks](lora-blocks.md) | ⬡ LoRA 1x/3x/5x/10x | UME_LORA_STACK | Stack LoRA models with strengths |

## Inputs

Configure generation parameters and prompts. Menu: `UmeAiRT/Inputs`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Generation Settings](generation-settings.md) | ⬡ Generation Settings | UME_SETTINGS | Dimensions, steps, CFG, sampler, seed |
| [Prompt Inputs](prompt-inputs.md) | ⬡ Positive/Negative Prompt | POSITIVE/NEGATIVE | Text prompt editors |

## Image

Load and process images for generation. Menu: `UmeAiRT/Image`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Image Loader](image-loader.md) | ⬡ Image Loader | UME_IMAGE | Load source image for img2img/inpaint |
| [Image Process](image-process.md) | ⬡ Image Process | UME_IMAGE | All-in-one: mode, denoise, resize, outpaint |
| [Image Process (Img2Img)](image-process.md#-image-process-img2img) | ⬡ Image Process (Img2Img) | UME_IMAGE | Dedicated img2img |
| [Image Process (Inpaint)](image-process.md#-image-process-inpaint) | ⬡ Image Process (Inpaint) | UME_IMAGE | Dedicated inpainting |
| [Image Process (Outpaint)](image-process.md#-image-process-outpaint) | ⬡ Image Process (Outpaint) | UME_IMAGE | Target dimensions + alignment |
| [ControlNet](controlnet.md) | ⬡ ControlNet Apply | UME_IMAGE | Apply ControlNet guidance to image bundle |

## Sampler

Central sampling hub. Menu: `UmeAiRT/Sampler`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [KSampler](ksampler.md) | ⬡ KSampler | UME_PIPELINE | Central sampling hub — creates the generation pipeline |

## Video

Video generation pipeline. Menu: `UmeAiRT/Video`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [LTX Video Settings](ltx-video-settings.md) | ⬡ LTX Video Settings | UME_VIDEO_SETTINGS | LTX-2.3 video+audio settings (ManualSigmas) |
| [LTX Video Generator](ltx-video-generator.md) | ⬡ LTX Video Generator | UME_VIDEO_PIPELINE | LTX-2.3 dual-pass T2V + I2V pipeline |
| ⬡ Video Generator | ⬡ Video Generator | UME_VIDEO_PIPELINE | WAN video generation (T2V + I2V) |
| ⬡ Video Settings | ⬡ Video Settings | UME_VIDEO_SETTINGS | WAN video settings (steps, CFG, scheduler) |
| ⬡ Video Output | ⬡ Video Output | — | Save video as MP4/WebM with optional audio muxing |

## Post-Process

Enhance generated images. Menu: `UmeAiRT/Post-Process`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [UltimateSD Upscale](ultimate-upscale.md) | ⬡ UltimateSD Upscale | UME_PIPELINE | Tiled upscaling with redraw |
| [SeedVR2 Upscale](seedvr2-upscale.md) | ⬡ SeedVR2 Upscale | UME_PIPELINE | AI-native upscaling |
| [Detail Refiner](detail-refiner.md) | ⬡ Detail Refiner | UME_PIPELINE | Multi-pass micro-detail enhancement |
| [Subject Detailer](facedetailer.md) | ⬡ Subject Detailer | UME_PIPELINE | Automatic face/hand enhancement |
| [Detailer Daemon](detailer-daemon.md) | ⬡ Detailer Daemon | UME_PIPELINE | Detail enhancement via sampling schedule |

## Output

Save final images. Menu: `UmeAiRT/Output`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Image Saver](image-saver.md) | ⬡ Image Saver | — | Save with metadata + naming |

## Interop (Pack / Unpack)

Convert between UmeAiRT bundles and standard ComfyUI types. Menu: `UmeAiRT/Interop`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Pack Pipeline](pack-pipeline.md) | ⬡ Pack Pipeline | UME_PIPELINE | Create pipeline from raw image for standalone upscale |

See [Pack / Unpack Overview](pack-unpack.md) for the full list of interoperability nodes.

## Utils

Standalone workflow helpers. Menu: `UmeAiRT/Utils`

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Bundle Downloader](bundle-downloader.md) | ⬡ 💾 Bundle Model Downloader | STRING | Pre-download models without loading |
| ⬡ UmeAiRT Signature | ⬡ UmeAiRT Signature | — | Branding node for canvas |

