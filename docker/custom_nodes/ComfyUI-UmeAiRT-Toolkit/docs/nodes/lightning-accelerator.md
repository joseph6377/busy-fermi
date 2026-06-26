# ⬡ ⚡ Lightning Accelerator

> Automatically apply Lightning acceleration LoRAs to QWEN models for ultra-fast generation (4 or 8 steps).

The Lightning Accelerator is a middleware node that sits on the `UME_BUNDLE` wire between any Loader and the Image Generator. It detects the QWEN model variant, loads the matching Lightning LoRA, and silently overrides generation parameters (CFG, steps, sampler, scheduler) for optimal accelerated sampling.

!!! tip "No rewiring needed"
    The node only touches the `UME_BUNDLE` wire. Your `Generation Settings` node stays connected to the sampler as usual — the Lightning overrides are applied transparently by the Image Generator.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `model_bundle` | `UME_BUNDLE` | ✅ | — | Model bundle from a Loader or Bundle Auto-Loader |
| `mode` | `COMBO` | ✅ | Off | `Off` / `4 Steps` / `8 Steps` — acceleration level |
| `lora_strength` | `FLOAT` | ❌ | 1.0 | Lightning LoRA strength (0.0–2.0, advanced) |
| `sampler_name` | `COMBO` | ❌ | euler | Sampler override (advanced) |
| `scheduler` | `COMBO` | ❌ | sgm_uniform | Scheduler override (advanced) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `model_bundle` | `UME_BUNDLE` | Bundle with Lightning LoRA applied + settings overrides embedded |

## How it Works

```
Bundle Auto-Loader ──▶ UME_BUNDLE ──▶ ⚡ Lightning Accelerator ──▶ UME_BUNDLE ──▶ Image Generator
                                       [Off / 4 Steps / 8 Steps]                    (reads overrides)
```

1. **Off** → Pure pass-through. Zero overhead.
2. **4 Steps / 8 Steps** → The node:
    - Checks the model is QWEN (`loader_type == "qwen"`). Non-QWEN models get a warning and pass-through.
    - Resolves the correct Lightning LoRA based on the model filename.
    - Auto-downloads the LoRA from the UmeAiRT CDN if not found locally.
    - Applies the LoRA to the model and CLIP.
    - Embeds overrides (`cfg=1.0`, `steps=4|8`, `sampler=euler`, `scheduler=sgm_uniform`) into the bundle.
3. The **Image Generator** reads the embedded overrides and applies them on top of the user's Generation Settings.

## Supported Models

| QWEN Variant | 4 Steps LoRA | 8 Steps LoRA |
|-------------|-------------|-------------|
| Image (generation) | `Qwen-Image-Lightning-4steps-V2.0` | `Qwen-Image-Lightning-8steps-V2.0` |
| Image_Edit | `Qwen-Image-Edit-Lightning-4steps-V1.0` | `Qwen-Image-Edit-Lightning-8steps-V1.0` |
| Image_Edit_2509 | `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32` | `Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32` |

!!! note "Unsupported variants"
    Models without a Lightning LoRA (e.g. Image_Distill) will log a warning and pass through unchanged.
