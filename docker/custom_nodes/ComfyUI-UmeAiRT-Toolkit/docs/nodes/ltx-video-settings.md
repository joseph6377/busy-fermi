# ⬡ LTX Video Settings

> Defines LTX-2.3 video+audio generation parameters with ManualSigmas presets.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `width` | `INT` | ✅ | 768 | Video width (multiples of 32). 768 is native LTX-2.3 default |
| `height` | `INT` | ✅ | 512 | Video height (multiples of 32). 512 is native LTX-2.3 default |
| `duration` | `FLOAT` | ✅ | 5.0 | Video duration in seconds (0.5–30.0) |
| `frame_rate` | `INT` | ✅ | 25 | Frames per second. 25 is native LTX-2.3 rate |
| `seed` | `INT` | ✅ | 0 | Seed for random number generation |
| `audio_enabled` | `BOOLEAN` | ❌ | True | Generate audio alongside video |
| `sigmas_preset` | `COMBO` | ❌ | Standard (8 steps) | Sigma schedule preset |
| `custom_sigmas` | `STRING` | ❌ | *(empty)* | Comma-separated sigma values (Custom preset only) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `video_settings` | `UME_VIDEO_SETTINGS` | Complete settings bundle for the LTX Video Generator |

## Sigmas Presets

| Preset | Pass 1 Steps | Pass 2 Steps | Notes |
|--------|:---:|:---:|-------|
| **Standard (8 steps)** | 8 | 3 | Best quality — recommended for most use cases |
| **Fast (4 steps)** | 3 | 3 | Requires distilled LoRA for acceptable quality |
| **Custom** | User-defined | Standard Pass 2 | Enter comma-separated sigma values |

!!! tip "Custom Sigmas"
    For the Custom preset, enter comma-separated descending sigma values starting from 1.0 and ending at 0.0.
    Example: `1.0, 0.85, 0.725, 0.422, 0.0`

!!! note "Resolution"
    LTX-2.3 uses 32-pixel alignment (step=32) unlike WAN which uses 16-pixel alignment.

!!! note "Audio"
    Audio generation requires the Audio VAE to be loaded in the LTX Loader. Disable `audio_enabled` for video-only output.
