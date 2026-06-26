# ⬡ Pack Pipeline

> A standalone node that creates a generation pipeline from an existing image, bypassing the initial text-to-image generation phase.

The Pack Pipeline is used to enable direct image upscaling or post-processing pipelines. It creates a `UME_PIPELINE` from raw components without executing a sampling block, making it ideal for pure image-to-image workflows like UltimateSD or SeedVR2 Upscale.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `image` | `IMAGE` | ✅ | — | The source image to pack into the pipeline |
| `bundle` | `UME_BUNDLE` | ❌ | — | Optional model bundle (Model, VAE, CLIP) |
| `settings` | `UME_SETTINGS` | ❌ | — | Optional generation settings |
| `positive` | `POSITIVE` | ❌ | — | Optional positive conditioning |
| `negative` | `NEGATIVE` | ❌ | — | Optional negative conditioning |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `gen_pipe` | `UME_PIPELINE` | A valid pipeline context ready for post-processing nodes |

## How it Works

Instead of relying on a `KSampler` to generate an initial image, the `Pack Pipeline` takes an already existing image and wraps it in a `GenerationContext`. Any optional inputs provided (models, prompts, settings) are attached to the context.

This allows downstream nodes like `⬡ UltimateSD Upscale` or `⬡ SeedVR2 Upscale` to function directly on user-provided images without throwing errors about missing pipeline dependencies.
