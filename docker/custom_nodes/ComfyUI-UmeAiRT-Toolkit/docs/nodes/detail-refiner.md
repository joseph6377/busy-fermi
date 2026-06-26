# ⬡ Detail Refiner

> Performs a secondary sampling pass to enhance micro-details and textures without altering global composition.

The Detail Refiner sits downstream of the main generator. It intercepts the raw latent and performs progressive refinement passes with mathematically decreasing denoise to sculpt high-frequency details.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `gen_pipe` | `UME_PIPELINE` | ✅ | — | Generation context |
| `passes` | `INT` | ✅ | 3 | Number of progressive refinement passes (1-5) |
| `denoise` | `FLOAT` | ✅ | 0.4 | Initial denoise strength. Halves every subsequent pass |
| `seed_offset` | `INT` | ✅ | 1 | Seed offset per pass to prevent color frying |

### Advanced Inputs (Overrides)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `steps` | `INT` | 0 | Override pipeline steps (0 = use pipeline) |
| `cfg` | `FLOAT` | 0.0 | Override pipeline CFG (0.0 = use pipeline) |
| `sampler_name` | `COMBO` | Pipeline | Override pipeline sampler |
| `scheduler` | `COMBO` | Pipeline | Override pipeline scheduler |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `gen_pipe` | `UME_PIPELINE` | Pipeline containing the refined latent/image |

## How it Works

1. **Lossless Interop**: On the first pass, it bypasses the standard VAE Encode step by directly retrieving the native `latent` from the `gen_pipe`. This prevents color degradation.
2. **Progressive Denoise**: If `passes = 3` and `denoise = 0.4`:
   - Pass 1: `denoise = 0.4`
   - Pass 2: `denoise = 0.2`
   - Pass 3: `denoise = 0.1`
3. **Turbo Auto-Inflation**: If a Turbo model is detected (e.g. FLUX Schnell, Z-IMG Turbo), the node automatically inflates the internal schedule resolution to `20` steps to guarantee enough discrete sampling operations and prevent artifacts.
4. **Silent Mode**: Executes invisibly in the background. Does not interrupt or pollute the ComfyUI canvas with heavy intermediate preview callbacks.
