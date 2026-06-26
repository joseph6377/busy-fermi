# ⬡ LTX Loader

> Loads LTX-2.3 video+audio models with Gemma 3 dual CLIP, video/audio VAEs, and optional spatial upscaler.

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `diff_model` | `COMBO` | ✅ | LTX-2.3 diffusion model (GGUF or safetensors) |
| `clip_gemma` | `COMBO` | ✅ | Gemma 3 text encoder (e.g. `gemma-3-12b-it-IQ4_XS.gguf`) |
| `clip_ltx` | `COMBO` | ✅ | LTX text projection / embeddings connector |
| `video_vae` | `COMBO` | ✅ | LTX Video VAE (e.g. `LTX2_video_vae_bf16.safetensors`) |
| `audio_vae` | `COMBO` | ✅ | LTX Audio VAE (e.g. `LTX2_audio_vae_bf16.safetensors`) |
| `latent_upscale_model` | `COMBO` | ❌ | Spatial upscaler 2x for dual-pass pipeline |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `model_bundle` | `UME_BUNDLE` | Complete LTX-2.3 bundle with model, dual CLIP, video VAE, audio VAE, and optional upscaler |

## How It Works

1. **Diffusion Model**: Loads the LTX-2.3 transformer (supports GGUF quantized and safetensors)
2. **Dual CLIP**: Loads Gemma 3 + LTX text projection as a combined text encoder (supports GGUF for both)
3. **Video VAE**: Loads the LTX video VAE (bf16)
4. **Audio VAE**: Loads the LTX audio VAE with prefix remapping (`audio_vae.` → `autoencoder.`)
5. **Upscaler** *(optional)*: Loads the spatial 2x latent upscaler for the dual-pass pipeline

!!! tip "GGUF Support"
    Both the diffusion model and text encoders support GGUF quantized formats for reduced VRAM usage.

!!! tip "Dual-Pass Pipeline"
    When a latent upscale model is connected, the LTX Video Generator automatically uses a dual-pass pipeline: Pass 1 at half resolution → upscale 2x → Pass 2 at full resolution.

!!! note "Alternative: Bundle Auto-Loader"
    If LTX-2.3 models are registered in `model_manifest.json`, you can use the `⬡ 📦 Bundle Auto-Loader` instead for automatic download and loading.
