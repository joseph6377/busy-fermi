"""
UmeAiRT Toolkit - LTX Video Sampler
--------------------------------------
LTX-2 specific video generation logic (dual-pass, AV latents, ManualSigmas).

Called by video_sampler.py orchestrator. Not a ComfyUI node itself.

Pipeline (matches reference ComfyUI workflow):
  LoRA → CLIPTextEncode → ConditioningZeroOut → LTXVConditioning (frame_rate) →
  EmptyLTXVLatentVideo (half res) → LTXVEmptyLatentAudio → [LTXVImgToVideoInplace] →
  LTXVConcatAVLatent → Pass 1 (LTXVScheduler, CFG=4, 20 steps, euler_ancestral) →
  LTXVSeparateAVLatent → LTXVCropGuides → LTXVLatentUpsampler → LTXVConcatAVLatent →
  Pass 2 (ManualSigmas [0.909375,0.725,0.421875,0.0], CFG=1, euler_ancestral, denoised_output) →
  LTXVSeparateAVLatent → VAEDecodeTiled (video) → Audio VAE Decode (audio)
"""

import torch
import nodes as comfy_nodes
import node_helpers
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import comfy.nested_tensor
import latent_preview
from .common import VideoGenerationContext, UmeBundle, UmeVideoSettings, log_node, validate_bundle
from .ltx_utils import ltx_spatio_temporal_tiled_decode
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# ManualSigmas for Pass 2 (from reference LTX-2 workflow)
# ---------------------------------------------------------------------------

PASS2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


# ---------------------------------------------------------------------------
# ManualSigmas presets (legacy — used by ltx_extender, ltx_audio_replacer, etc.)
# ---------------------------------------------------------------------------

SIGMA_PRESETS = {
    "standard": {
        "pass1": [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
        "pass2": [0.85, 0.725, 0.421875, 0.0],
    },
    "fast": {
        "pass1": [0.85, 0.725, 0.421875, 0.0],
        "pass2": [0.85, 0.725, 0.421875, 0.0],
    },
}


def _parse_sigmas(preset_name, pass_name, custom_sigmas_str=""):
    """Build a sigmas tensor from a preset name or custom string.

    Args:
        preset_name: "standard", "fast", or "custom".
        pass_name: "pass1" or "pass2".
        custom_sigmas_str: Comma-separated sigma values (only for "custom").

    Returns:
        1-D torch.FloatTensor of sigma values.
    """
    if preset_name == "custom" and custom_sigmas_str.strip():
        try:
            values = [float(v.strip()) for v in custom_sigmas_str.split(",") if v.strip()]
            return torch.FloatTensor(values)
        except ValueError:
            log_node(f"LTX Generator: Invalid custom sigmas '{custom_sigmas_str}', falling back to standard.", color="YELLOW")

    preset = SIGMA_PRESETS.get(preset_name, SIGMA_PRESETS["standard"])
    return torch.FloatTensor(preset[pass_name])


# Module-level LoRA loader (shared across calls)
_lora_loader = comfy_nodes.LoraLoader()


def _sample_custom_advanced(guider, sampler, sigmas, latent_dict, seed, use_denoised=False):
    """Reproduce the SamplerCustomAdvanced node behavior.
    
    This mirrors exactly what ComfyUI's SamplerCustomAdvanced does:
    - Generates noise via RandomNoise(seed)
    - Calls guider.sample() with callback for x0 tracking
    - Returns either 'output' or 'denoised_output' based on use_denoised flag
    
    Args:
        guider: CFGGuider instance
        sampler: Sampler object (e.g. euler_ancestral)
        sigmas: Sigma schedule tensor
        latent_dict: Latent dict with 'samples' and optional 'noise_mask'
        seed: Random seed for noise generation
        use_denoised: If True, return the x0 prediction (denoised_output). 
                      If False, return the raw sample (output).
    
    Returns:
        Dict with 'samples' key containing the result latent.
    """
    latent_image = latent_dict["samples"]
    
    # Fix empty latent channels (matches SamplerCustomAdvanced)
    latent_image = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher, latent_image,
        latent_dict.get("downscale_ratio_spacial", None),
        latent_dict.get("downscale_ratio_temporal", None)
    )
    
    noise_mask = latent_dict.get("noise_mask", None)
    
    # Generate noise
    noise = comfy.sample.prepare_noise(latent_image, seed)
    
    # Setup x0 callback for denoised_output tracking
    x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(
        noise, latent_image, sampler, sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    
    if use_denoised and "x0" in x0_output:
        # denoised_output: model's x0 prediction (clean latent)
        x0_out = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        if samples.is_nested:
            latent_shapes = [x.shape for x in samples.unbind()]
            x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
        result = latent_dict.copy()
        result.pop("downscale_ratio_spacial", None)
        result.pop("downscale_ratio_temporal", None)
        result["samples"] = x0_out
        return result
    else:
        # output: raw sample at final sigma
        result = latent_dict.copy()
        result.pop("downscale_ratio_spacial", None)
        result.pop("downscale_ratio_temporal", None)
        result["samples"] = samples
        return result


def generate_ltx(model_bundle: UmeBundle,
                 positive: str,
                 video_settings: UmeVideoSettings,
                 negative: str = None,
                 loras: Optional[List[Tuple[str, float, float]]] = None,
                 source_image=None):
    """Orchestrate the full LTX-2 video generation pipeline.

    Matches the reference ComfyUI LTX-2 T2V workflow exactly:
    - Pass 1: Full LTXVScheduler (20 steps), CFG=4, euler_ancestral → output
    - Pass 2: ManualSigmas [0.909375, 0.725, 0.421875, 0.0], CFG=1 → denoised_output

    Returns:
        Tuple of (VideoGenerationContext,)
    """
    validate_bundle(model_bundle, ["model", "clip", "vae"], context="Video Generator (LTX)")

    # --- Read settings ---
    width = video_settings.width
    height = video_settings.height
    duration = video_settings.duration
    fps = video_settings.frame_rate
    seed = video_settings.seed
    audio_enabled = video_settings.audio_enabled

    # Frame count aligned to 8n+1 (LTX requirement)
    frame_count = int(duration * fps) + 1
    frame_count = ((frame_count - 1) // 8) * 8 + 1

    model = model_bundle.model
    clip = model_bundle.clip
    vae = model_bundle.vae
    audio_vae = model_bundle.audio_vae
    latent_upscale_model = model_bundle.latent_upscale_model
    model_name = getattr(model_bundle, "model_name", "")

    pos_text = positive if isinstance(positive, str) else ""
    neg_text = negative if isinstance(negative, str) else ""

    is_i2v = source_image is not None
    mode_str = "I2V" if is_i2v else "T2V"
    has_upscaler = latent_upscale_model is not None

    # Reference workflow values
    p1_steps = video_settings.steps if video_settings.steps > 0 else 20
    p1_cfg = video_settings.cfg if video_settings.cfg > 0 else 4.0
    p2_cfg = 1.0  # Reference: CFG=1 for Pass 2 (always)
    sampler_name = video_settings.sampler_name or "euler_ancestral"

    log_node(f"🎬 LTX Generator: {mode_str} mode | {width}x{height} | "
             f"{duration}s ({frame_count} frames @ {fps}fps) | "
             f"Steps: {p1_steps} | CFG: {p1_cfg}/{p2_cfg} | Sampler: {sampler_name} | "
             f"Audio: {audio_enabled} | "
             f"Upscaler: {'Yes (dual-pass)' if has_upscaler else 'No (single-pass)'}", color="CYAN")

    # --- 1. Apply LoRAs ---
    applied_loras = []
    if loras:
        for lora_entry in loras:
            if len(lora_entry) >= 3:
                lora_name, strength_model, strength_clip = lora_entry[0], lora_entry[1], lora_entry[2]
            else:
                continue
            try:
                model, clip = _lora_loader.load_lora(
                    model, clip, lora_name, strength_model, strength_clip
                )
                applied_loras.append((lora_name, strength_model))
                log_node(f"  LoRA applied: {lora_name} (str={strength_model:.2f})", color="GREEN")
            except Exception as e:
                log_node(f"  LoRA failed: {lora_name}: {e}", color="RED")

    # --- 2. Encode prompts ---
    # Use encode_from_tokens_scheduled (same as CLIPTextEncode node) for proper
    # flow-matching model support (includes any extra dict keys like conditioning_lyrics)
    tokens_pos = clip.tokenize(pos_text)
    positive_cond = clip.encode_from_tokens_scheduled(tokens_pos)

    # LTX-2 reference: ConditioningZeroOut for negative conditioning
    # (zeroes out both the cond tensor and pooled_output, matching nodes.py ConditioningZeroOut)
    if neg_text:
        tokens_neg = clip.tokenize(neg_text)
        negative_cond = clip.encode_from_tokens_scheduled(tokens_neg)
    else:
        negative_cond = []
        for t in positive_cond:
            d = t[1].copy()
            pooled = d.get("pooled_output", None)
            if pooled is not None:
                d["pooled_output"] = torch.zeros_like(pooled)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            negative_cond.append([torch.zeros_like(t[0]), d])

    # --- 3. LTXVConditioning (inject frame_rate) ---
    positive_cond = node_helpers.conditioning_set_values(positive_cond, {"frame_rate": float(fps)})
    negative_cond = node_helpers.conditioning_set_values(negative_cond, {"frame_rate": float(fps)})
    log_node(f"  Conditioning: frame_rate={fps}", color="GREEN")

    # --- 4. Create empty latents ---
    # Reference: EmptyImage(full_res) → ImageScaleBy(0.5) → GetImageSize → EmptyLTXVLatentVideo
    if has_upscaler:
        pass1_w = ((width // 2) // 32) * 32
        pass1_h = ((height // 2) // 32) * 32
        if pass1_w < 512 or pass1_h < 320:
            log_node(f"  WARNING: {width}x{height} is too small for dual-pass. Disabling upscaler.", color="YELLOW")
            has_upscaler = False
            pass1_w = width
            pass1_h = height
    else:
        pass1_w = width
        pass1_h = height

    from comfy_extras.nodes_lt import EmptyLTXVLatentVideo
    video_out = EmptyLTXVLatentVideo.execute(width=pass1_w, height=pass1_h, length=frame_count, batch_size=1)
    video_latent_dict = video_out.args[0] if hasattr(video_out, 'args') else (video_out[0] if isinstance(video_out, tuple) else video_out)

    # Audio latent (if enabled)
    audio_latent_dict = None
    if audio_enabled and audio_vae is not None:
        from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio
        audio_out = LTXVEmptyLatentAudio.execute(audio_vae=audio_vae, frames_number=frame_count, frame_rate=fps, batch_size=1)
        audio_latent_dict = audio_out.args[0] if hasattr(audio_out, 'args') else (audio_out[0] if isinstance(audio_out, tuple) else audio_out)
        log_node(f"  Audio latent: {audio_latent_dict['samples'].shape}", color="GREEN")

    # --- 5. Optional I2V conditioning ---
    if is_i2v:
        from comfy_extras.nodes_lt import LTXVImgToVideoInplace
        video_out = LTXVImgToVideoInplace.execute(
            vae=vae, image=source_image, latent=video_latent_dict, strength=1.0
        )
        video_latent_dict = video_out.args[0] if hasattr(video_out, 'args') else (video_out[0] if isinstance(video_out, tuple) else video_out)
        log_node("  I2V: source image encoded into latent", color="GREEN")

    # --- 6. Combine AV latent ---
    if audio_latent_dict is not None:
        from comfy_extras.nodes_lt import LTXVConcatAVLatent
        concat_out = LTXVConcatAVLatent.execute(video_latent_dict, audio_latent_dict)
        av_latent = concat_out.args[0] if hasattr(concat_out, 'args') else (concat_out[0] if isinstance(concat_out, tuple) else concat_out)
    else:
        av_latent = video_latent_dict

    # --- 7. Pass 1 — Low-res sampling (full steps, output) ---
    # Reference: LTXVScheduler(20 steps, max_shift=2.05, base_shift=0.95, stretch=True, terminal=0.1)
    # Reference: CFGGuider(cfg=4), KSamplerSelect(euler_ancestral)
    # Reference: SamplerCustomAdvanced → uses 'output' (slot 0)
    from comfy_extras.nodes_lt import LTXVScheduler
    
    sigmas_out = LTXVScheduler.execute(
        steps=p1_steps,
        max_shift=2.05,
        base_shift=0.95,
        stretch=True,
        terminal=0.1,
        latent=av_latent
    )
    sigmas_p1 = sigmas_out.args[0] if hasattr(sigmas_out, 'args') else (sigmas_out[0] if isinstance(sigmas_out, tuple) else sigmas_out)

    log_node(f"  Pass 1: {p1_steps} steps, CFG={p1_cfg}, resolution {pass1_w}x{pass1_h}", color="CYAN")
    log_node(f"  Pass 1 sigmas: [{', '.join(f'{s:.4f}' for s in sigmas_p1[:5])}... {sigmas_p1[-1]:.4f}] ({len(sigmas_p1)} values)", color="CYAN")

    guider1 = comfy.samplers.CFGGuider(model)
    guider1.set_conds(positive=positive_cond, negative=negative_cond)
    guider1.set_cfg(p1_cfg)
    sampler = comfy.samplers.sampler_object(sampler_name)

    # Pass 1: use 'output' (raw sample, not denoised_output)
    pass1_result = _sample_custom_advanced(
        guider1, sampler, sigmas_p1, av_latent, seed,
        use_denoised=False  # Reference: uses 'output' slot
    )

    log_node("  Pass 1: ✅ Complete", color="GREEN")
    comfy.model_management.soft_empty_cache()

    # --- 8. Separate AV → CropGuides → Upscale video → Re-combine ---
    if has_upscaler:
        from comfy_extras.nodes_lt import LTXVCropGuides, LTXVConcatAVLatent
        
        has_audio_latent = pass1_result["samples"].is_nested
        
        # Separate AV latent (only if audio is present)
        if has_audio_latent:
            from comfy_extras.nodes_lt import LTXVSeparateAVLatent
            sep_out = LTXVSeparateAVLatent.execute(pass1_result)
            if hasattr(sep_out, 'args'):
                video_sampled = sep_out.args[0]
                audio_sampled = sep_out.args[1]
            else:
                video_sampled = sep_out[0]
                audio_sampled = sep_out[1]
        else:
            video_sampled = pass1_result
            audio_sampled = None

        # CropGuides — removes keyframe conditioning from latent (important for I2V)
        crop_out = LTXVCropGuides.execute(positive_cond, negative_cond, video_sampled)
        if hasattr(crop_out, 'args'):
            pos_p2, neg_p2, video_cropped = crop_out.args[0], crop_out.args[1], crop_out.args[2]
        else:
            pos_p2, neg_p2, video_cropped = positive_cond, negative_cond, video_sampled

        # Upscale video latent
        from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
        upscale_out = LTXVLatentUpsampler.execute(
            samples=video_cropped,
            upscale_model=latent_upscale_model,
            vae=vae
        )
        video_upscaled = upscale_out.args[0] if hasattr(upscale_out, 'args') else (upscale_out[0] if isinstance(upscale_out, tuple) else upscale_out)
        
        orig_shape = video_cropped["samples"].shape
        new_shape = video_upscaled['samples'].shape
        log_node(f"  Upscale: {orig_shape} → {new_shape}", color="GREEN")

        # Re-combine AV (only if audio was present)
        if audio_sampled is not None:
            concat_out = LTXVConcatAVLatent.execute(video_upscaled, audio_sampled)
            av_upscaled = concat_out.args[0] if hasattr(concat_out, 'args') else (concat_out[0] if isinstance(concat_out, tuple) else concat_out)
        else:
            av_upscaled = video_upscaled

        # --- 9. Pass 2 — Full-res sampling (ManualSigmas, CFG=1, denoised_output) ---
        # Reference: ManualSigmas("0.909375, 0.725, 0.421875, 0.0")
        # Reference: CFGGuider(cfg=1)
        # Reference: SamplerCustomAdvanced → uses 'denoised_output' (slot 1)
        comfy.model_management.soft_empty_cache()
        
        sigmas_p2 = torch.FloatTensor(PASS2_SIGMAS)
        
        log_node(f"  Pass 2: {len(sigmas_p2)-1} steps, CFG={p2_cfg}, resolution {width}x{height}", color="CYAN")
        log_node(f"  Pass 2 sigmas: [{', '.join(f'{s:.6f}' for s in sigmas_p2)}]", color="CYAN")

        guider2 = comfy.samplers.CFGGuider(model)
        guider2.set_conds(positive=pos_p2, negative=neg_p2)
        guider2.set_cfg(p2_cfg)  # Reference: CFG=1 for Pass 2
        sampler2 = comfy.samplers.sampler_object(sampler_name)

        # Pass 2: use 'denoised_output' (x0 prediction — reference uses slot 1)
        pass2_result = _sample_custom_advanced(
            guider2, sampler2, sigmas_p2, av_upscaled, seed,
            use_denoised=True  # Reference: uses 'denoised_output' slot
        )

        final_latent = pass2_result
        log_node("  Pass 2: ✅ Complete", color="GREEN")
    else:
        final_latent = pass1_result

    # --- 10. Decode ---
    # Separate AV if needed
    if final_latent["samples"].is_nested:
        from comfy_extras.nodes_lt import LTXVSeparateAVLatent
        sep_out = LTXVSeparateAVLatent.execute(final_latent)
        if hasattr(sep_out, 'args'):
            video_final = sep_out.args[0]
            audio_final = sep_out.args[1]
        else:
            video_final = sep_out[0]
            audio_final = sep_out[1] if len(sep_out) > 1 else None
        video_lat = video_final["samples"]
        audio_lat = audio_final["samples"] if audio_final else None
    else:
        video_lat = final_latent["samples"]
        audio_lat = None

    # Video decode (native ComfyUI VAEDecodeTiled — matches reference workflow)
    # Reference: VAEDecodeTiled(tile_size=512, overlap=64, temporal_size=4096, temporal_overlap=8)
    log_node("  Decoding video (VAEDecodeTiled native)...", color="CYAN")
    
    # --- WORKAROUND FOR COMFYUI ops.py BUG ---
    if hasattr(vae, "first_stage_model") and hasattr(vae.first_stage_model, "decoder"):
        needs_patch = False
        for module in vae.first_stage_model.decoder.modules():
            if hasattr(module, "timestep_conditioning") and module.timestep_conditioning:
                needs_patch = True
                break
        if needs_patch:
            log_node("  Applying VAE timestep_conditioning workaround", color="YELLOW")
            for module in vae.first_stage_model.decoder.modules():
                if hasattr(module, "timestep_conditioning"):
                    module.timestep_conditioning = False
    
    # Use ComfyUI native decode_tiled (same as VAEDecodeTiled node)
    tile_size = 512
    overlap = 64
    temporal_size = 4096
    temporal_overlap = 8
    
    compression = vae.spacial_compression_decode()
    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        t_tile = max(2, temporal_size // temporal_compression)
        t_overlap = max(1, min(t_tile // 2, temporal_overlap // temporal_compression))
    else:
        t_tile = None
        t_overlap = None
    
    frames = vae.decode_tiled(
        video_lat,
        tile_x=tile_size // compression,
        tile_y=tile_size // compression,
        overlap=overlap // compression,
        tile_t=t_tile,
        overlap_t=t_overlap
    )
    if len(frames.shape) == 5:  # Combine batches [B, T, H, W, C] → [B*T, H, W, C]
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    log_node(f"  Video decoded: {frames.shape[0]} frames", color="GREEN")

    # Audio decode
    audio_output = None
    if audio_lat is not None and audio_vae is not None:
        log_node("  Decoding audio...", color="CYAN")
        audio_decoded = audio_vae.decode(audio_lat).movedim(-1, 1).to(audio_lat.device)
        output_sample_rate = audio_vae.first_stage_model.output_sample_rate
        audio_output = {
            "waveform": audio_decoded,
            "sample_rate": int(output_sample_rate),
        }
        log_node(f"  Audio decoded: {audio_decoded.shape}, {output_sample_rate}Hz", color="GREEN")

    # --- 11. Build context ---
    ctx = VideoGenerationContext()
    ctx.model = model
    ctx.clip = clip
    ctx.vae = vae
    ctx.model_name = model_name
    ctx.width = width
    ctx.height = height
    ctx.duration = duration
    ctx.fps = fps
    ctx.frame_count = frames.shape[0]
    ctx.steps = p1_steps
    ctx.cfg = p1_cfg
    ctx.shift = 0.0
    ctx.sampler_name = sampler_name
    ctx.scheduler = "ltxv_scheduler"
    ctx.seed = seed
    ctx.denoise = 1.0
    ctx.loader_type = "ltx2"
    ctx.positive_prompt = pos_text
    ctx.negative_prompt = neg_text
    ctx.frames = frames
    ctx.loras = applied_loras
    ctx.source_image = source_image
    ctx.audio = audio_output
    ctx.audio_vae = audio_vae

    log_node(f"🎬 LTX Generator: ✅ {mode_str} complete — {frames.shape[0]} frames"
             f"{' + audio' if audio_output else ''}", color="GREEN")

    return (ctx,)
