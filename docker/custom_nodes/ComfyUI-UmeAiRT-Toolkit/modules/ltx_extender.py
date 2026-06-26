"""
UmeAiRT Toolkit - LTX Video Extender
---------------------------------------
LTX-2.3 specific video extension logic. Takes an existing video from
UME_VIDEO_PIPELINE and extends it by generating new frames conditioned
on the last seconds of the source video.

Called by video_extender.py orchestrator. Not a ComfyUI node itself.

Pipeline:
  Extract reference frames → VAE Encode → Create AV latents →
  Apply time-based masks → Pass 1 (half-res) → Upscale → Pass 2 (full-res) →
  Decode video + audio → Concatenate with original → Updated VideoGenerationContext
"""

import torch
import nodes as comfy_nodes
import node_helpers
import comfy.samplers
import comfy.model_management
import comfy.utils
import comfy.nested_tensor
from .common import VideoGenerationContext, UmeBundle, UmeVideoSettings, log_node, validate_bundle
from .ltx_utils import ltx_spatio_temporal_tiled_decode
from .ltx_sampler import SIGMA_PRESETS, _parse_sigmas
from typing import Optional, List, Tuple

# Module-level LoRA loader
_lora_loader = comfy_nodes.LoraLoader()


def extend_ltx(video_pipe: VideoGenerationContext,
               model_bundle: UmeBundle,
               positive: str,
               video_settings: UmeVideoSettings,
               negative: str = None,
               loras: Optional[List[Tuple[str, float, float]]] = None,
               extend_seconds: int = 10,
               reference_seconds: int = 3,
               extend_audio: bool = True):
    """Extend video by generating new frames from the last reference frames.

    Returns:
        Tuple of (VideoGenerationContext,)
    """
    validate_bundle(model_bundle, ["model", "clip", "vae"], context="Video Extender (LTX)")

    source_frames = video_pipe.frames
    if source_frames is None or source_frames.shape[0] == 0:
        raise ValueError("Video Extender: No frames in the video pipeline.")

    # --- Read settings from video_settings ---
    width = video_settings.width
    height = video_settings.height
    fps = video_settings.frame_rate
    seed = video_settings.seed
    audio_enabled = video_settings.audio_enabled and extend_audio
    sigmas_preset = video_settings.sigmas_preset or "standard"
    custom_sigmas = video_settings.custom_sigmas or ""

    model = model_bundle.model
    clip = model_bundle.clip
    vae = model_bundle.vae
    audio_vae = model_bundle.audio_vae
    latent_upscale_model = model_bundle.latent_upscale_model
    model_name = getattr(model_bundle, "model_name", "")

    pos_text = positive if isinstance(positive, str) else ""
    neg_text = negative if isinstance(negative, str) else ""

    has_upscaler = latent_upscale_model is not None

    # Calculate frame counts
    ref_frames = min(int(reference_seconds * fps), source_frames.shape[0])
    ext_frame_count = int(extend_seconds * fps) + 1
    ext_frame_count = ((ext_frame_count - 1) // 8) * 8 + 1  # Align to 8n+1
    total_new_frames = ref_frames + ext_frame_count

    # Total frames for the AV latent (reference context + new extension)
    total_frame_count = total_new_frames
    total_frame_count = ((total_frame_count - 1) // 8) * 8 + 1

    log_node(f"🔗 LTX Extender: {source_frames.shape[0]} frames → +{ext_frame_count} frames | "
             f"Ref: last {ref_frames} frames ({reference_seconds}s) | "
             f"Sigmas: {sigmas_preset} | Audio: {audio_enabled} | "
             f"Upscaler: {'Yes' if has_upscaler else 'No'}", color="CYAN")

    # --- 1. Apply LoRAs ---
    applied_loras = list(video_pipe.loras) if video_pipe.loras else []
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

    # --- 2. Extract reference frames ---
    last_frames = source_frames[-ref_frames:]
    log_node(f"  Reference context: last {last_frames.shape[0]} frames", color="GREEN")

    # --- 3. Encode prompts ---
    tokens_pos = clip.tokenize(pos_text)
    cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
    positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]

    if neg_text:
        tokens_neg = clip.tokenize(neg_text)
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
        negative_cond = [[cond_neg, {"pooled_output": pooled_neg}]]
    else:
        negative_cond = []
        for t in positive_cond:
            d = t[1].copy()
            pooled = d.get("pooled_output", None)
            if pooled is not None:
                d["pooled_output"] = torch.zeros_like(pooled)
            negative_cond.append([torch.zeros_like(t[0]), d])

    positive_cond = node_helpers.conditioning_set_values(positive_cond, {"frame_rate": float(fps)})
    negative_cond = node_helpers.conditioning_set_values(negative_cond, {"frame_rate": float(fps)})

    # --- 4. VAE Encode reference frames ---
    log_node("  Encoding reference frames...", color="CYAN")
    ref_encoded = vae.encode(last_frames[:, :, :, :3])
    log_node(f"  Reference latent: {ref_encoded.shape}", color="GREEN")

    # --- 5. Create video latent for extension ---
    if has_upscaler:
        pass1_w = ((width // 2) // 32) * 32
        pass1_h = ((height // 2) // 32) * 32
    else:
        pass1_w = width
        pass1_h = height

    video_latent = torch.zeros(
        [1, 128, ((total_frame_count - 1) // 8) + 1, pass1_h // 32, pass1_w // 32],
        device=comfy.model_management.intermediate_device()
    )

    # Inject reference frames into the start of the latent
    ref_latent_frames = ref_encoded.shape[2]
    video_latent[:, :, :ref_latent_frames] = ref_encoded[:, :, :ref_latent_frames]

    # Create noise mask — reference frames are fixed, extension is denoised
    noise_mask = torch.ones(
        (1, 1, video_latent.shape[2], 1, 1),
        dtype=torch.float32,
        device=video_latent.device
    )
    noise_mask[:, :, :ref_latent_frames] = 0.0  # Reference is fixed

    video_latent_dict = {"samples": video_latent, "noise_mask": noise_mask}

    # --- 6. Audio latent (if enabled) ---
    audio_latent_dict = None
    if audio_enabled and audio_vae is not None:
        z_channels = audio_vae.latent_channels
        audio_freq = audio_vae.first_stage_model.latent_frequency_bins
        num_audio_latents = audio_vae.first_stage_model.num_of_latents_from_frames(total_frame_count, fps)
        audio_latent = torch.zeros(
            (1, z_channels, num_audio_latents, audio_freq),
            device=comfy.model_management.intermediate_device()
        )
        audio_latent_dict = {"samples": audio_latent, "type": "audio"}

        # Encode source audio into reference portion if available
        if video_pipe.audio is not None:
            log_node("  Encoding reference audio...", color="CYAN")
            source_waveform = video_pipe.audio["waveform"]
            sample_rate = video_pipe.audio["sample_rate"]
            ref_audio_samples = int(reference_seconds * sample_rate)
            if source_waveform.shape[-1] > ref_audio_samples:
                ref_audio = source_waveform[..., -ref_audio_samples:]
            else:
                ref_audio = source_waveform
            ref_audio_encoded = audio_vae.encode(ref_audio.movedim(1, -1))
            ref_audio_frames = ref_audio_encoded.shape[2]
            audio_latent[:, :, :ref_audio_frames] = ref_audio_encoded[:, :, :ref_audio_frames]
            log_node(f"  Audio latent: {audio_latent.shape} (ref: {ref_audio_frames} frames)", color="GREEN")

    # --- 7. Combine AV latent ---
    if audio_latent_dict is not None:
        av_samples = comfy.nested_tensor.NestedTensor(
            (video_latent_dict["samples"], audio_latent_dict["samples"])
        )
        av_latent = {"samples": av_samples}
        video_nm = video_latent_dict.get("noise_mask", None)
        audio_nm = audio_latent_dict.get("noise_mask", None)
        if video_nm is not None or audio_nm is not None:
            if video_nm is None:
                video_nm = torch.ones_like(video_latent_dict["samples"])
            if audio_nm is None:
                audio_nm = torch.ones_like(audio_latent_dict["samples"])
            av_latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_nm, audio_nm))
    else:
        av_latent = video_latent_dict

    # --- 8. Apply time-based masking ---
    if audio_enabled and audio_vae is not None:
        from vendor.ltxvideo.latents import LTXVSetAudioVideoMaskByTime

        ref_end_time = reference_seconds
        total_time = reference_seconds + extend_seconds

        masker = LTXVSetAudioVideoMaskByTime()
        _, _, av_latent, _, _ = masker.run(
            av_latent=av_latent,
            positive=positive_cond,
            negative=negative_cond,
            model=model,
            vae=vae,
            audio_vae=audio_vae,
            start_time=ref_end_time,
            end_time=total_time,
            video_fps=float(fps),
            mask_video=True,
            mask_audio=True,
            mask_init_value_video=0.0,
            mask_init_value_audio=0.0,
            slope_len=3,
        )
        log_node(f"  AV masking applied: ref=[0, {ref_end_time}s], extend=[{ref_end_time}, {total_time}s]", color="GREEN")

    # --- 9. I2V conditioning — inject last frame ---
    from comfy_extras.nodes_lt import LTXVImgToVideoInplace
    last_frame_img = last_frames[-1:].clone()  # [1, H, W, C]

    if av_latent["samples"].is_nested:
        latents = av_latent["samples"].unbind()
        temp_video_dict = {"samples": latents[0]}
        if "noise_mask" in av_latent and av_latent["noise_mask"].is_nested:
            masks = av_latent["noise_mask"].unbind()
            temp_video_dict["noise_mask"] = masks[0]
        temp_video_dict = LTXVImgToVideoInplace.execute(
            vae=vae, image=last_frame_img, latent=temp_video_dict, strength=1.0
        )
        if hasattr(temp_video_dict, 'args'):
            temp_video_dict = temp_video_dict.args[0]
        av_latent["samples"] = comfy.nested_tensor.NestedTensor(
            (temp_video_dict["samples"], latents[1])
        )
        if "noise_mask" in temp_video_dict:
            audio_nm = audio_latent_dict.get("noise_mask", torch.ones_like(latents[1]))
            av_latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (temp_video_dict["noise_mask"], audio_nm)
            )
    else:
        av_latent = LTXVImgToVideoInplace.execute(
            vae=vae, image=last_frame_img, latent=av_latent, strength=1.0
        )
        if hasattr(av_latent, 'args'):
            av_latent = av_latent.args[0]

    log_node("  I2V: last reference frame injected", color="GREEN")

    # --- 10. Pass 1 — Low-res sampling ---
    sigmas_p1 = _parse_sigmas(sigmas_preset, "pass1", custom_sigmas)
    log_node(f"  Pass 1: {len(sigmas_p1)-1} steps, resolution {pass1_w}x{pass1_h}", color="CYAN")

    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive=positive_cond, negative=negative_cond)
    guider.set_cfg(1.0)

    sampler = comfy.samplers.sampler_object("euler")
    noise = comfy.samplers.prepare_noise(av_latent["samples"], seed)

    sampled = guider.sample(
        noise, av_latent["samples"], sampler, sigmas_p1,
        denoise_mask=av_latent.get("noise_mask", None),
        disable_pbar=False
    )

    sampled_latent = {"samples": sampled}
    if "noise_mask" in av_latent:
        sampled_latent["noise_mask"] = av_latent["noise_mask"]

    log_node("  Pass 1: ✅ Complete", color="GREEN")

    # --- 11. Separate AV → Upscale → Re-combine → Pass 2 ---
    if has_upscaler and sampled.is_nested:
        latents = sampled.unbind()
        video_sampled = {"samples": latents[0]}
        audio_sampled = {"samples": latents[1]}

        from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
        upsampler = LTXVLatentUpsampler()
        video_upscaled = upsampler.upsample_latent(
            samples=video_sampled, upscale_model=latent_upscale_model, vae=vae
        )[0]
        log_node(f"  Upscale: {latents[0].shape} → {video_upscaled['samples'].shape}", color="GREEN")

        from comfy_extras.nodes_lt import LTXVCropGuides
        crop_result = LTXVCropGuides.execute(positive_cond, negative_cond, video_upscaled)
        if hasattr(crop_result, 'args'):
            pos_p2, neg_p2, video_upscaled = crop_result.args[0], crop_result.args[1], crop_result.args[2]
        else:
            pos_p2, neg_p2 = positive_cond, negative_cond

        av_upscaled_samples = comfy.nested_tensor.NestedTensor(
            (video_upscaled["samples"], audio_sampled["samples"])
        )
        av_upscaled = {"samples": av_upscaled_samples}

        sigmas_p2 = _parse_sigmas(sigmas_preset, "pass2", "")
        log_node(f"  Pass 2: {len(sigmas_p2)-1} steps, resolution {width}x{height}", color="CYAN")

        guider2 = comfy.samplers.CFGGuider(model)
        guider2.set_conds(positive=pos_p2, negative=neg_p2)
        guider2.set_cfg(1.0)

        noise2 = comfy.samplers.prepare_noise(av_upscaled_samples, seed + 1)
        sampled2 = guider2.sample(
            noise2, av_upscaled_samples, sampler, sigmas_p2,
            denoise_mask=av_upscaled.get("noise_mask", None),
            disable_pbar=False
        )
        final_latent = {"samples": sampled2}
        log_node("  Pass 2: ✅ Complete", color="GREEN")
    else:
        final_latent = sampled_latent

    # --- 12. Decode ---
    if final_latent["samples"].is_nested:
        all_latents = final_latent["samples"].unbind()
        video_lat = all_latents[0]
        audio_lat = all_latents[1] if len(all_latents) > 1 else None
    else:
        video_lat = final_latent["samples"]
        audio_lat = None

    log_node("  Decoding video (spatio-temporal tiled)...", color="CYAN")
    new_frames = ltx_spatio_temporal_tiled_decode(
        vae, video_lat,
        spatial_tiles=4, spatial_overlap=1,
        temporal_tile_length=16, temporal_overlap=1
    )
    log_node(f"  Decoded: {new_frames.shape[0]} frames (ref + extension)", color="GREEN")

    # Audio decode
    new_audio = None
    if audio_lat is not None and audio_vae is not None:
        log_node("  Decoding audio...", color="CYAN")
        audio_decoded = audio_vae.decode(audio_lat).movedim(-1, 1).to(audio_lat.device)
        output_sample_rate = audio_vae.first_stage_model.output_sample_rate
        new_audio = {
            "waveform": audio_decoded,
            "sample_rate": int(output_sample_rate),
        }
        log_node(f"  Audio decoded: {audio_decoded.shape}, {output_sample_rate}Hz", color="GREEN")

    # --- 13. Concatenate with original ---
    extension_only = new_frames[ref_frames:]
    all_frames = torch.cat([source_frames, extension_only], dim=0)
    log_node(f"  Concatenated: {source_frames.shape[0]} + {extension_only.shape[0]} = {all_frames.shape[0]} frames", color="GREEN")

    # Concatenate audio
    final_audio = None
    if video_pipe.audio is not None and new_audio is not None and extend_audio:
        src_waveform = video_pipe.audio["waveform"]
        new_waveform = new_audio["waveform"]
        sample_rate = video_pipe.audio["sample_rate"]

        ref_audio_samples = int(reference_seconds * sample_rate)
        if new_waveform.shape[-1] > ref_audio_samples:
            ext_audio = new_waveform[..., ref_audio_samples:]
            combined_waveform = torch.cat([src_waveform, ext_audio], dim=-1)
        else:
            combined_waveform = src_waveform

        final_audio = {
            "waveform": combined_waveform,
            "sample_rate": sample_rate,
        }
        log_node(f"  Audio concatenated: {combined_waveform.shape[-1]} samples", color="GREEN")
    elif video_pipe.audio is not None:
        final_audio = video_pipe.audio  # Passthrough original audio

    # --- 14. Build output context ---
    total_duration = all_frames.shape[0] / fps

    ctx = VideoGenerationContext()
    ctx.model = model
    ctx.clip = clip
    ctx.vae = vae
    ctx.model_name = model_name
    ctx.width = width
    ctx.height = height
    ctx.duration = total_duration
    ctx.fps = fps
    ctx.frame_count = all_frames.shape[0]
    ctx.steps = len(sigmas_p1) - 1
    ctx.cfg = 1.0
    ctx.shift = 0.0
    ctx.sampler_name = "euler"
    ctx.scheduler = f"manual_sigmas_{sigmas_preset}"
    ctx.seed = seed
    ctx.denoise = 1.0
    ctx.loader_type = "ltx2"
    ctx.positive_prompt = pos_text
    ctx.negative_prompt = neg_text
    ctx.frames = all_frames
    ctx.loras = applied_loras
    ctx.source_image = video_pipe.source_image
    ctx.audio = final_audio
    ctx.audio_vae = audio_vae

    log_node(f"🔗 LTX Extender: ✅ Complete — {all_frames.shape[0]} frames ({total_duration:.1f}s)"
             f"{' + audio' if final_audio else ''}", color="GREEN")

    return (ctx,)
