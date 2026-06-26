"""
UmeAiRT Toolkit - WAN Video Extender
---------------------------------------
WAN-specific video extension logic. Takes an existing video from
UME_VIDEO_PIPELINE and extends it by generating new frames conditioned
on the last frame via VACE.

Called by video_extender.py orchestrator. Not a ComfyUI node itself.

Pipeline:
  Extract last frame → VACE conditioning → LoRA → ModelSamplingSD3 →
  CFGZeroStar → EasyCache → NAG → Sample → VAE Decode → ColorMatch →
  Trim overlap → Concatenate with original
"""

import torch
import nodes as comfy_nodes
import node_helpers
import comfy.samplers
import comfy.model_management
import comfy.utils
from .common import VideoGenerationContext, UmeBundle, UmeVideoSettings, log_node, validate_bundle
from .video_utils import patch_wan_model, apply_color_match
from typing import Optional, List, Tuple

# Module-level LoRA loader
_lora_loader = comfy_nodes.LoraLoader()


def extend_wan(video_pipe: VideoGenerationContext,
               model_bundle: UmeBundle,
               positive: str,
               video_settings: UmeVideoSettings,
               negative: str = None,
               loras: Optional[List[Tuple[str, float, float]]] = None):
    """Extend video by generating new frames from the last frame via VACE.

    Returns:
        Tuple of (VideoGenerationContext,)
    """
    validate_bundle(model_bundle, ["model", "clip", "vae"], context="Video Extender (WAN)")

    source_frames = video_pipe.frames
    if source_frames is None or source_frames.shape[0] == 0:
        raise ValueError("Video Extender: No frames in the video pipeline.")

    # --- Read settings (with override support from Lightning/Optimization) ---
    overrides = getattr(model_bundle, "overrides", None) or {}
    width = video_settings.width
    height = video_settings.height
    duration = video_settings.duration
    fps = 16  # WAN native framerate
    frame_count = int(duration * fps) + 1
    steps = overrides.get("steps", video_settings.steps)
    cfg = overrides.get("cfg", video_settings.cfg)
    shift = overrides.get("shift", video_settings.shift)
    sampler_name = overrides.get("sampler_name", video_settings.sampler_name)
    scheduler = overrides.get("scheduler", video_settings.scheduler)
    seed = video_settings.seed
    denoise = overrides.get("denoise", 1.0)

    model = model_bundle.model
    clip = model_bundle.clip
    vae = model_bundle.vae
    model_name = getattr(model_bundle, "model_name", "")
    pos_text = positive if isinstance(positive, str) else ""
    neg_text = negative if isinstance(negative, str) else ""

    log_node(f"🔗 Video Extender: {source_frames.shape[0]} frames → +{frame_count} frames | "
             f"{width}x{height} | {steps} steps | CFG {cfg} | Shift {shift}", color="CYAN")

    # --- 1. Apply LoRAs ---
    applied_loras = list(video_pipe.loras) if video_pipe.loras else []
    if loras:
        for lora_entry in loras:
            if len(lora_entry) == 4:
                lora_name, strength_model, strength_clip, _ = lora_entry
            else:
                lora_name, strength_model, strength_clip = lora_entry
            try:
                model, clip = _lora_loader.load_lora(
                    model, clip, lora_name, strength_model, strength_clip
                )
                applied_loras.append((lora_name, strength_model))
                log_node(f"  LoRA applied: {lora_name} (str={strength_model:.2f})", color="GREEN")
            except Exception as e:
                log_node(f"  LoRA failed: {lora_name}: {e}", color="RED")

    # --- 2. Patch model: ModelSamplingSD3 + CFGZeroStar + EasyCache + NAG ---
    model = patch_wan_model(model, shift, overrides)

    # --- 3. Extract last frame as VACE start ---
    last_frame = source_frames[-1:].clone()  # [1, H, W, C]
    log_node(f"  Reference: last frame extracted ({last_frame.shape})", color="GREEN")

    # --- 4. Encode prompts ---
    tokens_pos = clip.tokenize(pos_text)
    cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
    positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]

    tokens_neg = clip.tokenize(neg_text)
    cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
    negative_cond = [[cond_neg, {"pooled_output": pooled_neg}]]

    # --- 5. Build VACE conditioning (last_frame as start, free continuation) ---
    latent_length = ((frame_count - 1) // 4) + 1

    start_resized = comfy.utils.common_upscale(
        last_frame.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)

    # Control video: first frame = last_frame, rest = gray (0.5)
    control_video = torch.ones(
        (frame_count, height, width, 3),
        device=start_resized.device, dtype=start_resized.dtype
    ) * 0.5
    control_video[0] = start_resized[0, :, :, :3]

    # Mask: 1.0 = generate, 0.0 = known
    mask = torch.ones(
        (frame_count, height, width, 1),
        device=start_resized.device, dtype=start_resized.dtype
    )
    mask[0] = 0.0  # First frame is known

    log_node("  VACE: Start frame conditioned (free continuation)", color="GREEN")

    # Native WanVaceToVideo conditioning logic
    control_video_centered = control_video - 0.5
    inactive = (control_video_centered * (1 - mask)) + 0.5
    reactive = (control_video_centered * mask) + 0.5

    inactive_latent = vae.encode(inactive[:, :, :, :3])
    reactive_latent = vae.encode(reactive[:, :, :, :3])
    control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)

    # Pixelize mask via 8×8 block downsampling
    vae_stride = 8
    height_mask = height // vae_stride
    width_mask = width // vae_stride
    mask_down = mask.view(frame_count, height_mask, vae_stride, width_mask, vae_stride)
    mask_down = mask_down.permute(2, 4, 0, 1, 3)
    mask_down = mask_down.reshape(vae_stride * vae_stride, frame_count, height_mask, width_mask)
    mask_down = torch.nn.functional.interpolate(
        mask_down.unsqueeze(0),
        size=(latent_length, height_mask, width_mask),
        mode='nearest-exact'
    ).squeeze(0)
    mask_down = mask_down.unsqueeze(0)

    strength = 1.0
    positive_cond = node_helpers.conditioning_set_values(
        positive_cond,
        {"vace_frames": [control_video_latent], "vace_mask": [mask_down], "vace_strength": [strength]},
        append=True
    )
    negative_cond = node_helpers.conditioning_set_values(
        negative_cond,
        {"vace_frames": [control_video_latent], "vace_mask": [mask_down], "vace_strength": [strength]},
        append=True
    )

    # --- 6. Create empty latent ---
    latent = torch.zeros(
        [1, 16, latent_length, height // 8, width // 8],
        device=comfy.model_management.intermediate_device()
    )
    out_latent = {"samples": latent}
    log_node(f"  Latent prepared: {latent.shape}", color="GREEN")

    # --- 7. Sample ---
    from comfy_extras.nodes_custom_sampler import (
        CFGGuider, RandomNoise, BasicScheduler, KSamplerSelect, SamplerCustomAdvanced
    )

    noise = RandomNoise.get_noise(seed)[0]
    sampler_obj = KSamplerSelect.get_sampler(sampler_name)[0]
    sigmas = BasicScheduler.get_sigmas(model, scheduler, steps, denoise)[0]
    guider = CFGGuider.get_guider(model, positive_cond, negative_cond, cfg)[0]

    log_node(f"  Sampling: {sampler_name}/{scheduler}, {steps} steps, seed={seed}...", color="CYAN")
    _, denoised_output = SamplerCustomAdvanced.sample(noise, guider, sampler_obj, sigmas, out_latent)

    # --- 8. VAE Decode ---
    log_node("  VAE Decode...", color="CYAN")
    new_frames = vae.decode(denoised_output["samples"])
    if len(new_frames.shape) == 5:
        new_frames = new_frames.reshape(-1, new_frames.shape[-3], new_frames.shape[-2], new_frames.shape[-1])
    log_node(f"  Decoded: {new_frames.shape[0]} frames", color="GREEN")

    # --- 9. ColorMatch (Reinhard transfer from last source frame) ---
    new_frames = apply_color_match(new_frames, last_frame, width, height)

    # --- 10. Trim overlap and concatenate ---
    # The first frame of new_frames is the conditioned start frame (= last frame of source)
    # so we skip it to avoid duplication
    extension_only = new_frames[1:]
    all_frames = torch.cat([source_frames, extension_only], dim=0)
    total_duration = all_frames.shape[0] / fps

    log_node(f"  Concatenated: {source_frames.shape[0]} + {extension_only.shape[0]} = "
             f"{all_frames.shape[0]} frames ({total_duration:.1f}s)", color="GREEN")

    # --- 11. Build output context ---
    ctx = VideoGenerationContext()
    ctx.model = model_bundle.model  # Original (unpatched) for metadata
    ctx.clip = model_bundle.clip
    ctx.vae = model_bundle.vae
    ctx.clip_vision = getattr(model_bundle, "clip_vision", None)
    ctx.model_name = model_name
    ctx.width = width
    ctx.height = height
    ctx.duration = total_duration
    ctx.fps = fps
    ctx.frame_count = all_frames.shape[0]
    ctx.steps = steps
    ctx.cfg = cfg
    ctx.shift = shift
    ctx.sampler_name = sampler_name
    ctx.scheduler = scheduler
    ctx.seed = seed
    ctx.denoise = denoise
    ctx.loader_type = getattr(model_bundle, "loader_type", "wan")
    ctx.positive_prompt = pos_text
    ctx.negative_prompt = neg_text
    ctx.frames = all_frames
    ctx.source_image = video_pipe.source_image
    ctx.loras = applied_loras

    log_node(f"🔗 Video Extender: ✅ Complete — {all_frames.shape[0]} frames ({total_duration:.1f}s)", color="GREEN")

    return (ctx,)
