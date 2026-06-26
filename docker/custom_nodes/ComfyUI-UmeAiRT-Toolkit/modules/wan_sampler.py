"""
UmeAiRT Toolkit - WAN Video Sampler
--------------------------------------
WAN-specific video generation logic (T2V, I2V, VACE, FunControl, MoE).

Called by video_sampler.py orchestrator. Not a ComfyUI node itself.

Pipeline: LoRA → ModelSamplingSD3 → CFGZeroStar → EasyCache → NAG →
          CLIP Encode → WanImageToVideo/WanVaceToVideo/WanFunControlToVideo →
          SamplerCustomAdvanced → VAE Decode
"""

import torch
import nodes as comfy_nodes
import node_helpers
import comfy.samplers
import comfy.model_management
import comfy.utils
import comfy.clip_vision
from .common import VideoGenerationContext, UmeBundle, UmeVideoSettings, UmeVaceFrames, UmeFunControl, log_node, validate_bundle
from .video_utils import patch_wan_model, apply_color_match
from typing import Optional, List, Tuple

# Module-level LoRA loader (shared across calls)
_lora_loader = comfy_nodes.LoraLoader()


def generate_wan(model_bundle: UmeBundle,
                 positive: str,
                 video_settings: UmeVideoSettings,
                 negative: str = None,
                 loras: Optional[List[Tuple[str, float, float]]] = None,
                 source_image=None,
                 vace_frames: Optional[UmeVaceFrames] = None,
                 funcontrol: Optional[UmeFunControl] = None):
    """Orchestrate the full WAN video generation pipeline.

    Returns:
        Tuple of (VideoGenerationContext,)
    """
    validate_bundle(model_bundle, ["model", "clip", "vae"], context="Video Generator (WAN)")

    # Read settings (with override support from Lightning/Optimization nodes)
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
    clip_vision = getattr(model_bundle, "clip_vision", None)
    model_name = getattr(model_bundle, "model_name", "")
    pos_text = positive if isinstance(positive, str) else ""
    neg_text = negative if isinstance(negative, str) else ""

    # Mode detection: FunControl > FLF2V > VACE > I2V > T2V
    is_funcontrol = funcontrol is not None
    is_flf2v = (vace_frames is not None and clip_vision is not None 
                and not is_funcontrol)  # FLF2V: I2V model + start/end frames
    is_vace = (vace_frames is not None and clip_vision is None 
               and not is_funcontrol)   # VACE: T2V model + start/end frames
    is_i2v = (source_image is not None and clip_vision is not None 
              and not is_vace and not is_flf2v and not is_funcontrol)

    # Prevent user from using an I2V model for pure T2V without an image
    if clip_vision is not None and source_image is None and not is_vace and not is_flf2v and not is_funcontrol:
        raise ValueError("You have loaded an Image-to-Video (I2V) model, but no 'source_image' is connected! Please connect an image or switch the loader to a Text-to-Video (T2V) model.")

    mode_str = ("FunControl" if is_funcontrol else 
                ("FLF2V" if is_flf2v else
                 ("VACE" if is_vace else ("I2V" if is_i2v else "T2V"))))
    log_node(f"🎬 Video Generator: {mode_str} mode | {width}x{height} | "
             f"{duration}s ({frame_count} frames) | {steps} steps | CFG {cfg} | Shift {shift}", color="CYAN")

    # --- 1. Apply LoRAs (with High/Low noise targeting for MoE) ---
    applied_loras = []
    if loras:
        for lora_entry in loras:
            # Support both 3-tuple (legacy) and 4-tuple (WAN LoRA Block) formats
            if len(lora_entry) == 4:
                lora_name, strength_model, strength_clip, target = lora_entry
            else:
                lora_name, strength_model, strength_clip = lora_entry
                target = "both"

            # Apply to high-noise model (default model) if target is "both" or "high"
            if target in ("both", "high"):
                try:
                    model, clip = _lora_loader.load_lora(
                        model, clip, lora_name, strength_model, strength_clip
                    )
                    target_label = f" [{target}]" if target != "both" else ""
                    applied_loras.append((lora_name, strength_model))
                    log_node(f"  LoRA applied{target_label}: {lora_name} (str={strength_model:.2f})", color="GREEN")
                except Exception as e:
                    log_node(f"  LoRA failed: {lora_name}: {e}", color="RED")
            elif target == "low":
                # Skip high-noise application; will be applied to low-noise model below
                applied_loras.append((lora_name, strength_model))
                log_node(f"  LoRA deferred to Low-Noise: {lora_name} (str={strength_model:.2f})", color="CYAN")

    # --- 2. Patch model(s): ModelSamplingSD3 + CFGZeroStar + EasyCache + NAG ---
    model = patch_wan_model(model, shift, overrides, label="High-Noise" if model_bundle.is_moe else "")
    model_low = None
    if model_bundle.is_moe:
        model_low = model_bundle.model_low_noise
        # Apply LoRAs to low-noise model (only those targeting "both" or "low")
        if loras:
            for lora_entry in loras:
                if len(lora_entry) == 4:
                    lora_name, strength_model, strength_clip, target = lora_entry
                else:
                    lora_name, strength_model, strength_clip = lora_entry
                    target = "both"

                if target in ("both", "low"):
                    try:
                        model_low, _ = _lora_loader.load_lora(
                            model_low, clip, lora_name, strength_model, strength_clip
                        )
                        target_label = f" [{target}]" if target != "both" else ""
                        log_node(f"  LoRA applied to Low-Noise{target_label}: {lora_name} (str={strength_model:.2f})", color="GREEN")
                    except Exception as e:
                        log_node(f"  LoRA skipped on Low-Noise model: {lora_name}: {e}", color="YELLOW")
        model_low = patch_wan_model(model_low, shift, overrides, label="Low-Noise")
        log_node("  MoE: Dual-model pipeline active (High + Low noise experts)", color="CYAN")

    # --- 3. Encode prompts via CLIP ---
    tokens_pos = clip.tokenize(pos_text)
    cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
    positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]

    tokens_neg = clip.tokenize(neg_text)
    cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
    negative_cond = [[cond_neg, {"pooled_output": pooled_neg}]]

    # --- 4. Encode CLIP Vision (if I2V, FLF2V, or FunControl mode) ---
    clip_vision_output = None
    if is_funcontrol and clip_vision is not None and funcontrol.source_image is not None:
        clip_vision_output = clip_vision.encode_image(funcontrol.source_image)
        log_node("  CLIP Vision: encoded FunControl source image", color="GREEN")
    elif is_flf2v and clip_vision is not None:
        clip_vision_start = clip_vision.encode_image(vace_frames.start_image)
        log_node("  CLIP Vision: encoded FLF2V start image", color="GREEN")
        if vace_frames.end_image is not None:
            clip_vision_end = clip_vision.encode_image(vace_frames.end_image)
            log_node("  CLIP Vision: encoded FLF2V end image", color="GREEN")
            states = torch.cat([
                clip_vision_start.penultimate_hidden_states,
                clip_vision_end.penultimate_hidden_states
            ], dim=-2)
            clip_vision_output = comfy.clip_vision.Output()
            clip_vision_output.penultimate_hidden_states = states
        else:
            clip_vision_output = clip_vision_start
    elif is_i2v and clip_vision is not None:
        clip_vision_output = clip_vision.encode_image(source_image)
        log_node("  CLIP Vision: encoded source image", color="GREEN")

    # --- 5. Create video latent + conditioning ---
    latent = torch.zeros(
        [1, 16, ((frame_count - 1) // 4) + 1, height // 8, width // 8],
        device=comfy.model_management.intermediate_device()
    )

    if is_funcontrol:
        # === FunControl Conditioning (WanFunControlToVideo) ===
        positive_cond, negative_cond = _build_funcontrol_conditioning(
            funcontrol, positive_cond, negative_cond,
            vae, clip_vision_output, width, height, frame_count
        )
    elif is_vace:
        # === VACE Conditioning (Start+End frames) ===
        positive_cond, negative_cond = _build_vace_conditioning(
            vace_frames, positive_cond, negative_cond,
            vae, width, height, frame_count
        )
    elif is_flf2v:
        # === FLF2V Conditioning (Start+End frames for I2V model) ===
        positive_cond, negative_cond = _build_flf2v_conditioning(
            vace_frames, positive_cond, negative_cond,
            vae, width, height, frame_count, latent
        )
    elif source_image is not None:
        # === Standard I2V Conditioning (WanImageToVideo) ===
        start_img = comfy.utils.common_upscale(
            source_image[:frame_count].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        image = torch.ones(
            (frame_count, height, width, start_img.shape[-1]),
            device=start_img.device, dtype=start_img.dtype
        ) * 0.5
        image[:start_img.shape[0]] = start_img

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = torch.ones(
            (1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
            device=start_img.device, dtype=start_img.dtype
        )
        mask[:, :, :((start_img.shape[0] - 1) // 4) + 1] = 0.0

        positive_cond = node_helpers.conditioning_set_values(
            positive_cond, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative_cond = node_helpers.conditioning_set_values(
            negative_cond, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

    if clip_vision_output is not None:
        positive_cond = node_helpers.conditioning_set_values(
            positive_cond, {"clip_vision_output": clip_vision_output}
        )
        negative_cond = node_helpers.conditioning_set_values(
            negative_cond, {"clip_vision_output": clip_vision_output}
        )

    out_latent = {"samples": latent}
    log_node(f"  Latent prepared: {latent.shape}", color="GREEN")

    # --- 6. Sample ---
    from comfy_extras.nodes_custom_sampler import (
        CFGGuider, RandomNoise, BasicScheduler, KSamplerSelect, SamplerCustomAdvanced
    )

    noise = RandomNoise.get_noise(seed)[0]
    sampler_obj = KSamplerSelect.get_sampler(sampler_name)[0]

    if model_bundle.is_moe and model_low is not None:
        # === MoE Dual-Pass Sampling (WAN 2.2 14B) ===
        split_step = steps // 2
        sigmas = BasicScheduler.get_sigmas(model, scheduler, steps, denoise)[0]
        high_sigmas = sigmas[:split_step + 1]
        low_sigmas = sigmas[split_step:]

        # Pass 1: High-Noise Expert
        guider_high = CFGGuider.get_guider(model, positive_cond, negative_cond, cfg)[0]
        log_node(f"  MoE Pass 1/2 (High-Noise): {sampler_name}/{scheduler}, "
                 f"steps 0→{split_step} of {steps}, seed={seed}...", color="CYAN")
        output_high, _ = SamplerCustomAdvanced.sample(noise, guider_high, sampler_obj, high_sigmas, out_latent)

        # Pass 2: Low-Noise Expert (no noise, continues from pass 1 latent)
        from comfy_extras.nodes_custom_sampler import DisableNoise
        no_noise = DisableNoise.get_noise()[0]
        guider_low = CFGGuider.get_guider(model_low, positive_cond, negative_cond, cfg)[0]
        log_node(f"  MoE Pass 2/2 (Low-Noise): {sampler_name}/{scheduler}, "
                 f"steps {split_step}→{steps} of {steps}...", color="CYAN")
        _, denoised_output = SamplerCustomAdvanced.sample(no_noise, guider_low, sampler_obj, low_sigmas, output_high)
    else:
        # === Single-Pass Sampling (WAN 2.1, WAN 2.2 5B) ===
        sigmas = BasicScheduler.get_sigmas(model, scheduler, steps, denoise)[0]
        guider = CFGGuider.get_guider(model, positive_cond, negative_cond, cfg)[0]
        log_node(f"  Sampling: {sampler_name}/{scheduler}, {steps} steps, seed={seed}...", color="CYAN")
        output, denoised_output = SamplerCustomAdvanced.sample(noise, guider, sampler_obj, sigmas, out_latent)

    # --- 7. VAE Decode ---
    log_node("  VAE Decode...", color="CYAN")
    frames = vae.decode(denoised_output["samples"])
    if len(frames.shape) == 5:
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    log_node(f"  ✅ Video generated: {frames.shape[0]} frames", color="GREEN")

    # --- 8. Color Match (VACE / FLF2V / FunControl) ---
    if (is_vace or is_flf2v) and vace_frames.color_match and vace_frames.start_image is not None:
        frames = apply_color_match(frames, vace_frames.start_image, width, height)
    elif is_funcontrol and funcontrol.source_image is not None:
        frames = apply_color_match(frames, funcontrol.source_image, width, height)

    # --- 9. Build VideoGenerationContext ---
    ctx = VideoGenerationContext()
    ctx.model = model_bundle.model  # Original (unpatched) for metadata
    ctx.clip = model_bundle.clip
    ctx.vae = model_bundle.vae
    ctx.clip_vision = clip_vision
    ctx.model_name = model_name
    ctx.width = width
    ctx.height = height
    ctx.duration = duration
    ctx.fps = fps
    ctx.frame_count = frame_count
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
    ctx.frames = frames
    ctx.source_image = (
        funcontrol.source_image if is_funcontrol else 
        (vace_frames.start_image if (is_vace or is_flf2v) else source_image)
    )
    ctx.loras = applied_loras

    return (ctx,)


# ---------------------------------------------------------------------------
# Internal conditioning builders
# ---------------------------------------------------------------------------

def _build_flf2v_conditioning(vace_frames, positive_cond, negative_cond,
                              vae, width, height, frame_count, latent):
    """Build FLF2V conditioning (I2V model with start+end frame anchoring).

    Replicates the ComfyUI WanFirstLastFrameToVideo logic:
    1. Build image tensor: start frame at [0], end frame at [-1], gray fill between
    2. VAE-encode to create concat_latent_image
    3. Build mask: 0.0 for known frames, 1.0 for to-generate frames
    4. Reshape mask to [1, 4, T//4, H, W] (ComfyUI FLF2V convention)
    """
    start_img = vace_frames.start_image
    end_img = vace_frames.end_image

    # 1. Upscale start and end images to target dimensions
    start_resized = None
    if start_img is not None:
        start_resized = comfy.utils.common_upscale(
            start_img[:frame_count].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

    end_resized = None
    if end_img is not None:
        end_resized = comfy.utils.common_upscale(
            end_img[-frame_count:].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

    # Determine device and dtype from resized image or fallback
    ref_tensor = start_resized if start_resized is not None else end_resized
    device = ref_tensor.device if ref_tensor is not None else comfy.model_management.intermediate_device()
    dtype = ref_tensor.dtype if ref_tensor is not None else torch.float32

    # 2. Build composite image tensor [frame_count, H, W, 3] filled with gray (0.5)
    image = torch.ones(
        (frame_count, height, width, 3),
        device=device, dtype=dtype
    ) * 0.5

    # 3. Build mask [1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]] filled with 1.0
    mask = torch.ones(
        (1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]),
        device=device, dtype=dtype
    )

    if start_resized is not None:
        image[:start_resized.shape[0]] = start_resized[:, :, :, :3]
        mask[:, :, :start_resized.shape[0] + 3] = 0.0
        log_node("  FLF2V: Start frame conditioned", color="GREEN")

    if end_resized is not None:
        image[-end_resized.shape[0]:] = end_resized[:, :, :, :3]
        mask[:, :, -end_resized.shape[0]:] = 0.0
        log_node("  FLF2V: End frame conditioned", color="GREEN")

    # 4. VAE encode the composite image
    concat_latent_image = vae.encode(image)

    # 5. Reshape mask to [1, 4, latent_T, latent_H, latent_W]
    mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

    # 6. Apply to conditioning
    positive_cond = node_helpers.conditioning_set_values(
        positive_cond, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
    )
    negative_cond = node_helpers.conditioning_set_values(
        negative_cond, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
    )

    return positive_cond, negative_cond


def _build_vace_conditioning(vace_frames, positive_cond, negative_cond,
                             vae, width, height, frame_count):
    """Build VACE conditioning using the native WanVaceToVideo approach.

    Replicates the ComfyUI WanVaceToVideo conditioning pipeline exactly:
    1. Build control_video (known frames + gray fill) and per-pixel masks
    2. Split control_video into inactive/reactive halves via the mask
    3. VAE-encode both halves separately → 32-channel latent
    4. Pixelize mask via 8×8 block downsampling → latent resolution
    5. Condition via vace_frames/vace_mask/vace_strength keys
    """
    start_img = vace_frames.start_image
    end_img = vace_frames.end_image
    latent_length = ((frame_count - 1) // 4) + 1

    # Resize start image to target dimensions
    start_resized = comfy.utils.common_upscale(
        start_img[:1].movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)

    # Build control_video: [frame_count, H, W, 3] — gray (0.5) for unknown frames
    control_video = torch.ones(
        (frame_count, height, width, 3),
        device=start_resized.device, dtype=start_resized.dtype
    ) * 0.5
    control_video[0] = start_resized[0, :, :, :3]

    # Build per-pixel mask: [frame_count, H, W, 1] — 1.0 = generate, 0.0 = known
    mask = torch.ones(
        (frame_count, height, width, 1),
        device=start_resized.device, dtype=start_resized.dtype
    )
    mask[0] = 0.0  # First frame is known

    if end_img is not None:
        end_resized = comfy.utils.common_upscale(
            end_img[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        control_video[-1] = end_resized[0, :, :, :3]
        mask[-1] = 0.0  # Last frame is known
        log_node("  VACE: Start+End frames conditioned", color="GREEN")
    else:
        log_node("  VACE: Start frame conditioned (free continuation)", color="GREEN")

    # === Native WanVaceToVideo logic ===

    # Split control_video into inactive (masked-out regions) and reactive (unmasked)
    control_video_centered = control_video - 0.5
    inactive = (control_video_centered * (1 - mask)) + 0.5
    reactive = (control_video_centered * mask) + 0.5

    # VAE-encode both halves separately, then concatenate → 32 channels
    inactive_latent = vae.encode(inactive[:, :, :, :3])
    reactive_latent = vae.encode(reactive[:, :, :, :3])
    control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)

    # Pixelize mask via 8×8 block downsampling → latent resolution
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
    mask_down = mask_down.unsqueeze(0)  # [1, 64, T_latent, H_latent, W_latent]

    # Apply VACE-specific conditioning keys (NOT concat_latent_image!)
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

    return positive_cond, negative_cond


def _build_funcontrol_conditioning(funcontrol, positive_cond, negative_cond,
                                   vae, clip_vision_output, width, height, frame_count):
    """Build FunControl conditioning using WanFunControlToVideo approach.

    Replicates the ComfyUI WanFunControlToVideo pipeline:
    1. Resize source image + control video to target dimensions
    2. VAE-encode the composite image sequence (source frame + control frames)
    3. Build concat_latent_image + mask for conditioning
    4. Inject CLIP Vision output for I2V-style image guidance
    """
    source_img = funcontrol.source_image
    control_video = funcontrol.control_video
    strength = funcontrol.strength

    # Resize source image
    src_resized = comfy.utils.common_upscale(
        source_img[:1].movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)

    # Resize control video frames
    ctrl_resized = comfy.utils.common_upscale(
        control_video[:frame_count].movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)

    # Build the composite image sequence:
    # First frame = source image, rest = control frames (or gray fill if not enough)
    image = torch.ones(
        (frame_count, height, width, src_resized.shape[-1]),
        device=src_resized.device, dtype=src_resized.dtype
    ) * 0.5
    image[0] = src_resized[0, :, :, :3]

    # Build control_hint from the control video
    control_hint = torch.ones(
        (frame_count, height, width, 3),
        device=ctrl_resized.device, dtype=ctrl_resized.dtype
    ) * 0.5
    n_ctrl = min(ctrl_resized.shape[0], frame_count)
    control_hint[:n_ctrl] = ctrl_resized[:n_ctrl, :, :, :3]

    # Encode the source image sequence (like WanImageToVideo)
    concat_latent_image = vae.encode(image[:, :, :, :3])

    # Build mask: first frame is known (0.0), rest is to generate (1.0)
    latent_length = ((frame_count - 1) // 4) + 1
    mask = torch.ones(
        (1, 1, latent_length, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
        device=src_resized.device, dtype=src_resized.dtype
    )
    mask[:, :, :1] = 0.0

    # Encode control video as conditioning hint
    control_latent = vae.encode(control_hint[:, :, :, :3])

    # Apply FunControl conditioning: concat_latent_image + control hint
    positive_cond = node_helpers.conditioning_set_values(
        positive_cond,
        {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask,
            "control_video": control_latent,
            "control_strength": strength,
        }
    )
    negative_cond = node_helpers.conditioning_set_values(
        negative_cond,
        {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask,
            "control_video": control_latent,
            "control_strength": strength,
        }
    )

    # Inject CLIP Vision output for image guidance
    if clip_vision_output is not None:
        positive_cond = node_helpers.conditioning_set_values(
            positive_cond, {"clip_vision_output": clip_vision_output}
        )
        negative_cond = node_helpers.conditioning_set_values(
            negative_cond, {"clip_vision_output": clip_vision_output}
        )

    log_node(f"  FunControl: {n_ctrl} control frames encoded (strength={strength:.2f})", color="GREEN")
    return positive_cond, negative_cond
