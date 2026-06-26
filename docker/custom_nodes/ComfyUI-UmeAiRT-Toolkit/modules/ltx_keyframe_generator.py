"""
UmeAiRT Toolkit - LTX Keyframe Generator
-------------------------------------------
Enhanced LTX Video Generator that accepts keyframe images at specific
frame positions to guide generation.

Auto-detects mode:
- If middle_frame is connected → 3 keyframes (first, middle, last)
- If only first + last → 2 keyframes (first, last)

Uses LTXVBaseSampler's optional_cond_images + optional_cond_indices.
The BaseSampler handles latent creation, I2V conditioning, and sampling
internally — we just pass keyframe images and indices.
"""

import torch
import nodes as comfy_nodes
import node_helpers
import comfy.samplers
import comfy.model_management
import comfy.utils
from .common import VideoGenerationContext, UmeBundle, UmeVideoSettings, log_node, validate_bundle
from .ltx_utils import ltx_spatio_temporal_tiled_decode
from .ltx_sampler import _parse_sigmas
from typing import Optional, List, Tuple


class UmeAiRT_LTXKeyframeGenerator:
    """LTX Keyframe Generator — video generation guided by keyframe images.

    Accepts 2 or 3 keyframe images (first + last, or first + middle + last)
    and generates video that transitions through them.

    Auto-detects mode:
    - middle_frame connected → 3-keyframe mode (first, midpoint, last)
    - middle_frame not connected → 2-keyframe mode (first, last)

    Delegates all latent creation and sampling to LTXVBaseSampler.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "From LTX Loader or Bundle Auto-Loader."}),
                "positive": ("POSITIVE", {"forceInput": True, "tooltip": "Describe the scene/action for the video."}),
                "video_settings": ("UME_VIDEO_SETTINGS", {"tooltip": "From LTX Video Settings node."}),
                "first_frame": ("IMAGE", {"tooltip": "First keyframe image (frame 0)."}),
                "last_frame": ("IMAGE", {"tooltip": "Last keyframe image (final frame)."}),
            },
            "optional": {
                "middle_frame": ("IMAGE", {"tooltip": "Middle keyframe image. If connected, enables 3-keyframe mode with auto-calculated midpoint."}),
                "negative": ("NEGATIVE", {"forceInput": True, "tooltip": "Describe what to avoid. Optional."}),
                "loras": ("UME_LORA_STACK", {"tooltip": "Connect a LoRA Block node."}),
                "cond_image_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Conditioning strength for keyframe images. Lower = more creative freedom.",
                    "advanced": True,
                }),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "generate"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Generates LTX video guided by keyframe images. Connect first + last frames (2-keyframe), or add a middle frame (3-keyframe)."

    def __init__(self):
        self._lora_loader = comfy_nodes.LoraLoader()

    def generate(self,
                 model_bundle: UmeBundle,
                 positive: str,
                 video_settings: UmeVideoSettings,
                 first_frame=None,
                 last_frame=None,
                 middle_frame=None,
                 negative: str = None,
                 loras: Optional[List[Tuple[str, float, float]]] = None,
                 cond_image_strength: float = 1.0):
        """Generate video guided by keyframe images."""

        validate_bundle(model_bundle, ["model", "clip", "vae"], context="LTX Keyframe Generator")

        if first_frame is None or last_frame is None:
            raise ValueError("Keyframe Generator: first_frame and last_frame are required.")

        # --- Read settings ---
        width = video_settings.width
        height = video_settings.height
        duration = video_settings.duration
        fps = video_settings.frame_rate
        seed = video_settings.seed
        sigmas_preset = video_settings.sigmas_preset or "standard"
        custom_sigmas = video_settings.custom_sigmas or ""

        model = model_bundle.model
        clip = model_bundle.clip
        vae = model_bundle.vae
        model_name = getattr(model_bundle, "model_name", "")

        pos_text = positive if isinstance(positive, str) else ""
        neg_text = negative if isinstance(negative, str) else ""

        # --- Frame count (8n+1 aligned) ---
        frame_count = int(duration * fps) + 1
        frame_count = ((frame_count - 1) // 8) * 8 + 1

        # --- Build keyframe images + indices ---
        has_middle = middle_frame is not None
        if has_middle:
            midpoint = (frame_count - 1) // 2
            cond_images = torch.cat([first_frame, middle_frame, last_frame], dim=0)
            cond_indices = f"0,{midpoint},{frame_count - 1}"
            mode_str = f"3-keyframe (0, {midpoint}, {frame_count - 1})"
        else:
            cond_images = torch.cat([first_frame, last_frame], dim=0)
            cond_indices = f"0,{frame_count - 1}"
            mode_str = f"2-keyframe (0, {frame_count - 1})"

        log_node(f"🎬 LTX Keyframe Generator: {mode_str} | {width}x{height} | "
                 f"{duration}s ({frame_count} frames @ {fps}fps) | "
                 f"Strength: {cond_image_strength} | Sigmas: {sigmas_preset}", color="CYAN")

        # --- 1. Apply LoRAs ---
        applied_loras = []
        if loras:
            for lora_name, strength_model, strength_clip in loras:
                try:
                    model, clip = self._lora_loader.load_lora(
                        model, clip, lora_name, strength_model, strength_clip
                    )
                    applied_loras.append((lora_name, strength_model))
                    log_node(f"  LoRA applied: {lora_name} (str={strength_model:.2f})", color="GREEN")
                except Exception as e:
                    log_node(f"  LoRA failed: {lora_name}: {e}", color="RED")

        # --- 2. Encode prompts ---
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

        # --- 3. Build guider ---
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive=positive_cond, negative=negative_cond)
        guider.set_cfg(1.0)

        # --- 4. Run BaseSampler with keyframe conditioning ---
        # BaseSampler handles: EmptyLatent creation, I2V (frame 0), guide injection,
        # and multi-step sampling (high/middle/low sigmas + CropGuides)
        sigmas = _parse_sigmas(sigmas_preset, "pass1", custom_sigmas)
        log_node(f"  BaseSampler: {len(sigmas)-1} steps, {width}x{height}, "
                 f"keyframes: {cond_indices}", color="CYAN")

        from vendor.ltxvideo.easy_samplers import LTXVBaseSampler

        sampler = comfy.samplers.sampler_object("euler")
        noise_gen = comfy.samplers.Noise_RandomNoise(seed)

        sampled_latent, pos_out, neg_out = LTXVBaseSampler().sample(
            model=model,
            vae=vae,
            noise=noise_gen,
            sampler=sampler,
            sigmas=sigmas,
            guider=guider,
            num_frames=frame_count,
            width=width,
            height=height,
            optional_cond_images=cond_images,
            optional_cond_indices=cond_indices,
            strength=cond_image_strength,
        )

        log_node("  BaseSampler: ✅ Complete", color="GREEN")

        # --- 5. Decode ---
        video_lat = sampled_latent["samples"]

        log_node("  Decoding video (spatio-temporal tiled)...", color="CYAN")
        frames = ltx_spatio_temporal_tiled_decode(
            vae, video_lat,
            spatial_tiles=4, spatial_overlap=1,
            temporal_tile_length=16, temporal_overlap=1,
        )
        log_node(f"  Video decoded: {frames.shape[0]} frames", color="GREEN")

        # --- 6. Build context ---
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
        ctx.steps = len(sigmas) - 1
        ctx.cfg = 1.0
        ctx.shift = 0.0
        ctx.sampler_name = "euler"
        ctx.scheduler = f"manual_sigmas_{sigmas_preset}"
        ctx.seed = seed
        ctx.denoise = 1.0
        ctx.loader_type = "ltx2"
        ctx.positive_prompt = pos_text
        ctx.negative_prompt = neg_text
        ctx.frames = frames
        ctx.loras = applied_loras
        ctx.source_image = first_frame
        ctx.audio = None  # BaseSampler is video-only; audio needs separate handling
        ctx.audio_vae = model_bundle.audio_vae

        log_node(f"🎬 LTX Keyframe Generator: ✅ {mode_str} — {frames.shape[0]} frames",
                 color="GREEN")

        return (ctx,)
