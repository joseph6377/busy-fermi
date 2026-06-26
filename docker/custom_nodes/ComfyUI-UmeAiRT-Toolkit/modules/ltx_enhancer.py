"""
UmeAiRT Toolkit - LTX Video Enhancer / Upscaler
--------------------------------------------------
Takes an existing video from UME_VIDEO_PIPELINE and enhances/upscales it
using the LTXVLoopingSampler with overlapping temporal chunks.

Pipeline:
  Scale frames → VAE Encode → LTXVLoopingSampler (guidance from encoded latents)
  → Tiled VAE Decode → Updated VideoGenerationContext
"""

import torch
import nodes as comfy_nodes
import node_helpers
import comfy.samplers
import comfy.model_management
import comfy.utils
from .common import VideoGenerationContext, UmeBundle, log_node, validate_bundle
from .ltx_utils import ltx_spatio_temporal_tiled_decode
from typing import Optional, List, Tuple


class UmeAiRT_LTXVideoEnhancer:
    """LTX Video Enhancer — upscales and enhances video quality using guided re-sampling.

    Takes an existing video pipeline and re-processes it through the LTXVLoopingSampler
    with the original frames as guidance, producing higher-quality output at up to the
    specified max resolution.

    Uses vendored LTXVLoopingSampler (0 external dependencies).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "From LTX Loader or Bundle Auto-Loader."}),
                "video_pipe": ("UME_VIDEO_PIPELINE", {"tooltip": "Video pipeline from the Video Generator node."}),
            },
            "optional": {
                "positive": ("POSITIVE", {
                    "forceInput": True,
                    "tooltip": "Enhancement prompt. Optional — defaults to empty (pure structural guidance)."
                }),
                "loras": ("UME_LORA_STACK", {"tooltip": "Connect a LoRA Block node."}),
            },
            "hidden": {
                "max_resolution": ("INT", {
                    "default": 1920, "min": 512, "max": 3840, "step": 32,
                    "tooltip": "Maximum resolution (longest edge). Frames are scaled to fit.",
                    "advanced": True,
                }),
                "denoise_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to denoise. Lower = closer to original, higher = more creative.",
                    "advanced": True,
                }),
                "chunk_frames": ("INT", {
                    "default": 56, "min": 24, "max": 200, "step": 8,
                    "tooltip": "Temporal tile size in pixel frames for the LoopingSampler.",
                    "advanced": True,
                }),
                "chunk_overlap": ("INT", {
                    "default": 24, "min": 16, "max": 80, "step": 8,
                    "tooltip": "Overlap between temporal tiles in pixel frames.",
                    "advanced": True,
                }),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "enhance"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Enhances and upscales LTX video quality using guided re-sampling with overlapping temporal chunks."

    def __init__(self):
        self._lora_loader = comfy_nodes.LoraLoader()

    def enhance(self,
                model_bundle: UmeBundle,
                video_pipe: VideoGenerationContext,
                positive: str = None,
                loras: Optional[List[Tuple[str, float, float]]] = None,
                max_resolution: int = 1920,
                denoise_strength: float = 0.5,
                chunk_frames: int = 56,
                chunk_overlap: int = 24):
        """Enhance video using LTXVLoopingSampler with guided re-sampling."""

        validate_bundle(model_bundle, ["model", "clip", "vae"], context="LTX Video Enhancer")

        source_frames = video_pipe.frames
        if source_frames is None or source_frames.shape[0] == 0:
            raise ValueError("Video Enhancer: No frames in the video pipeline.")

        # --- Read models ---
        model = model_bundle.model
        clip = model_bundle.clip
        vae = model_bundle.vae
        fps = video_pipe.fps

        pos_text = positive if isinstance(positive, str) else ""
        source_audio = video_pipe.audio

        log_node(f"✨ LTX Enhancer: {source_frames.shape[0]} frames, "
                 f"{video_pipe.width}x{video_pipe.height} → max {max_resolution}px | "
                 f"Chunks: {chunk_frames}f / {chunk_overlap}f overlap | "
                 f"Denoise: {denoise_strength}", color="CYAN")

        # --- 1. Apply LoRAs ---
        applied_loras = list(video_pipe.loras) if video_pipe.loras else []
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

        # --- 2. Scale frames to target resolution ---
        n, h, w, c = source_frames.shape
        scale = min(max_resolution / max(h, w), 1.0)  # Don't upscale beyond max
        if scale < 1.0:
            new_h = ((int(h * scale)) // 32) * 32
            new_w = ((int(w * scale)) // 32) * 32
        else:
            new_h = ((h) // 32) * 32
            new_w = ((w) // 32) * 32

        if new_h != h or new_w != w:
            # [N, H, W, C] → [N, C, H, W] for interpolate
            scaled = torch.nn.functional.interpolate(
                source_frames.permute(0, 3, 1, 2),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).clamp(0, 1)
            log_node(f"  Scaled: {h}x{w} → {new_h}x{new_w}", color="GREEN")
        else:
            scaled = source_frames

        # --- 3. Encode prompts ---
        tokens_pos = clip.tokenize(pos_text)
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
        positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]

        # ConditioningZeroOut for negative
        negative_cond = []
        for t in positive_cond:
            d = t[1].copy()
            pooled = d.get("pooled_output", None)
            if pooled is not None:
                d["pooled_output"] = torch.zeros_like(pooled)
            negative_cond.append([torch.zeros_like(t[0]), d])

        # LTXVConditioning (inject frame_rate)
        positive_cond = node_helpers.conditioning_set_values(positive_cond, {"frame_rate": float(fps)})
        negative_cond = node_helpers.conditioning_set_values(negative_cond, {"frame_rate": float(fps)})

        # --- 4. VAE Encode source frames as guiding latents ---
        log_node("  Encoding source frames to latent space...", color="CYAN")
        guiding_latent = vae.encode(scaled[:, :, :, :3])
        guiding_latent_dict = {"samples": guiding_latent}
        log_node(f"  Guiding latent: {guiding_latent.shape}", color="GREEN")

        # Use same latent as input (will be denoised with guidance)
        input_latent_dict = {"samples": guiding_latent.clone()}

        # --- 5. Build guider ---
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive=positive_cond, negative=negative_cond)
        guider.set_cfg(1.0)

        # Build sigmas for enhancement (lower denoise = fewer effective steps)
        from .ltx_sampler import SIGMA_PRESETS
        base_sigmas = SIGMA_PRESETS["standard"]["pass2"]  # [0.85, 0.725, 0.421875, 0.0]
        sigmas = torch.FloatTensor(base_sigmas)

        sampler = comfy.samplers.sampler_object("euler")
        seed = video_pipe.seed

        # --- 6. LTXVLoopingSampler ---
        log_node(f"  Running LoopingSampler: {chunk_frames}f tiles, {chunk_overlap}f overlap...", color="CYAN")

        from vendor.ltxvideo.looping_sampler import LTXVLoopingSampler

        noise_gen = comfy.samplers.Noise_RandomNoise(seed)

        looping_result = LTXVLoopingSampler().sample(
            model=model,
            vae=vae,
            noise=noise_gen,
            sampler=sampler,
            sigmas=sigmas,
            guider=guider,
            latents=input_latent_dict,
            guiding_strength=denoise_strength,
            adain_factor=0.0,
            temporal_tile_size=chunk_frames,
            temporal_overlap=chunk_overlap,
            temporal_overlap_cond_strength=0.5,
            horizontal_tiles=1,
            vertical_tiles=1,
            spatial_overlap=1,
            optional_guiding_latents=guiding_latent_dict,
        )

        enhanced_latent = looping_result[0]
        log_node(f"  LoopingSampler: ✅ Output latent {enhanced_latent['samples'].shape}", color="GREEN")

        # --- 7. VAE Decode ---
        log_node("  Decoding enhanced video (spatio-temporal tiled)...", color="CYAN")
        enhanced_frames = ltx_spatio_temporal_tiled_decode(
            vae, enhanced_latent["samples"],
            spatial_tiles=4, spatial_overlap=1,
            temporal_tile_length=16, temporal_overlap=1
        )
        log_node(f"  Enhanced: {enhanced_frames.shape[0]} frames @ {new_w}x{new_h}", color="GREEN")

        # --- 8. Build output context ---
        ctx = VideoGenerationContext()
        ctx.model = model
        ctx.clip = clip
        ctx.vae = vae
        ctx.model_name = video_pipe.model_name
        ctx.width = new_w
        ctx.height = new_h
        ctx.duration = video_pipe.duration
        ctx.fps = fps
        ctx.frame_count = enhanced_frames.shape[0]
        ctx.steps = len(base_sigmas) - 1
        ctx.cfg = 1.0
        ctx.shift = 0.0
        ctx.sampler_name = "euler"
        ctx.scheduler = "manual_sigmas_standard"
        ctx.seed = seed
        ctx.denoise = denoise_strength
        ctx.loader_type = "ltx2"
        ctx.positive_prompt = pos_text
        ctx.negative_prompt = video_pipe.negative_prompt
        ctx.frames = enhanced_frames
        ctx.loras = applied_loras
        ctx.source_image = video_pipe.source_image
        ctx.audio = source_audio  # Audio passed through unchanged
        ctx.audio_vae = video_pipe.audio_vae

        log_node(f"✨ LTX Enhancer: ✅ Complete — {enhanced_frames.shape[0]} frames"
                 f"{' + audio (passthrough)' if source_audio else ''}", color="GREEN")

        return (ctx,)
