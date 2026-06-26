"""
UmeAiRT Toolkit - LTX Prompt Director (Chainable)
----------------------------------------------------
Stackable prompt scheduling system for LTX video generation.

Two nodes:
- Prompt Segment: defines a single temporal segment (start_time + prompt).
  Chains together via UME_PROMPT_SCHEDULE (like LoRA Blocks).
- Prompt Director: consumes the schedule chain and generates video
  with per-chunk prompt conditioning via LTXVLoopingSampler.
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


# ---------------------------------------------------------------------------
# Prompt Segment — stackable block
# ---------------------------------------------------------------------------

class UmeAiRT_PromptSegment:
    """Prompt Segment — defines a temporal segment with its prompt.

    Chain multiple segments together to build a prompt schedule.
    Each segment defines when its prompt starts (in seconds).
    The prompt runs until the next segment starts (or end of video).

    Example chain:
        Segment(0s, "forest") → Segment(3s, "walking") → Segment(6s, "sunset") → Director
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_time": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                    "tooltip": "When this prompt segment starts (in seconds from beginning of video).",
                }),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Prompt text for this temporal segment. Describes what happens during this time window.",
                }),
            },
            "optional": {
                "previous": ("UME_PROMPT_SCHEDULE", {
                    "tooltip": "Connect from a previous Prompt Segment to chain schedules.",
                }),
            }
        }

    RETURN_TYPES = ("UME_PROMPT_SCHEDULE",)
    RETURN_NAMES = ("schedule",)
    FUNCTION = "build"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Defines a temporal prompt segment. Chain multiple segments to build a prompt schedule for the Prompt Director."

    def build(self, start_time: float = 0.0, prompt: str = "", previous=None):
        """Build or extend a prompt schedule."""
        schedule = list(previous) if previous else []
        schedule.append({
            "start_time": start_time,
            "prompt": prompt.strip(),
        })
        return (schedule,)


# ---------------------------------------------------------------------------
# Prompt Director — consumes the schedule
# ---------------------------------------------------------------------------

class UmeAiRT_LTXPromptDirector:
    """LTX Prompt Director — generates video with per-chunk prompt conditioning.

    Consumes a UME_PROMPT_SCHEDULE (from chained Prompt Segments) and generates
    video where each temporal chunk uses the conditioning from the matching
    time window.

    Uses LTXVLoopingSampler's optional_positive_conditionings to apply
    different prompts at different points in the video.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "From LTX Loader or Bundle Auto-Loader."}),
                "video_settings": ("UME_VIDEO_SETTINGS", {"tooltip": "From LTX Video Settings node."}),
                "schedule": ("UME_PROMPT_SCHEDULE", {"tooltip": "From chained Prompt Segment nodes."}),
            },
            "optional": {
                "negative": ("NEGATIVE", {"forceInput": True, "tooltip": "Negative prompt (shared across all segments). Optional."}),
                "loras": ("UME_LORA_STACK", {"tooltip": "Connect a LoRA Block node."}),
                "source_image": ("IMAGE", {"tooltip": "Source image for I2V conditioning. If not connected, T2V mode."}),
            },
            "hidden": {
                "chunk_frames": ("INT", {
                    "default": 56, "min": 24, "max": 200, "step": 8,
                    "tooltip": "Temporal tile size in pixel frames for the LoopingSampler.",
                    "advanced": True,
                }),
                "chunk_overlap": ("INT", {
                    "default": 24, "min": 16, "max": 80, "step": 8,
                    "tooltip": "Overlap between temporal tiles.",
                    "advanced": True,
                }),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "generate"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Generates LTX video with per-segment prompt conditioning. Connect Prompt Segment chain for temporal prompt scheduling."

    def __init__(self):
        self._lora_loader = comfy_nodes.LoraLoader()

    def _sort_schedule(self, schedule):
        """Sort schedule by start_time and validate."""
        sorted_schedule = sorted(schedule, key=lambda s: s["start_time"])
        # Remove empty prompts
        sorted_schedule = [s for s in sorted_schedule if s["prompt"]]
        if not sorted_schedule:
            raise ValueError("Prompt Director: No valid prompt segments in schedule. "
                             "Each segment needs a non-empty prompt.")
        return sorted_schedule

    def _encode_schedule(self, schedule, clip, fps, frame_count):
        """Encode each segment's prompt and map to temporal chunk conditionings.

        Returns a list of conditionings, one per temporal chunk used by
        the LoopingSampler.
        """
        encoded = []
        for seg in schedule:
            tokens = clip.tokenize(seg["prompt"])
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conditioning = [[cond, {"pooled_output": pooled}]]
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"frame_rate": float(fps)}
            )
            encoded.append({
                "start_time": seg["start_time"],
                "start_frame": int(seg["start_time"] * fps),
                "conditioning": conditioning,
                "prompt": seg["prompt"],
            })
        return encoded

    def _build_chunk_conditionings(self, encoded_segments, chunk_frames, chunk_overlap, total_latent_frames, time_scale_factor):
        """Map encoded segments to temporal chunks for the LoopingSampler.

        Returns a list of conditionings, one per chunk.
        """
        latent_chunk_size = chunk_frames // time_scale_factor
        latent_overlap = chunk_overlap // time_scale_factor
        chunk_step = latent_chunk_size - latent_overlap

        conditionings = []
        chunk_start = 0
        chunk_idx = 0

        while chunk_start < total_latent_frames:
            chunk_end = min(chunk_start + latent_chunk_size, total_latent_frames)
            # Center of chunk in pixel frames
            chunk_center_px = (chunk_start + (chunk_end - chunk_start) // 2) * time_scale_factor

            # Find the matching segment (last segment whose start_frame <= chunk center)
            matched = encoded_segments[0]
            for seg in encoded_segments:
                if seg["start_frame"] <= chunk_center_px:
                    matched = seg
                else:
                    break

            conditionings.append(matched["conditioning"])
            chunk_start += chunk_step
            chunk_idx += 1

        return conditionings

    def generate(self,
                 model_bundle: UmeBundle,
                 video_settings: UmeVideoSettings,
                 schedule: list,
                 negative: str = None,
                 loras: Optional[List[Tuple[str, float, float]]] = None,
                 source_image=None,
                 chunk_frames: int = 56,
                 chunk_overlap: int = 24):
        """Generate video with per-segment prompt conditioning."""

        validate_bundle(model_bundle, ["model", "clip", "vae"], context="LTX Prompt Director")

        sorted_schedule = self._sort_schedule(schedule)

        # --- Read settings ---
        width = video_settings.width
        height = video_settings.height
        duration = video_settings.duration
        fps = video_settings.frame_rate
        seed = video_settings.seed
        audio_enabled = video_settings.audio_enabled
        sigmas_preset = video_settings.sigmas_preset or "standard"
        custom_sigmas = video_settings.custom_sigmas or ""

        model = model_bundle.model
        clip = model_bundle.clip
        vae = model_bundle.vae
        audio_vae = model_bundle.audio_vae
        latent_upscale_model = model_bundle.latent_upscale_model
        model_name = getattr(model_bundle, "model_name", "")

        neg_text = negative if isinstance(negative, str) else ""
        is_i2v = source_image is not None
        has_upscaler = latent_upscale_model is not None

        frame_count = int(duration * fps) + 1
        frame_count = ((frame_count - 1) // 8) * 8 + 1

        # Log schedule
        log_node(f"🎬 LTX Prompt Director: {len(sorted_schedule)} segments | "
                 f"{width}x{height} | {duration}s ({frame_count} frames @ {fps}fps)", color="CYAN")
        for i, seg in enumerate(sorted_schedule):
            log_node(f"  [{i}] {seg['start_time']:.1f}s: \"{seg['prompt'][:60]}{'...' if len(seg['prompt']) > 60 else ''}\"", color="GREEN")

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

        # --- 2. Encode schedule ---
        encoded_segments = self._encode_schedule(sorted_schedule, clip, fps, frame_count)

        # --- 3. Encode negative ---
        first_cond = encoded_segments[0]["conditioning"]
        if neg_text:
            tokens_neg = clip.tokenize(neg_text)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            negative_cond = [[cond_neg, {"pooled_output": pooled_neg}]]
        else:
            negative_cond = []
            for t in first_cond:
                d = t[1].copy()
                pooled = d.get("pooled_output", None)
                if pooled is not None:
                    d["pooled_output"] = torch.zeros_like(pooled)
                negative_cond.append([torch.zeros_like(t[0]), d])

        negative_cond = node_helpers.conditioning_set_values(negative_cond, {"frame_rate": float(fps)})

        # --- 4. Create empty latents ---
        pass1_w = ((width // 2) // 32) * 32 if has_upscaler else width
        pass1_h = ((height // 2) // 32) * 32 if has_upscaler else height

        video_latent = torch.zeros(
            [1, 128, ((frame_count - 1) // 8) + 1, pass1_h // 32, pass1_w // 32],
            device=comfy.model_management.intermediate_device(),
        )
        video_latent_dict = {"samples": video_latent}

        # --- 5. Optional I2V conditioning ---
        if is_i2v:
            from comfy_extras.nodes_lt import LTXVImgToVideoInplace
            video_latent_dict = LTXVImgToVideoInplace.execute(
                vae=vae, image=source_image, latent=video_latent_dict, strength=1.0
            )
            if hasattr(video_latent_dict, 'args'):
                video_latent_dict = video_latent_dict.args[0]
            log_node("  I2V: source image encoded into latent", color="GREEN")

        # --- 6. Build chunk conditionings ---
        time_scale_factor = vae.downscale_index_formula[0]
        total_latent_frames = video_latent_dict["samples"].shape[2]
        chunk_conditionings = self._build_chunk_conditionings(
            encoded_segments, chunk_frames, chunk_overlap,
            total_latent_frames, time_scale_factor,
        )
        log_node(f"  Chunk conditionings: {len(chunk_conditionings)} chunks mapped", color="GREEN")

        # --- 7. Run LoopingSampler with per-chunk conditioning ---
        sigmas = _parse_sigmas(sigmas_preset, "pass1", custom_sigmas)
        log_node(f"  LoopingSampler: {len(sigmas)-1} steps, {chunk_frames}f tiles, "
                 f"{chunk_overlap}f overlap...", color="CYAN")

        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive=first_cond, negative=negative_cond)
        guider.set_cfg(1.0)

        from vendor.ltxvideo.looping_sampler import LTXVLoopingSampler

        noise_gen = comfy.samplers.Noise_RandomNoise(seed)
        sampler_obj = comfy.samplers.sampler_object("euler")

        looping_result = LTXVLoopingSampler().sample(
            model=model,
            vae=vae,
            noise=noise_gen,
            sampler=sampler_obj,
            sigmas=sigmas,
            guider=guider,
            latents=video_latent_dict,
            guiding_strength=0.0,
            adain_factor=0.0,
            temporal_tile_size=chunk_frames,
            temporal_overlap=chunk_overlap,
            temporal_overlap_cond_strength=0.5,
            horizontal_tiles=1,
            vertical_tiles=1,
            spatial_overlap=1,
            optional_positive_conditionings=chunk_conditionings,
        )

        final_latent = looping_result[0]
        log_node(f"  LoopingSampler: ✅ Output latent {final_latent['samples'].shape}", color="GREEN")

        # --- 8. Decode ---
        log_node("  Decoding video (spatio-temporal tiled)...", color="CYAN")
        frames = ltx_spatio_temporal_tiled_decode(
            vae, final_latent["samples"],
            spatial_tiles=4, spatial_overlap=1,
            temporal_tile_length=16, temporal_overlap=1,
        )
        log_node(f"  Video decoded: {frames.shape[0]} frames", color="GREEN")

        # --- 9. Build context ---
        combined_prompt = " | ".join(
            f"[{s['start_time']:.1f}s] {s['prompt']}" for s in sorted_schedule
        )

        ctx = VideoGenerationContext()
        ctx.model = model
        ctx.clip = clip
        ctx.vae = vae
        ctx.model_name = model_name
        ctx.width = pass1_w
        ctx.height = pass1_h
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
        ctx.positive_prompt = combined_prompt
        ctx.negative_prompt = neg_text
        ctx.frames = frames
        ctx.loras = applied_loras
        ctx.source_image = source_image
        if audio_enabled:
            log_node("  ⚠️ Note: LoopingSampler does not support AV joint generation. "
                     "Audio can be added with the Audio Replacer node.", color="YELLOW")
        ctx.audio = None  # LoopingSampler is video-only; use Audio Replacer for audio
        ctx.audio_vae = audio_vae

        log_node(f"🎬 LTX Prompt Director: ✅ Complete — {frames.shape[0]} frames, "
                 f"{len(sorted_schedule)} segments", color="GREEN")

        return (ctx,)
