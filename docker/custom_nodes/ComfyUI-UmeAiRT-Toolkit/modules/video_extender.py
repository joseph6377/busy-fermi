"""
UmeAiRT Toolkit - Video Extender (Orchestrator)
--------------------------------------------------
Unified video extender node. Dispatches to the appropriate backend
based on the model type in the bundle:
  - WAN (VACE continuation) → wan_extender.py
  - LTX-2.3 (reference frames + AV latents) → ltx_extender.py

Pattern: mirrors video_sampler.py (thin orchestrator + specialized sub-modules).
"""

from .common import UmeBundle, UmeVideoSettings, VideoGenerationContext, log_node, validate_bundle
from .wan_extender import extend_wan
from .ltx_extender import extend_ltx
from typing import Optional, List, Tuple


class UmeAiRT_VideoExtender:
    """Unified video extender — dispatches to WAN or LTX pipeline based on model type.

    Takes an existing video from UME_VIDEO_PIPELINE and extends it by generating
    new frames conditioned on the source video's last frame(s).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_pipe": ("UME_VIDEO_PIPELINE", {"tooltip": "Video pipeline to extend. The last frame(s) will be used as reference."}),
                "model_bundle": ("UME_BUNDLE", {"tooltip": "WAN VACE or LTX model bundle."}),
                "positive": ("POSITIVE", {"forceInput": True, "tooltip": "Describe the action/scene for the extended portion of the video."}),
                "video_settings": ("UME_VIDEO_SETTINGS", {"tooltip": "Settings for the extension segment (duration, steps, etc.)."}),
            },
            "optional": {
                "negative": ("NEGATIVE", {"forceInput": True, "tooltip": "Describe what to avoid. Optional."}),
                "loras": ("UME_LORA_STACK", {"tooltip": "Connect a LoRA Block node."}),
            },
            "hidden": {
                # LTX-specific hidden params (ignored by WAN)
                "extend_seconds": ("INT", {
                    "default": 10, "min": 1, "max": 60,
                    "tooltip": "Duration of the extension in seconds. (LTX only)",
                    "advanced": True,
                }),
                "reference_seconds": ("INT", {
                    "default": 3, "min": 1, "max": 10,
                    "tooltip": "Seconds of source video to use as reference context. (LTX only)",
                    "advanced": True,
                }),
                "extend_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extend audio along with video. Disable to keep original audio only. (LTX only)",
                    "advanced": True,
                }),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "extend"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Extends an existing video by generating new frames conditioned on the source video. Supports WAN (VACE) and LTX pipelines."

    def extend(self,
               video_pipe: VideoGenerationContext,
               model_bundle: UmeBundle,
               positive: str,
               video_settings: UmeVideoSettings,
               negative: str = None,
               loras: Optional[List[Tuple[str, float, float]]] = None,
               extend_seconds: int = 10,
               reference_seconds: int = 3,
               extend_audio: bool = True):
        """Dispatch to the appropriate video extension backend."""

        validate_bundle(model_bundle, ["model", "clip", "vae"], context="Video Extender")

        if model_bundle.loader_type == "ltx2":
            return extend_ltx(
                video_pipe, model_bundle, positive, video_settings,
                negative, loras,
                extend_seconds, reference_seconds, extend_audio
            )
        else:
            # WAN pipeline — LTX-specific params are ignored
            return extend_wan(
                video_pipe, model_bundle, positive, video_settings,
                negative, loras
            )
