"""
UmeAiRT Toolkit - Video Generator (Orchestrator)
---------------------------------------------------
Unified video generation node. Dispatches to the appropriate backend
based on the model type in the bundle:
  - WAN (T2V, I2V, VACE, FunControl, MoE) → wan_sampler.py
  - LTX-2.3 (dual-pass, AV, ManualSigmas) → ltx_sampler.py

Pattern: mirrors block_sampler.py (thin orchestrator + specialized sub-modules).
"""

import comfy.samplers
from .common import UmeBundle, UmeVideoSettings, UmeVaceFrames, UmeFunControl, log_node, validate_bundle
from .wan_sampler import generate_wan
from .ltx_sampler import generate_ltx
from typing import Optional, List, Tuple


class UmeAiRT_VideoGenerator:
    """Unified video generator — dispatches to WAN or LTX pipeline based on model type.

    Supports:
    - WAN 2.1/2.2: T2V, I2V, VACE, FunControl, MoE (dual-expert)
    - LTX-2.3: T2V, I2V with dual-pass + audio
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "Model bundle from a WAN/LTX Loader or Bundle Auto-Loader node."}),
                "positive": ("POSITIVE", {"forceInput": True, "tooltip": "Describe the motion/action for the video. Connect a Prompt Input or CLIP Text Encode node."}),
                "video_settings": ("UME_VIDEO_SETTINGS", {"tooltip": "Settings from Video Settings node."}),
            },
            "optional": {
                "negative": ("NEGATIVE", {"forceInput": True, "tooltip": "Describe what to avoid. Optional."}),
                "loras": ("UME_LORA_STACK", {"tooltip": "Connect a LoRA Block node."}),
                "source_image": ("IMAGE", {"tooltip": "Source image for Image-to-Video (I2V). If not connected, Text-to-Video (T2V) mode is used."}),
                "vace_frames": ("UME_VACE_FRAMES", {"tooltip": "VACE start+end frame conditioning from a Video VACE Prep node. Overrides source_image when connected. (WAN only)"}),
                "funcontrol": ("UME_FUNCONTROL", {"tooltip": "FunControl conditioning from a Video ControlNet Apply node. Provides pose/depth/canny motion guidance. (WAN only)"}),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Generates video frames based on the provided video models, settings, and prompts. Supports WAN and LTX pipelines."

    def process(self,
                model_bundle: UmeBundle,
                positive: str,
                video_settings: UmeVideoSettings,
                negative: str = None,
                loras: Optional[List[Tuple[str, float, float]]] = None,
                source_image=None,
                vace_frames: Optional[UmeVaceFrames] = None,
                funcontrol: Optional[UmeFunControl] = None):
        """Dispatch to the appropriate video generation backend."""

        validate_bundle(model_bundle, ["model", "clip", "vae"], context="Video Generator")

        if model_bundle.loader_type == "ltx2":
            # LTX pipeline — vace_frames and funcontrol are WAN-only, silently ignored
            if vace_frames is not None:
                log_node("  ⚠️ VACE frames ignored (LTX pipeline does not use VACE)", color="YELLOW")
            if funcontrol is not None:
                log_node("  ⚠️ FunControl ignored (LTX pipeline does not use FunControl)", color="YELLOW")
            return generate_ltx(
                model_bundle, positive, video_settings,
                negative, loras, source_image
            )
        else:
            # WAN pipeline (default)
            return generate_wan(
                model_bundle, positive, video_settings,
                negative, loras, source_image,
                vace_frames, funcontrol
            )
