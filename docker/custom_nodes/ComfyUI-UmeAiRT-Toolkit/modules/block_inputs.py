import torch
import os
import folder_paths
import nodes as comfy_nodes
import comfy.sd
import comfy.utils
from .common import GenerationContext, UmeSettings, UmeVideoSettings, UmeImage, resize_tensor, apply_outpaint_padding, log_node
import torchvision.transforms.functional as TF
from .logger import logger
from typing import Tuple, Dict, Any, Optional, List


# ──────────────────────────────────────────────────────────────────
# LTX-2.3 Video Settings
# ──────────────────────────────────────────────────────────────────

class UmeAiRT_LTXVideoSettings:
    """LTX-2.3 video settings — backward-compatible alias with LTX defaults.

    Uses ManualSigmas presets instead of standard samplers/schedulers.
    Frame rate defaults to 25fps (LTX-2.3 native). Audio enabled by default.
    WAN-specific fields (steps, cfg, shift, etc.) are set to LTX defaults internally.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {
                    "default": 768, "min": 256, "max": 4096, "step": 32, "display": "slider",
                    "tooltip": "Video width (multiples of 32). 768 is native LTX-2.3 default."
                }),
                "height": ("INT", {
                    "default": 512, "min": 256, "max": 4096, "step": 32, "display": "slider",
                    "tooltip": "Video height (multiples of 32). 512 is native LTX-2.3 default."
                }),
                "duration": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 30.0, "step": 0.5, "display": "slider",
                    "tooltip": "Video duration in seconds. Longer = more VRAM."
                }),
                "frame_rate": ("INT", {
                    "default": 25, "min": 1, "max": 60,
                    "tooltip": "Frames per second. 25 is the LTX-2.3 native rate."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for random number generation."}),
            },
            "optional": {
                "audio_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate audio alongside video. Disable for video-only output."
                }),
                "sigmas_preset": (["Standard (8 steps)", "Fast (4 steps)", "Custom"], {
                    "default": "Standard (8 steps)",
                    "tooltip": "Sigma schedule preset. Standard=8 steps (best quality), Fast=4 steps (requires distilled LoRA)."
                }),
                "custom_sigmas": ("STRING", {
                    "default": "", "advanced": True,
                    "tooltip": "Comma-separated sigma values for pass 1 (e.g. '1.0, 0.85, 0.725, 0.422, 0.0'). Only used with Custom preset."
                }),
            }
        }
    RETURN_TYPES = ("UME_VIDEO_SETTINGS",)
    RETURN_NAMES = ("video_settings",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "LTX-2.3 video+audio settings with ManualSigmas presets."

    def process(self, width, height, duration, frame_rate, seed,
                audio_enabled=True, sigmas_preset="Standard (8 steps)", custom_sigmas=""):
        preset_map = {
            "Standard (8 steps)": "standard",
            "Fast (4 steps)": "fast",
            "Custom": "custom",
        }
        return (UmeVideoSettings(
            width=width, height=height, duration=duration,
            steps=0, cfg=1.0, shift=0.0,  # LTX uses ManualSigmas, not steps/cfg/shift
            sampler_name="euler", scheduler="simple",  # Placeholder — not used by LTX pipeline
            seed=seed,
            frame_rate=frame_rate,
            audio_enabled=audio_enabled,
            sigmas_preset=preset_map.get(sigmas_preset, "standard"),
            custom_sigmas=custom_sigmas,
        ),)


def get_lora_inputs(count):
    inputs = {
        "required": {},
        "optional": {
            "loras": ("UME_LORA_STACK", {"tooltip": "Optional input to chain multiple LoRA stacks."}),
        }
    }
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    for i in range(1, count + 1):
        inputs["optional"][f"lora_{i}_on"] = ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off", "tooltip": f"Toggle LoRA {i} on or off."})
        inputs["optional"][f"lora_{i}_name"] = (lora_list, {"tooltip": f"Select LoRA model {i}."})
        inputs["optional"][f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05, "display": "slider", "tooltip": f"Strength for LoRA {i}. Negative values invert the effect."})
    return inputs

def process_lora_stack(loras, **kwargs):
    current_stack = []
    if loras:
        current_stack.extend(loras)
    
    indices = set()
    for k in kwargs.keys():
        if k.startswith("lora_") and "_name" in k:
            parts = k.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                indices.add(int(parts[1]))
    
    sorted_indices = sorted(list(indices))

    for i in sorted_indices:
        is_on = kwargs.get(f"lora_{i}_on", True)
        name = kwargs.get(f"lora_{i}_name")
        strength = kwargs.get(f"lora_{i}_strength", 1.0)
        
        if is_on and name and name != "None":
            # Unified strength for model and clip
            current_stack.append((name, strength, strength))
            
    return (current_stack,)

def _make_lora_block_class(count):
    """Factory to create LoRA Block node classes with a given slot count."""
    class _LoraBlock:
        @classmethod
        def INPUT_TYPES(s): return get_lora_inputs(count)
        RETURN_TYPES = ("UME_LORA_STACK",)
        RETURN_NAMES = ("loras",)
        FUNCTION = "process"
        CATEGORY = "UmeAiRT/Loaders/LoRA"
        def process(self, loras=None, **kwargs): return process_lora_stack(loras, **kwargs)
    _LoraBlock.__name__ = f"UmeAiRT_LoraBlock_{count}"
    _LoraBlock.__qualname__ = f"UmeAiRT_LoraBlock_{count}"
    _LoraBlock.__doc__ = f"A Node to select and stack up to {count} LoRA model(s) with their strengths."
    _LoraBlock.DESCRIPTION = f"Applies up to {count} LoRA models to the generation pipeline."
    return _LoraBlock

UmeAiRT_LoraBlock_1  = _make_lora_block_class(1)
UmeAiRT_LoraBlock_3  = _make_lora_block_class(3)
UmeAiRT_LoraBlock_5  = _make_lora_block_class(5)
UmeAiRT_LoraBlock_10 = _make_lora_block_class(10)


# ──────────────────────────────────────────────────────────────────
# WAN 2.2 LoRA Blocks (High/Low Noise targeting for MoE)
# ──────────────────────────────────────────────────────────────────

_WAN_LORA_TARGETS = ["Both", "High-Noise Only", "Low-Noise Only"]

def get_wan_lora_inputs(count):
    """Build INPUT_TYPES dict for a WAN 2.2 LoRA Block with per-slot target selection.

    Identical to ``get_lora_inputs`` but adds a ``target`` dropdown per slot
    so the user can direct each LoRA to the High-Noise expert, the Low-Noise
    expert, or both (default).
    """
    inputs = {
        "required": {},
        "optional": {
            "loras": ("UME_LORA_STACK", {"tooltip": "Optional input to chain multiple LoRA stacks."}),
        }
    }
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    for i in range(1, count + 1):
        inputs["optional"][f"lora_{i}_on"] = ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off", "tooltip": f"Toggle LoRA {i} on or off."})
        inputs["optional"][f"lora_{i}_name"] = (lora_list, {"tooltip": f"Select LoRA model {i}."})
        inputs["optional"][f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05, "display": "slider", "tooltip": f"Strength for LoRA {i}. Negative values invert the effect."})
        inputs["optional"][f"lora_{i}_target"] = (_WAN_LORA_TARGETS, {"default": "Both", "tooltip": f"Which WAN 2.2 MoE expert to apply LoRA {i} to. 'Both' applies to High and Low noise models. Use 'High-Noise Only' or 'Low-Noise Only' for targeted application."})
    return inputs

def process_wan_lora_stack(loras, **kwargs):
    """Process LoRA inputs into a stack with 4-element tuples including the target field.

    Produces ``(name, model_strength, clip_strength, target)`` tuples where
    *target* is ``"both"``, ``"high"``, or ``"low"``.  The 4-tuple format is
    backwards-compatible: the ``VideoGenerator`` falls back to ``"both"`` for
    legacy 3-element tuples.
    """
    current_stack = []
    if loras:
        current_stack.extend(loras)

    indices = set()
    for k in kwargs.keys():
        if k.startswith("lora_") and "_name" in k:
            parts = k.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                indices.add(int(parts[1]))

    target_map = {
        "Both": "both",
        "High-Noise Only": "high",
        "Low-Noise Only": "low",
    }

    for i in sorted(indices):
        is_on = kwargs.get(f"lora_{i}_on", True)
        name = kwargs.get(f"lora_{i}_name")
        strength = kwargs.get(f"lora_{i}_strength", 1.0)
        target_label = kwargs.get(f"lora_{i}_target", "Both")
        target = target_map.get(target_label, "both")

        if is_on and name and name != "None":
            current_stack.append((name, strength, strength, target))

    return (current_stack,)

def _make_wan_lora_block_class(count):
    """Factory to create WAN 2.2 LoRA Block node classes with high/low noise targeting."""
    class _WanLoraBlock:
        @classmethod
        def INPUT_TYPES(s): return get_wan_lora_inputs(count)
        RETURN_TYPES = ("UME_LORA_STACK",)
        RETURN_NAMES = ("loras",)
        FUNCTION = "process"
        CATEGORY = "UmeAiRT/Loaders/LoRA/WAN"
        def process(self, loras=None, **kwargs): return process_wan_lora_stack(loras, **kwargs)
    _WanLoraBlock.__name__ = f"UmeAiRT_WanLoraBlock_{count}"
    _WanLoraBlock.__qualname__ = f"UmeAiRT_WanLoraBlock_{count}"
    _WanLoraBlock.__doc__ = f"A WAN 2.2 LoRA Block with {count} slot(s). Each slot can target High-Noise, Low-Noise, or Both MoE experts."
    _WanLoraBlock.DESCRIPTION = f"Applies up to {count} LoRA(s) to WAN 2.2 MoE pipeline with per-slot High/Low noise targeting."
    return _WanLoraBlock

UmeAiRT_WanLoraBlock_1  = _make_wan_lora_block_class(1)
UmeAiRT_WanLoraBlock_3  = _make_wan_lora_block_class(3)
UmeAiRT_WanLoraBlock_5  = _make_wan_lora_block_class(5)
UmeAiRT_WanLoraBlock_10 = _make_wan_lora_block_class(10)


# --- ControlNet Blocks ---

def _get_controlnet_models():
    """Combine local controlnet models with auto-downloadable ones from manifest."""
    local_models = folder_paths.get_filename_list("controlnet")
    
    manifest_models = []
    try:
        from .manifest import load_manifest
        data = load_manifest()
        cnet_data = data.get("_CONTROLNET_MODELS", {})
        for model_name in cnet_data.keys():
            if model_name not in local_models:
                manifest_models.append(model_name)
    except Exception as e:
        log_node(f"ControlNet Apply: Could not load manifest for remote models: {e}", color="YELLOW")
        
    return ["None"] + sorted(list(set(local_models + manifest_models)))

class UmeAiRT_ControlNetImageApply:
    """Injects ControlNet configuration into an image bundle.

    Basic mode shows only strength. Advanced inputs expose start/end percent
    and an optional override control image.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "Input Image Bundle."}),
                "control_net_name": (_get_controlnet_models(), {"tooltip": "Select ControlNet model."}),
                "preprocessor": (["None", "UmeAiRT_Canny", "UmeAiRT_Depth", "UmeAiRT_SoftEdge", "UmeAiRT_Lineart", "UmeAiRT_DWPose"], {"default": "None", "tooltip": "Apply a native ControlNet preprocessor automatically."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider", "tooltip": "How strongly the ControlNet guides the image. Start with 1.0 and lower if the effect is too strong."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "advanced": True, "tooltip": "When the ControlNet starts influencing (0.0 = from the beginning). Raise to let the AI establish composition first."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "advanced": True, "tooltip": "When the ControlNet stops influencing (1.0 = until the very end). Lower to let the AI refine details freely."}),
            },
            "optional": {
                "optional_control_image": ("IMAGE", {"advanced": True, "tooltip": "Optional: Override control image."}),
            }
        }

    RETURN_TYPES = ("UME_IMAGE", "IMAGE")
    RETURN_NAMES = ("image_bundle", "cnet_image")
    FUNCTION = "apply_controlnet"
    CATEGORY = "UmeAiRT/Image"
    DESCRIPTION = "Applies a ControlNet to the generation pipeline using an image bundle."

    def apply_controlnet(self, image_bundle, control_net_name: str, strength: float, start_percent: float = 0.0, end_percent: float = 1.0, optional_control_image: Optional[Any] = None, preprocessor: str = "None"):
        import copy
        new_bundle = copy.copy(image_bundle)
        cnet_stack = list(new_bundle.controlnets) if new_bundle.controlnets else []

        if control_net_name != "None":
            # Auto-download remote model if requested
            local_models = folder_paths.get_filename_list("controlnet")
            if control_net_name not in local_models:
                log_node(f"ControlNet Apply: '{control_net_name}' not found locally. Auto-downloading...", color="CYAN")
                from .manifest import download_bundle_files
                try:
                    resolved_files, meta, dn, sk, err = download_bundle_files("_CONTROLNET_MODELS", control_net_name)
                    if err:
                        raise RuntimeError(f"ControlNet Apply: Failed to auto-download {control_net_name}: {', '.join(err)}")
                except Exception as e:
                    log_node(f"ControlNet Apply: Remote manifest resolution error for '{control_net_name}': {e}", color="RED")
                    raise RuntimeError(f"ControlNet Apply: failed to retrieve '{control_net_name}': {e}")

            control_use_image = optional_control_image if optional_control_image is not None else new_bundle.image

            if control_use_image is None:
                raise ValueError("ControlNet Image Apply: No Image found in bundle and no optional image provided.")

            if preprocessor == "UmeAiRT_Canny":
                try:
                    from .preprocessors.canny_core import apply_canny
                    log_node("ControlNet Apply: Running native Canny Preprocessor...", color="CYAN")
                    control_use_image = apply_canny(control_use_image, low_threshold=100, high_threshold=200)
                    log_node("ControlNet Apply: Native Canny Success.", color="GREEN")
                except Exception as e:
                    log_node(f"UmeAiRT ControlNet Apply: Native Canny failed: {e}", color="RED")
            elif preprocessor == "UmeAiRT_Depth":
                try:
                    from .manifest import download_bundle_files
                    log_node("ControlNet Apply: Fetching Depth Model...", color="CYAN")
                    resolved, _, _, _, errs = download_bundle_files("PREPROCESSORS/Depth", "Zoe-N")
                    if errs:
                        log_node(f"Failed to download Depth Model from UmeAiRT Asset CDN: {errs}", color="RED")
                    else:
                        import os
                        model_path = os.path.join(folder_paths.models_dir, "preprocessors", "depth", "Intel-zoedepth-nyu-kitti")
                        from .preprocessors.depth_core import apply_zoedepth
                        log_node("ControlNet Apply: Running native Depth Preprocessor...", color="CYAN")
                        control_use_image = apply_zoedepth(control_use_image, model_path)
                        log_node("ControlNet Apply: Native Depth Success.", color="GREEN")
                except Exception as e:
                    log_node(f"UmeAiRT ControlNet Apply: Native Depth failed: {e}", color="RED")
            elif preprocessor == "UmeAiRT_SoftEdge":
                try:
                    from .manifest import download_bundle_files
                    log_node("ControlNet Apply: Fetching SoftEdge Model...", color="CYAN")
                    resolved, _, _, _, errs = download_bundle_files("PREPROCESSORS/SoftEdge", "HED")
                    if errs:
                        log_node(f"Failed to download SoftEdge Model from UmeAiRT Asset CDN: {errs}", color="RED")
                    else:
                        import os
                        model_path = os.path.join(folder_paths.models_dir, "models_base", "ControlNetHED.pth")
                        from .preprocessors.hed_core import apply_hed
                        log_node("ControlNet Apply: Running native SoftEdge Preprocessor...", color="CYAN")
                        control_use_image = apply_hed(control_use_image, model_path)
                        log_node("ControlNet Apply: Native SoftEdge Success.", color="GREEN")
                except Exception as e:
                    log_node(f"UmeAiRT ControlNet Apply: Native SoftEdge failed: {e}", color="RED")
            elif preprocessor == "UmeAiRT_Lineart":
                try:
                    from .manifest import download_bundle_files
                    log_node("ControlNet Apply: Fetching Lineart Model...", color="CYAN")
                    resolved, _, _, _, errs = download_bundle_files("PREPROCESSORS/Lineart", "Standard")
                    if errs:
                        log_node(f"Failed to download Lineart Model from UmeAiRT Asset CDN: {errs}", color="RED")
                    else:
                        import os
                        model_path_coarse = os.path.join(folder_paths.models_dir, "models_base", "sk_model2.pth")
                        model_path_fine = os.path.join(folder_paths.models_dir, "models_base", "sk_model.pth")
                        from .preprocessors.lineart_core import apply_lineart
                        log_node("ControlNet Apply: Running native Lineart Preprocessor...", color="CYAN")
                        control_use_image = apply_lineart(control_use_image, model_path_fine, model_path_coarse, use_coarse=True)
                        log_node("ControlNet Apply: Native Lineart Success.", color="GREEN")
                except Exception as e:
                    log_node(f"UmeAiRT ControlNet Apply: Native Lineart failed: {e}", color="RED")
            elif preprocessor == "UmeAiRT_DWPose":
                try:
                    from .manifest import download_bundle_files
                    log_node("ControlNet Apply: Fetching DWPose Model...", color="CYAN")
                    resolved, _, _, _, errs = download_bundle_files("PREPROCESSORS/Pose", "DWPose")
                    if errs:
                        log_node(f"Failed to download DWPose Model from UmeAiRT Asset CDN: {errs}", color="RED")
                    else:
                        import os
                        model_path_det = os.path.join(folder_paths.models_dir, "models_base", "yolox_l.onnx")
                        model_path_pose = os.path.join(folder_paths.models_dir, "models_base", "dw-ll_ucoco_384.onnx")
                        from .preprocessors.dwpose_core import apply_dwpose
                        log_node("ControlNet Apply: Running native DWPose Preprocessor...", color="CYAN")
                        control_use_image = apply_dwpose(control_use_image, model_path_det, model_path_pose)
                        log_node("ControlNet Apply: Native DWPose Success.", color="GREEN")
                except Exception as e:
                    log_node(f"UmeAiRT ControlNet Apply: Native DWPose failed: {e}", color="RED")

            union_type_map = {
                "UmeAiRT_Canny": "canny/lineart/anime_lineart/mlsd",
                "UmeAiRT_Depth": "depth",
                "UmeAiRT_SoftEdge": "hed/pidi/scribble/ted",
                "UmeAiRT_Lineart": "canny/lineart/anime_lineart/mlsd",
                "UmeAiRT_DWPose": "openpose",
            }
            c_type = union_type_map.get(preprocessor, None)

            cnet_stack.append((control_net_name, control_use_image, strength, start_percent, end_percent, c_type))
            out_image = control_use_image
        else:
            out_image = optional_control_image if optional_control_image is not None else new_bundle.image

        new_bundle.controlnets = cnet_stack

        return (new_bundle, out_image)



# --- Parameter Blocks ---


class UmeAiRT_GenerationSettings:
    """Standalone settings node — outputs a dict of generation parameters."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "display": "slider", "tooltip": "Target width of the generated image."}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "display": "slider", "tooltip": "Target height of the generated image."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1, "display": "slider", "tooltip": "Total sampling steps."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5, "display": "slider", "tooltip": "CFG Scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise scheduler."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for random number generation."}),
            }
        }
    RETURN_TYPES = ("UME_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Inputs"
    DESCRIPTION = "Defines core generation parameters: resolution, steps, CFG, sampler, and seed."

    def process(self, width: int, height: int, steps: int, cfg: float, sampler_name: str, scheduler: str, seed: int):
        return (UmeSettings(width=width, height=height, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, seed=seed),)


class UmeAiRT_VideoSettings:
    """Video generation settings — outputs a UME_VIDEO_SETTINGS bundle.

    Supports both WAN and LTX pipelines. WAN-specific fields (steps, cfg, shift,
    sampler, scheduler) are ignored by LTX. LTX-specific fields (frame_rate,
    audio_enabled, sigmas_preset) are in optional and ignored by WAN.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {
                    "default": 848, "min": 256, "max": 4096, "step": 16, "display": "slider",
                    "tooltip": "Video width. WAN optimal: 848/1280. LTX optimal: 768."
                }),
                "height": ("INT", {
                    "default": 480, "min": 256, "max": 4096, "step": 16, "display": "slider",
                    "tooltip": "Video height. WAN optimal: 480/720. LTX optimal: 512."
                }),
                "duration": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.5, "display": "slider", "tooltip": "Video duration in seconds. Set to 0.0 for a single image (1 frame). Longer = more VRAM."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1, "display": "slider", "tooltip": "Total sampling steps. (WAN only — LTX uses sigmas presets)"}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 15.0, "step": 0.5, "display": "slider", "tooltip": "CFG Scale. (WAN only — LTX uses 1.0)"}),
                "shift": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider", "advanced": True, "tooltip": "ModelSamplingSD3 shift value. (WAN only)"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm. (WAN only — LTX uses euler)"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise scheduler. (WAN only)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for random number generation."}),
            },
            "optional": {
                "frame_rate": ("INT", {
                    "default": 16, "min": 1, "max": 60, "step": 1, "display": "slider", "advanced": True,
                    "tooltip": "Frames per second. WAN=16, LTX=25."
                }),
                "audio_enabled": ("BOOLEAN", {
                    "default": False, "advanced": True,
                    "tooltip": "Generate audio alongside video. (LTX only)"
                }),
                "sigmas_preset": (["None", "Standard (8 steps)", "Fast (4 steps)", "Custom"], {
                    "default": "None", "advanced": True,
                    "tooltip": "ManualSigmas preset. (LTX only — overrides steps/cfg/scheduler)"
                }),
                "custom_sigmas": ("STRING", {
                    "default": "", "advanced": True,
                    "tooltip": "Comma-separated sigma values for pass 1. (LTX only, Custom preset)"
                }),
            }
        }
    RETURN_TYPES = ("UME_VIDEO_SETTINGS",)
    RETURN_NAMES = ("video_settings",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Defines video generation parameters: resolution, duration, steps, CFG, and sampler. Supports WAN and LTX."

    def process(self, width, height, duration, steps, cfg, shift, sampler_name, scheduler, seed,
                frame_rate=16, audio_enabled=False, sigmas_preset="None", custom_sigmas=""):
        # Map sigmas_preset display names to internal keys
        preset_map = {
            "None": "",
            "Standard (8 steps)": "standard",
            "Fast (4 steps)": "fast",
            "Custom": "custom",
        }
        return (UmeVideoSettings(
            width=width, height=height, duration=duration,
            steps=steps, cfg=cfg, shift=shift,
            sampler_name=sampler_name, scheduler=scheduler, seed=seed,
            frame_rate=frame_rate,
            audio_enabled=audio_enabled,
            sigmas_preset=preset_map.get(sigmas_preset, ""),
            custom_sigmas=custom_sigmas,
        ),)



# --- Image Blocks ---

class UmeAiRT_BlockImageLoader(comfy_nodes.LoadImage):
    """Image loader formatted as a Block.

    Outputs a unified UME_IMAGE bundle plus raw IMAGE and MASK tensors.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files.sort()
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True, "tooltip": "Select an image file to load from disk."}),
            },
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "load_block_image"
    CATEGORY = "UmeAiRT/Image"
    DESCRIPTION = "Loads an image from disk and packages it into a UME_IMAGE bundle."

    def load_block_image(self, image: str):
        """Loads the specified image file and wraps it in an UmeImage dataclass."""
        out = super().load_image(image)
        img, mask = out[0], out[1]
        
        image_bundle = UmeImage(image=img, mask=mask, mode="img2img", denoise=0.75)
        return (image_bundle,)

def process_image_core(image_bundle, mode: str, denoise: float = 0.75, auto_resize: bool = False, mask_blur: int = 0, 
                      padding_left: int = 0, padding_top: int = 0, padding_right: int = 0, padding_bottom: int = 0):
    image = image_bundle.image
    mask = image_bundle.mask
    
    if image is None: raise ValueError("Block Image Process: Bundle has no image.")

    B, H, W, C = image.shape
    final_image, final_mask = image, mask

    if mode == "outpaint":
         pad_l, pad_t, pad_r, pad_b = padding_left, padding_top, padding_right, padding_bottom
         final_image, final_mask = apply_outpaint_padding(
             final_image, final_mask, pad_l, pad_t, pad_r, pad_b, overlap=8, feathering=40
         )

    if (mode == "inpaint" or mode == "outpaint") and final_mask is not None and mask_blur > 0:
         if len(final_mask.shape) == 2: m = final_mask.unsqueeze(0).unsqueeze(0)
         else: m = final_mask
         k = mask_blur
         if k % 2 == 0: k += 1
         m = TF.gaussian_blur(m, kernel_size=k)
         final_mask = m.squeeze(0).squeeze(0) if len(final_mask.shape) == 2 else m

    final_mode = "inpaint" if mode in ["inpaint", "outpaint"] else "img2img"
    if mode == "img2img": final_mask = None

    return (UmeImage(image=final_image, mask=final_mask, mode=final_mode, denoise=denoise, auto_resize=auto_resize),)


class UmeAiRT_BlockImageProcess:
    """Structural pre-processor for UME_IMAGE bundles in Block-based workflows.

    Handles cropping, padding (Outpaint mapping), and conditional context tagging.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI changes the image. 1.0 = completely new image, 0.5 = keeps half the original detail."}),
                "mode": (["img2img", "inpaint", "outpaint", "txt2img"], {"default": "img2img", "tooltip": "How to process the image: img2img (transform), inpaint (fill masked area), or outpaint (extend edges)."}),
            },
            "optional": {
                "auto_resize": ("BOOLEAN", {"default": False, "label_on": "Resize to Settings", "label_off": "Keep Original", "tooltip": "Automatically resize the source image to match the width/height from Generation Settings."}),
                "mask_blur": ("INT", {"default": 10, "tooltip": "Softens the edge of the inpaint mask for smoother blending. Higher = softer transitions."}),
                "padding_left": ("INT", {"default": 0, "tooltip": "Pixels to add on the left side when using outpaint mode."}), "padding_top": ("INT", {"default": 0, "tooltip": "Pixels to add on the top side when using outpaint mode."}),
                "padding_right": ("INT", {"default": 0, "tooltip": "Pixels to add on the right side when using outpaint mode."}), "padding_bottom": ("INT", {"default": 0, "tooltip": "Pixels to add on the bottom side when using outpaint mode."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process_image"
    CATEGORY = "UmeAiRT/Image"
    DESCRIPTION = "Processes an image bundle for img2img, inpaint, or outpaint workflows."

    def process_image(self, image_bundle, denoise: float = 0.75, mode: str = "img2img", auto_resize: bool = False, mask_blur: int = 0, 
                      padding_left: int = 0, padding_top: int = 0, padding_right: int = 0, padding_bottom: int = 0):
        return process_image_core(image_bundle, mode=mode, denoise=denoise, auto_resize=auto_resize, mask_blur=mask_blur, 
                                  padding_left=padding_left, padding_top=padding_top, padding_right=padding_right, padding_bottom=padding_bottom)


class UmeAiRT_ImageProcess_Img2Img:
    """Pre-processor for Img2Img workflows."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI changes the image."}),
            },
            "optional": {
                "auto_resize": ("BOOLEAN", {"default": False, "label_on": "Resize to Settings", "label_off": "Keep Original", "tooltip": "Automatically resize the source image to match the width/height from Generation Settings."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Image"
    DESCRIPTION = "Prepares an image bundle for standard image-to-image workflows."

    def process(self, image_bundle, denoise=0.75, auto_resize=False):
        return process_image_core(image_bundle, mode="img2img", denoise=denoise, auto_resize=auto_resize)


class UmeAiRT_ImageProcess_Inpaint:
    """Pre-processor for Inpaint workflows."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI changes the image inside the mask."}),
            },
            "optional": {
                "mask_blur": ("INT", {"default": 10, "tooltip": "Softens the edge of the inpaint mask for smoother blending."}),
                "auto_resize": ("BOOLEAN", {"default": False, "label_on": "Resize to Settings", "label_off": "Keep Original", "tooltip": "Automatically resize the source image to match the width/height from Generation Settings."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Image"
    DESCRIPTION = "Prepares an image bundle with a mask for inpainting workflows."

    def process(self, image_bundle, denoise=0.75, mask_blur=10, auto_resize=False):
        return process_image_core(image_bundle, mode="inpaint", denoise=denoise, mask_blur=mask_blur, auto_resize=auto_resize)


class UmeAiRT_ImageProcess_Outpaint:
    """Outpaint configurator — tags the image bundle with target dimensions.

    Does NOT modify the image. The KSampler handles the actual resize,
    padding, mask generation, and blurring at execution time.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI changes the image in the padded areas."}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64, "display": "slider", "tooltip": "Desired final width of the outpainted image."}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64, "display": "slider", "tooltip": "Desired final height of the outpainted image."}),
            },
            "optional": {
                "horizontal_align": (["center", "left", "right"], {"default": "center", "advanced": True, "tooltip": "Where to place the source image horizontally within the target canvas."}),
                "vertical_align": (["center", "top", "bottom"], {"default": "center", "advanced": True, "tooltip": "Where to place the source image vertically within the target canvas."}),
                "mask_blur": ("INT", {"default": 10, "advanced": True, "tooltip": "Softens the edge of the outpaint mask for smoother blending."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Image"
    DESCRIPTION = "Configures an image bundle with padding dimensions for outpainting."

    def process(self, image_bundle, denoise=0.75, target_width=1024, target_height=1024,
                horizontal_align="center", vertical_align="center", mask_blur=10):
        import copy
        new_bundle = copy.copy(image_bundle)
        
        new_bundle.mode = "outpaint"
        new_bundle.denoise = denoise
        new_bundle.auto_resize = False
        new_bundle.outpaint_target_w = target_width
        new_bundle.outpaint_target_h = target_height
        new_bundle.outpaint_h_align = horizontal_align
        new_bundle.outpaint_v_align = vertical_align
        new_bundle.outpaint_mask_blur = mask_blur
        return (new_bundle,)


class UmeAiRT_ImageProcess_Kontext:
    """Pre-processor for FLUX Kontext image editing workflows.

    Accepts 1-2 reference images and produces a bundle tagged with mode="kontext".
    The sampler uses ImageStitch + FluxKontextImageScale + ReferenceLatent
    to inject the source image(s) into the conditioning pipeline.

    Kontext prompts describe the edit to perform on the source image, e.g.
    "Change the background to a beach while keeping the person in the same pose".
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "The primary image bundle to process."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional second reference image for multi-image Kontext editing (stitched side-by-side with the primary image)."}),
                "auto_resize": ("BOOLEAN", {"default": False, "label_on": "Resize to Settings", "label_off": "Keep Original", "tooltip": "Automatically resize the source image to match the width/height from Generation Settings."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Image"
    DESCRIPTION = "Prepares image bundles for FLUX Kontext reference-based editing."

    def process(self, image_bundle, reference_image=None, auto_resize=False):
        import copy
        new_bundle = copy.copy(image_bundle)
        new_bundle.mode = "kontext"
        new_bundle.denoise = 1.0  # Kontext always uses full denoise
        new_bundle.reference_image = reference_image
        new_bundle.auto_resize = auto_resize
        return (new_bundle,)


class UmeAiRT_ImageProcess_Edit:
    """Pre-processor for QWEN Image Edit workflows.

    Accepts 1-3 reference images for multi-image editing using
    TextEncodeQwenImageEditPlus. The BlockSampler handles the
    specialized QWEN encoding and sampling pipeline internally.

    Edit prompts describe the transformation to apply, e.g.
    "Replace the cat with a dalmatian" or
    "Make these 3 characters walking down a street in Japan at night".
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "Primary image to edit. This is the first reference image."}),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "Optional second reference image for multi-image editing."}),
                "image_3": ("IMAGE", {"tooltip": "Optional third reference image for multi-image editing."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Image"
    DESCRIPTION = "Prepares image bundles for QWEN Image Edit multi-image editing workflows."

    def process(self, image_bundle, image_2=None, image_3=None):
        import copy
        new_bundle = copy.copy(image_bundle)
        new_bundle.mode = "qwen_edit"
        new_bundle.denoise = 1.0
        new_bundle.edit_images = [image_2, image_3]
        return (new_bundle,)

class UmeAiRT_Positive_Input:
    """Multiline text editor for the positive prompt. Outputs a STRING."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Positive prompt."}),
            }
        }

    RETURN_TYPES = ("POSITIVE",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "pass_through"
    CATEGORY = "UmeAiRT/Inputs"
    DESCRIPTION = "Input node for the positive text prompt."

    def pass_through(self, positive):
        return (positive,)


class UmeAiRT_Negative_Input:
    """Multiline text editor for the negative prompt. Outputs a STRING."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "negative": ("STRING", {"default": "text, watermark", "multiline": True, "dynamicPrompts": True, "tooltip": "Negative prompt."}),
            }
        }

    RETURN_TYPES = ("NEGATIVE",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "pass_through"
    CATEGORY = "UmeAiRT/Inputs"
    DESCRIPTION = "Input node for the negative text prompt."

    def pass_through(self, negative):
        return (negative,)


class UmeAiRT_BlockVideoLoader:
    """Video loader formatted as a Block.

    Uses PyAV to decode a video from disk into native ComfyUI tensors,
    and outputs a unified UME_VIDEO_PIPELINE.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        # Basic filtering to avoid overwhelming the list with text files, etc.
        video_files = [f for f in files if f.lower().endswith(('.mp4', '.webm', '.mov', '.avi', '.mkv'))]
        if not video_files:
            video_files = ["No video found in input directory"]
        else:
            video_files.sort()
            
        return {
            "required": {
                "video": (video_files, {"video_upload": True, "tooltip": "Select a video file to load from disk."}),
                "force_fps": ("INT", {"default": 0, "min": 0, "max": 120, "tooltip": "Force a specific FPS. 0 means use the video's native FPS."}),
            },
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "load_video"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Loads a video from disk natively using PyAV and packages it into a UME_VIDEO_PIPELINE."

    def load_video(self, video: str, force_fps: int = 0):
        if video == "No video found in input directory":
            raise ValueError("No video selected.")

        import av
        import numpy as np
        from .common import VideoGenerationContext
        
        video_path = folder_paths.get_annotated_filepath(video)
        if not os.path.exists(video_path):
            video_path = os.path.join(folder_paths.get_input_directory(), video)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video}")
            
        container = av.open(video_path)
        if not container.streams.video:
            raise ValueError(f"No video streams found in {video_path}")
            
        video_stream = container.streams.video[0]
        
        fps = video_stream.average_rate
        native_fps = int(fps.numerator / fps.denominator) if fps and fps.denominator != 0 else 16
        final_fps = force_fps if force_fps > 0 else native_fps
        
        frames = []
        for frame in container.decode(video=0):
            img_np = frame.to_ndarray(format='rgb24')
            frames.append(img_np)
            
        if not frames:
            raise ValueError(f"No frames could be decoded from {video_path}")
            
        frames_np = np.stack(frames)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0  # [N, H, W, 3]
        
        # Audio extraction
        audio_data = None
        try:
            if container.streams.audio:
                audio_stream = container.streams.audio[0]
                audio_frames = []
                sample_rate = audio_stream.rate
                for frame in container.decode(audio=0):
                    audio_frames.append(frame.to_ndarray())
                if audio_frames:
                    # PyAV audio to_ndarray usually returns [channels, samples]
                    waveform = np.concatenate(audio_frames, axis=1) # [channels, total_samples]
                    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0) # [1, channels, total_samples]
                    audio_data = {"waveform": waveform_tensor, "sample_rate": sample_rate}
        except Exception as e:
            log_node(f"Block Video Loader: Audio extraction failed: {e}", color="YELLOW")
            
        container.close()
        
        duration = frames_tensor.shape[0] / final_fps
        
        ctx = VideoGenerationContext()
        ctx.frames = frames_tensor
        ctx.width = frames_tensor.shape[2]
        ctx.height = frames_tensor.shape[1]
        ctx.duration = duration
        ctx.fps = final_fps
        ctx.frame_count = frames_tensor.shape[0]
        ctx.audio = audio_data
        
        log_node(f"🎞️ Block Video Loader: Loaded {ctx.frame_count} frames @ {ctx.fps}fps ({ctx.width}x{ctx.height})", color="CYAN")
        if audio_data:
            log_node(f"🔊 Block Video Loader: Loaded audio ({audio_data['waveform'].shape[2]} samples @ {audio_data['sample_rate']}Hz)", color="CYAN")
            
        return (ctx,)

