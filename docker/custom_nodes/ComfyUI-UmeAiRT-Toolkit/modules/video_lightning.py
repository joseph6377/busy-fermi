"""
UmeAiRT Toolkit - Video Lightning Accelerator
-----------------------------------------------
Sits between the WAN Loader and the Video Generator on the UME_BUNDLE wire.
Automatically loads the correct lightx2v step-distilled LoRA for WAN models
and embeds settings overrides (cfg, steps, sampler, scheduler) into the
bundle's ``overrides`` dict.

Mirrors the pattern of block_lightning.py (QWEN Lightning Accelerator).
"""

import os
import folder_paths
import nodes as comfy_nodes
import comfy.samplers
from .common import UmeBundle, log_node
from .manifest import load_manifest, download_bundle_files


# ---------------------------------------------------------------------------
# Internal mapping: (model_variant, step_count) → LoRA filename
# WAN lightx2v LoRAs for step-distilled acceleration.
# ---------------------------------------------------------------------------

_WAN_LORA_MAP = {
    # WAN 2.1 — 4 steps
    ("2.1", "i2v", 4): "WAN2.1/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
    ("2.1", "t2v", 4): "WAN2.1/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
    
    # WAN 2.1 VACE — fallback to T2V (same base architecture, no VACE-specific distillation exists)
    ("2.1", "vace", 4): "WAN2.1/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
    
    # WAN 2.2 HIGH — 4 steps
    ("2.2", "i2v", "high", 4): "WAN2.2/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
    ("2.2", "t2v", "high", 4): "WAN2.2/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors",
    
    # WAN 2.2 LOW — 4 steps
    ("2.2", "i2v", "low", 4): "WAN2.2/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
    ("2.2", "t2v", "low", 4): "WAN2.2/Wan2.2-Lightning_T2V-A14B-4steps-lora_LOW_fp16.safetensors",
}

def _resolve_wan_lora(model_name, step_count, force_noise=None):
    """Determine the Lightning LoRA filename for a WAN model."""
    name_lower = model_name.lower().replace("_", "-")
    
    version = "2.1"
    if "2.2" in name_lower or "2-2" in name_lower:
        version = "2.2"
        
    task = None
    if "i2v" in name_lower:
        task = "i2v"
    elif "t2v" in name_lower:
        task = "t2v"
    elif "vace" in name_lower:
        task = "vace"
        log_node("⚡ Video Lightning: Using T2V Lightning LoRA as fallback for VACE model "
                 "(no VACE-specific distillation available).", color="CYAN")
        
    if not task:
        return None
        
    if version == "2.1":
        return _WAN_LORA_MAP.get((version, task, step_count))
    else:
        if force_noise is not None:
            noise = force_noise
        else:
            noise = "high" if "high" in name_lower else "low"
        return _WAN_LORA_MAP.get((version, task, noise, step_count))


def _try_manifest_download_wan(lora_filename):
    """Attempt to download a WAN Lightning LoRA from the manifest if not present locally.

    Looks up the ``_ACCELERATION_LORAS`` section of the model manifest.

    Args:
        lora_filename (str): The LoRA filename to look for.

    Returns:
        bool: True if the file is now available locally.
    """
    basename = os.path.basename(lora_filename)

    # Check if already present
    existing = folder_paths.get_full_path("loras", basename)
    if existing and os.path.exists(existing):
        return True

    existing = folder_paths.get_full_path("loras", lora_filename)
    if existing and os.path.exists(existing):
        return True

    # Try downloading from manifest
    try:
        data = load_manifest()
        accel_section = data.get("_ACCELERATION_LORAS", {})

        for family_key, family_data in accel_section.items():
            if not isinstance(family_data, dict):
                continue
            for variant_key, variant_data in family_data.items():
                if variant_key.startswith("_") or not isinstance(variant_data, dict):
                    continue
                files = variant_data.get("files", [])
                for file_entry in files:
                    file_path = file_entry.get("path", "")
                    if os.path.basename(file_path) == basename:
                        log_node(f"Video Lightning: ⬇️ Downloading {basename}...", color="CYAN")
                        download_bundle_files(f"_ACCELERATION_LORAS/{family_key}", variant_key)
                        check = folder_paths.get_full_path("loras", basename)
                        if check and os.path.exists(check):
                            return True
                        check = folder_paths.get_full_path("loras", lora_filename)
                        return check is not None and os.path.exists(check)
    except Exception as e:
        log_node(f"Video Lightning: Manifest download failed: {e}", color="YELLOW")

    return False


class UmeAiRT_VideoLightningAccelerator:
    """Lightning Acceleration LoRA applicator for WAN video models.

    Sits on the UME_BUNDLE wire between a WAN Loader and the Video Generator.
    Loads the correct lightx2v LoRA (4 steps) based on the selected WAN model
    variant (I2V or T2V), applies it to the model, and embeds settings
    overrides (cfg, steps, sampler, scheduler) into the bundle.

    The Video Generator reads ``model_bundle.overrides`` and silently applies
    them on top of the user's VideoSettings.

    When set to "Off", acts as a pure pass-through.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "Model bundle from a WAN Loader or Bundle Auto-Loader node."}),
                "mode": (["Off", "4 Steps"], {"default": "Off", "tooltip": "Lightning acceleration mode. Off = pass-through, 4 Steps = apply lightx2v LoRA and override settings."}),
            },
            "optional": {
                "lora_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "display": "slider", "advanced": True,
                    "tooltip": "Strength of the Lightning LoRA on the model. Default 1.0 is recommended."
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler", "advanced": True,
                    "tooltip": "Sampler override when Lightning is active. Euler is recommended."
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "sgm_uniform", "advanced": True,
                    "tooltip": "Scheduler override when Lightning is active. SGM Uniform is recommended."
                }),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Applies lightning acceleration overrides for video generation."

    def __init__(self):
        self._lora_loader = comfy_nodes.LoraLoader()

    def process(self, model_bundle, mode="Off",
                lora_strength=1.0, sampler_name="euler", scheduler="sgm_uniform"):
        # --- Pass-through when Off ---
        if mode == "Off":
            return (model_bundle,)

        step_count = 4  # Currently only 4-step distillation available
        loader_type = getattr(model_bundle, "loader_type", "")

        # --- Guard: only WAN models are supported ---
        if loader_type != "wan":
            log_node(
                f"Video Lightning: ⚠️ Non-WAN model detected (loader_type='{loader_type}'). "
                f"Video Lightning LoRAs are only available for WAN models. Passing through.",
                color="YELLOW"
            )
            return (model_bundle,)

        # --- Resolve the correct LoRA file(s) ---
        model_name = getattr(model_bundle, "model_name", "")
        model_low_noise = getattr(model_bundle, "model_low_noise", None)
        
        # Determine if we should force specific noise experts based on MoE presence
        force_high = "high" if model_low_noise is not None else None
        lora_rel_path = _resolve_wan_lora(model_name, step_count, force_noise=force_high)

        if not lora_rel_path:
            log_node(
                f"Video Lightning: ⚠️ No Lightning LoRA mapping found for model '{model_name}'. Passing through.",
                color="YELLOW"
            )
            return (model_bundle,)

        lora_basename = os.path.basename(lora_rel_path)

        def _get_or_download_lora(rel_path, basename):
            path = folder_paths.get_full_path("loras", basename)
            if not path or not os.path.exists(path):
                path = folder_paths.get_full_path("loras", rel_path)
            if not path or not os.path.exists(path):
                if _try_manifest_download_wan(rel_path):
                    path = folder_paths.get_full_path("loras", basename)
                    if not path:
                        path = folder_paths.get_full_path("loras", rel_path)
            return path

        lora_path = _get_or_download_lora(lora_rel_path, lora_basename)

        if not lora_path or not os.path.exists(lora_path):
            log_node(
                f"Video Lightning: ❌ LoRA file not found: '{lora_basename}'. "
                f"Please place it in your ComfyUI/models/loras/WAN2.1/ folder. Passing through.",
                color="RED"
            )
            return (model_bundle,)

        # --- Apply the LoRA to the model(s) ---
        model = model_bundle.model
        clip = model_bundle.clip

        if not model or not clip:
            log_node("Video Lightning: ❌ Bundle is missing model or CLIP. Cannot apply LoRA.", color="RED")
            return (model_bundle,)

        try:
            model_patched, clip_patched = self._lora_loader.load_lora(
                model, clip, lora_basename, lora_strength, lora_strength
            )
            
            model_low_noise_patched = None
            if model_low_noise is not None:
                lora_low_rel_path = _resolve_wan_lora(model_name, step_count, force_noise="low")
                if lora_low_rel_path:
                    lora_low_basename = os.path.basename(lora_low_rel_path)
                    lora_low_path = _get_or_download_lora(lora_low_rel_path, lora_low_basename)
                    
                    if lora_low_path and os.path.exists(lora_low_path):
                        # Apply the low-noise LoRA to the low-noise expert
                        model_low_noise_patched, _ = self._lora_loader.load_lora(
                            model_low_noise, clip, lora_low_basename, lora_strength, 0.0
                        )
                        log_node(f"Video Lightning: Applied low-noise expert LoRA '{lora_low_basename}'", color="GREEN")
                    else:
                        log_node(f"Video Lightning: ⚠️ Low-noise LoRA not found: '{lora_low_basename}'. Skipping low-noise patch.", color="YELLOW")
                        model_low_noise_patched = model_low_noise
                else:
                    model_low_noise_patched = model_low_noise
                    
        except Exception as e:
            log_node(f"Video Lightning: ❌ Failed to load LoRA: {e}", color="RED")
            return (model_bundle,)

        # --- Build overrides dict ---
        overrides = {
            "cfg": 1.0,
            "steps": step_count,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
        }

        # --- Create patched bundle with overrides ---
        new_bundle = UmeBundle(
            model=model_patched,
            model_low_noise=model_low_noise_patched,
            clip=clip_patched,
            vae=model_bundle.vae,
            model_name=model_bundle.model_name,
            bundle_type=model_bundle.bundle_type,
            loader_type=model_bundle.loader_type,
            shift=model_bundle.shift,
            overrides=overrides,
            clip_vision=getattr(model_bundle, "clip_vision", None),
        )

        log_node(
            f"⚡ Video Lightning: Applied {lora_basename} "
            f"(strength={lora_strength:.2f}) — "
            f"Overrides: CFG→1.0, Steps→{step_count}, "
            f"Sampler→{sampler_name}, Scheduler→{scheduler}",
            color="GREEN"
        )

        return (new_bundle,)
