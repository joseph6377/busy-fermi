"""ComfyUI node definition for Lightning acceleration LoRA loading.

Sits between the Bundle Loader and the Image Generator on the UME_BUNDLE wire.
Automatically loads the correct Lightning LoRA for the selected QWEN model
variant and embeds settings overrides (cfg, steps, sampler, scheduler) into
the bundle's ``overrides`` dict.  The BlockSampler reads these overrides and
silently applies them on top of the user's GenerationSettings.

The LoRA mapping is driven by the ``_ACCELERATION_LORAS`` section in the
model manifest — prefixed with ``_`` so it never appears in the
Bundle Auto-Loader dropdown.
"""

import os
import folder_paths
import nodes as comfy_nodes
import comfy.samplers
from .common import UmeBundle, log_node
from .manifest import load_manifest, download_bundle_files


# ---------------------------------------------------------------------------
# Internal mapping: (model_variant_key, step_count) → LoRA filename
# Each entry maps a QWEN model variant to its best Lightning LoRA file.
# The keys are lowercase substring patterns matched against model_name.
# Order matters: more specific patterns must come first.
# ---------------------------------------------------------------------------

_QWEN_LORA_MAP = {
    # Image_Edit_2509 variants (must be checked before generic "image-edit")
    ("image-edit-2509", 4): "QWEN/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32.safetensors",
    ("image-edit-2509", 8): "QWEN/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors",
    # Image_Edit variants
    ("image-edit", 4): "QWEN/Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
    ("image-edit", 8): "QWEN/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors",
    # Image (generation) — fallback
    ("image", 4): "QWEN/Qwen-Image-Lightning-4steps-V2.0.safetensors",
    ("image", 8): "QWEN/Qwen-Image-Lightning-8steps-V2.0.safetensors",
}

# Ordered variant patterns for matching (most specific first)
_QWEN_VARIANT_PATTERNS = [
    "image-edit-2509",  # Must match before "image-edit"
    "image-edit",       # Must match before "image"
    "image",            # Fallback for all other QWEN models
]


def _resolve_qwen_lora(model_name, step_count):
    """Determine the Lightning LoRA filename for a QWEN model.

    Args:
        model_name (str): The model filename from the bundle.
        step_count (int): 4 or 8.

    Returns:
        str or None: The LoRA relative path (e.g. "QWEN/Qwen-Image-..."),
                     or None if no match found.
    """
    name_lower = model_name.lower().replace("_", "-")
    for pattern in _QWEN_VARIANT_PATTERNS:
        if pattern in name_lower:
            return _QWEN_LORA_MAP.get((pattern, step_count))
    return None


def _try_manifest_download(lora_filename):
    """Attempt to download a Lightning LoRA from the manifest if not present locally.

    Looks up the ``_ACCELERATION_LORAS`` section of the model manifest.

    Args:
        lora_filename (str): The LoRA filename to look for (e.g.
            "QWEN/Qwen-Image-Lightning-4steps-V2.0.safetensors").

    Returns:
        bool: True if the file is now available locally, False otherwise.
    """
    basename = os.path.basename(lora_filename)

    # First check if already present
    existing = folder_paths.get_full_path("loras", basename)
    if existing and os.path.exists(existing):
        return True

    # Also check with subfolder path
    existing = folder_paths.get_full_path("loras", lora_filename)
    if existing and os.path.exists(existing):
        return True

    # Try downloading from manifest
    try:
        data = load_manifest()
        accel_section = data.get("_ACCELERATION_LORAS", {})

        # Search all families for a matching file
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
                        # Found it — download via the standard bundle mechanism
                        log_node(f"Lightning: ⬇️ Downloading {basename}...", color="CYAN")
                        download_bundle_files(f"_ACCELERATION_LORAS/{family_key}", variant_key)
                        # Verify it's now available
                        check = folder_paths.get_full_path("loras", basename)
                        if check and os.path.exists(check):
                            return True
                        check = folder_paths.get_full_path("loras", lora_filename)
                        return check is not None and os.path.exists(check)
    except Exception as e:
        log_node(f"Lightning: Manifest download failed: {e}", color="YELLOW")

    return False


class UmeAiRT_LightningAccelerator:
    """Lightning Acceleration LoRA applicator for QWEN models.

    Sits on the UME_BUNDLE wire between a Loader and the Image Generator.
    Loads the correct Lightning LoRA (4 or 8 steps) based on the selected
    QWEN model variant, applies it to the model, and embeds settings
    overrides (cfg, steps, sampler, scheduler) into the bundle.

    The BlockSampler reads ``model_bundle.overrides`` and silently applies
    them on top of the user's GenerationSettings — the user does not need
    to change their Settings node.

    When set to "Off", acts as a pure pass-through.
    When connected to a non-QWEN model, logs a warning and passes through.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "Model bundle from a Loader or Bundle Auto-Loader node."}),
                "mode": (["Off", "4 Steps", "8 Steps", "Turbo (Anima)"], {"default": "Off", "tooltip": "Lightning acceleration mode. Off = pass-through, 4/8 Steps = QWEN Lightning LoRA, Turbo (Anima) = Anima Turbo LoRA."}),
            },
            "optional": {
                "lora_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "display": "slider", "advanced": True,
                    "tooltip": "Strength of the Lightning LoRA on the model. Default 1.0 is recommended."
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler", "advanced": True,
                    "tooltip": "Sampler override when Lightning is active. Euler is recommended for Lightning LoRAs."
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "sgm_uniform", "advanced": True,
                    "tooltip": "Scheduler override when Lightning is active. SGM Uniform is recommended for Lightning LoRAs."
                }),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = "Applies lightning acceleration overrides for QWEN models to speed up generation."

    def __init__(self):
        self._lora_loader = comfy_nodes.LoraLoader()

    def process(self, model_bundle, mode="Off",
                lora_strength=1.0, sampler_name="euler", scheduler="sgm_uniform"):
        # --- Pass-through when Off ---
        if mode == "Off":
            return (model_bundle,)

        step_count = 4 if mode == "4 Steps" else 8
        loader_type = getattr(model_bundle, "loader_type", "")

        # --- Guard: check supported models ---
        if mode in ["4 Steps", "8 Steps"] and loader_type != "qwen":
            log_node(
                f"Lightning: ⚠️ Non-QWEN model detected (loader_type='{loader_type}'). "
                f"Lightning 4/8 Steps are only available for QWEN models. Passing through.",
                color="YELLOW"
            )
            return (model_bundle,)
            
        if mode == "Turbo (Anima)" and loader_type != "anima":
            log_node(
                f"Lightning: ⚠️ Non-Anima model detected (loader_type='{loader_type}'). "
                f"Turbo (Anima) is only available for Anima models. Passing through.",
                color="YELLOW"
            )
            return (model_bundle,)

        # --- Resolve the correct LoRA file ---
        model_name = getattr(model_bundle, "model_name", "")
        if mode == "Turbo (Anima)":
            lora_rel_path = "Anima/anima-turbo-lora-v0.2.safetensors"
        else:
            lora_rel_path = _resolve_qwen_lora(model_name, step_count)

        if not lora_rel_path:
            log_node(
                f"Lightning: ⚠️ No Lightning LoRA mapping found for model '{model_name}'. Passing through.",
                color="YELLOW"
            )
            return (model_bundle,)

        lora_basename = os.path.basename(lora_rel_path)

        # --- Check local availability, try auto-download if missing ---
        lora_path = folder_paths.get_full_path("loras", lora_basename)
        if not lora_path or not os.path.exists(lora_path):
            # Try with subfolder
            lora_path = folder_paths.get_full_path("loras", lora_rel_path)

        if not lora_path or not os.path.exists(lora_path):
            # Attempt manifest download
            if _try_manifest_download(lora_rel_path):
                lora_path = folder_paths.get_full_path("loras", lora_basename)
                if not lora_path:
                    lora_path = folder_paths.get_full_path("loras", lora_rel_path)

        if not lora_path or not os.path.exists(lora_path):
            log_node(
                f"Lightning: ❌ LoRA file not found: '{lora_basename}'. "
                f"Please place it in your ComfyUI/models/loras/QWEN/ folder. Passing through.",
                color="RED"
            )
            return (model_bundle,)

        # --- Apply the LoRA to the model ---
        model = model_bundle.model
        clip = model_bundle.clip

        if not model or not clip:
            log_node("Lightning: ❌ Bundle is missing model or CLIP. Cannot apply LoRA.", color="RED")
            return (model_bundle,)

        try:
            lora_name = lora_basename
            model_patched, clip_patched = self._lora_loader.load_lora(
                model, clip, lora_name, lora_strength, lora_strength
            )
        except Exception as e:
            log_node(f"Lightning: ❌ Failed to load LoRA '{lora_basename}': {e}", color="RED")
            return (model_bundle,)

        # --- Build overrides dict ---
        if mode == "Turbo (Anima)":
            overrides = {
                "cfg": 1.0,
                # Note: No steps override for Anima Turbo, user keeps control
                "sampler_name": sampler_name,
                "scheduler": scheduler,
            }
            msg = (
                f"⚡ Lightning: Applied {lora_basename} "
                f"(strength={lora_strength:.2f}) — "
                f"Overrides: CFG→1.0, Sampler→{sampler_name}, Scheduler→{scheduler}"
            )
        else:
            overrides = {
                "cfg": 1.0,
                "steps": step_count,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
            }
            msg = (
                f"⚡ Lightning: Applied {lora_basename} "
                f"(strength={lora_strength:.2f}) — "
                f"Overrides: CFG→1.0, Steps→{step_count}, "
                f"Sampler→{sampler_name}, Scheduler→{scheduler}"
            )

        # --- Create patched bundle with overrides ---
        new_bundle = UmeBundle(
            model=model_patched,
            clip=clip_patched,
            vae=model_bundle.vae,
            model_name=model_bundle.model_name,
            bundle_type=model_bundle.bundle_type,
            loader_type=model_bundle.loader_type,
            shift=model_bundle.shift,
            overrides=overrides,
        )

        log_node(msg, color="GREEN")

        return (new_bundle,)
