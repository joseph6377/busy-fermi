"""
UmeAiRT Toolkit - Sampler ControlNet Application
--------------------------------------------------
Handles loading, caching, and applying ControlNet models
to conditioning tensors during the sampling pipeline.
"""

import traceback
from .common import log_node


def apply_controlnets(controlnets, positive_cond, negative_cond,
                      cnet_loader, cnet_apply, controlnet_cache, vae=None, is_flux=False):
    """Load and apply all ControlNet models to the conditioning.

    Each ControlNet definition is a tuple of:
        (name, image, strength, start_percent, end_percent[, union_type])

    Models are cached by name to avoid redundant disk loads across runs.

    Args:
        controlnets: List of ControlNet definition tuples.
        positive_cond: Positive conditioning to modify.
        negative_cond: Negative conditioning to modify.
        cnet_loader: ComfyUI ControlNetLoader instance.
        cnet_apply: ComfyUI ControlNetApplyAdvanced instance.
        controlnet_cache (dict): Shared cache dict {name: loaded_model}.
        vae: Optional VAE model for FLUX ControlNets that encode in latent space.
        is_flux (bool): Whether the parent diffusion model is FLUX-type.

    Returns:
        tuple: (modified_positive_cond, modified_negative_cond)
    """
    if not controlnets:
        return positive_cond, negative_cond

    for cnet_def in controlnets:
        if len(cnet_def) == 6:
            c_name, c_image, c_str, c_start, c_end, c_type = cnet_def
        else:
            c_name, c_image, c_str, c_start, c_end = cnet_def
            c_type = None

        if c_name == "None" or c_image is None:
            continue

        try:
            # Load or retrieve from cache
            if c_name in controlnet_cache:
                c_model = controlnet_cache[c_name]
            else:
                c_model = cnet_loader.load_controlnet(c_name)[0]
                controlnet_cache[c_name] = c_model

            # Handle Union ControlNet types
            if c_type is not None:
                try:
                    if is_flux:
                        # Shakker-Labs FLUX Union Pro type mapping (different from SDXL!)
                        # Ref: https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro
                        FLUX_UNION_TYPES = {
                            "canny/lineart/anime_lineart/mlsd": 0,
                            "tile": 1,
                            "depth": 2,
                            "blur": 3,
                            "openpose": 4,
                            "gray": 5,
                            "low_quality": 6,
                        }
                        type_number = FLUX_UNION_TYPES.get(c_type, -1)
                        source = "FLUX Union"
                    else:
                        # Standard SDXL Union type mapping
                        from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
                        type_number = UNION_CONTROLNET_TYPES.get(c_type, -1)
                        source = "SDXL Union"

                    if type_number >= 0:
                        c_model = c_model.copy()
                        c_model.set_extra_arg("control_type", [type_number])
                    else:
                        log_node(f"Image Generator: ⚠️ '{c_type}' is not supported by {source} model. "
                                 f"ControlNet will be applied without type hint — results may vary.", color="YELLOW")
                except Exception as e:
                    log_node(f"Image Generator: Could not set Union type: {e}", color="YELLOW")

            # Build kwargs — pass VAE for FLUX ControlNets that encode in latent space
            apply_kwargs = {}
            if vae is not None:
                apply_kwargs["vae"] = vae

            positive_cond, negative_cond = cnet_apply.apply_controlnet(
                positive_cond, negative_cond, c_model, c_image, c_str, c_start, c_end,
                **apply_kwargs
            )

            log_node(f"Image Generator: ControlNet '{c_name}' applied (strength={c_str}).", color="GREEN")
        except Exception as e:
            log_node(f"Image Generator ControlNet Error: {e}", color="RED")
            traceback.print_exc()

    return positive_cond, negative_cond
