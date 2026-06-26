"""
UmeAiRT Toolkit - Pack Pipeline (Passthrough)
-----------------------------------------------
Assembles a UME_PIPELINE from individual components WITHOUT sampling.

Designed as a drop-in replacement for the BlockSampler when the goal is
to upscale an existing image rather than generate a new one.  All inputs
are optional except `images` — the node never blocks, and downstream
upscale nodes are responsible for validating what they actually need.
"""

import nodes as comfy_nodes
from .common import GenerationContext, log_node, validate_bundle, resize_tensor
from typing import Tuple, Dict, Any, Optional, List


class UmeAiRT_PackPipeline:
    """Packs an existing image + optional models/settings into a UME_PIPELINE.

    Use this instead of the Image Generator when you already have an image
    and just want to run it through the post-processing chain (upscale,
    face detailer, etc.) without any AI generation step.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("UME_IMAGE", {"tooltip": "Connect an Image Loader or Image Process node. This is the image that will be passed to the pipeline."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for upscale reproducibility. Right-click → 'Randomize' to auto-generate a new seed each run."}),
            },
            "optional": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "Model bundle from a Loader node. Required for UltimateSD Upscale (provides the model/VAE/CLIP for tile sampling)."}),
                "positive": ("POSITIVE", {"forceInput": True, "tooltip": "Positive prompt. Required for UltimateSD Upscale (used for tile conditioning)."}),
                "negative": ("NEGATIVE", {"forceInput": True, "tooltip": "Negative prompt. Optional, used by UltimateSD Upscale."}),
                "settings": ("UME_SETTINGS", {"tooltip": "Generation settings (steps, cfg, sampler, scheduler). Seed from here is overridden by the Pack Pipeline seed. Required for UltimateSD Upscale."}),
                "loras": ("UME_LORA_STACK", {"tooltip": "LoRA stack to apply to the model before upscaling. Only used by UltimateSD Upscale."}),
            }
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "pack"
    CATEGORY = "UmeAiRT/Interop"
    DESCRIPTION = "Pass-through node to package independent components into a UME_PIPELINE for upscale/post-processing."

    def __init__(self):
        self._lora_loader = comfy_nodes.LoraLoader()

    def pack(self,
             images: Dict[str, Any],
             seed: int = 0,
             model_bundle=None,
             positive: Optional[str] = None,
             negative: Optional[str] = None,
             settings=None,
             loras: Optional[List[Tuple[str, float, float]]] = None) -> Tuple[GenerationContext]:

        ctx = GenerationContext()

        # ── 1. Image (always present — required input) ──────────────
        image = images.image
        if image is None:
            raise ValueError("Pack Pipeline: Image bundle contains no image.")
        ctx.source_image = image

        # ── 2. Model Bundle (optional) ──────────────────────────────
        model, clip, vae = None, None, None
        if model_bundle is not None:
            model = model_bundle.model
            clip = model_bundle.clip
            vae = model_bundle.vae
            ctx.model_name = model_bundle.model_name or ""
            log_node(f"Pack Pipeline: Model bundle loaded ({ctx.model_name or 'unnamed'})")
        else:
            log_node("Pack Pipeline: No model bundle — SeedVR2 will work, UltimateSD Upscale will not.", color="YELLOW")

        # ── 3. LoRAs (optional, requires model+clip) ────────────────
        if loras and model and clip:
            loaded_loras_meta = []
            for lora_def in loras:
                name, str_model, str_clip = lora_def
                if name != "None":
                    try:
                        model, clip = self._lora_loader.load_lora(model, clip, name, str_model, str_clip)
                        loaded_loras_meta.append({"name": name, "strength": str_model})
                    except Exception as e:
                        log_node(f"Pack Pipeline LoRA Error ({name}): {e}", color="RED")
            ctx.loras = loaded_loras_meta
        elif loras and (not model or not clip):
            log_node("Pack Pipeline: LoRAs provided but no model bundle — skipping LoRA application.", color="YELLOW")

        ctx.model = model
        ctx.clip = clip
        ctx.vae = vae

        # ── 4. Settings (optional) ──────────────────────────────────
        if settings is not None:
            ctx.width = settings.width
            ctx.height = settings.height
            ctx.steps = settings.steps
            ctx.cfg = settings.cfg
            ctx.sampler_name = settings.sampler_name
            ctx.scheduler = settings.scheduler
            # Settings seed is used as fallback only — Pack Pipeline seed takes priority
            ctx.seed = settings.seed
        else:
            # Use image dimensions as width/height defaults
            B, H, W, C = image.shape
            ctx.width = W
            ctx.height = H
            log_node("Pack Pipeline: No settings — using image dimensions and defaults.", color="YELLOW")

        # ── 4b. Seed override — Pack Pipeline seed always wins ──────
        ctx.seed = seed

        # ── 5. Prompts (optional) ───────────────────────────────────
        ctx.positive_prompt = positive if positive is not None else ""
        ctx.negative_prompt = negative if negative is not None else ""

        # ── 6. Handle auto_resize if set in Image Process ───────────
        if images.auto_resize and settings is not None and image is not None:
            target_h, target_w = settings.height, settings.width
            B, H, W, C = image.shape
            if H != target_h or W != target_w:
                log_node(f"Pack Pipeline: Auto-resizing {W}x{H} → {target_w}x{target_h}")
                image = resize_tensor(image, target_h, target_w, interp_mode="bilinear")

        # ── 7. Denoise from image bundle ────────────────────────────
        ctx.denoise = images.denoise if images.denoise is not None else 1.0

        # ── 8. Set image directly — no sampling ────────────────────
        ctx.image = image

        # ── Summary log ─────────────────────────────────────────────
        B, H, W, C = ctx.image.shape
        parts = [f"Pack Pipeline: Ready | {W}x{H}"]
        if ctx.model is not None:
            parts.append(f"Model: {ctx.model_name or 'loaded'}")
        if ctx.positive_prompt:
            parts.append("Prompt: ✓")
        log_node(" | ".join(parts), color="GREEN")

        return (ctx,)
