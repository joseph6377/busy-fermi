"""
UmeAiRT Toolkit - Video Optimization
---------------------------------------
Middleware node that sits on the UME_BUNDLE wire between the WAN Loader
and the Video Generator. Injects optimization settings (EasyCache, CFGZeroStar,
NAG) into the bundle's overrides dict.

The Video Generator reads these overrides and applies the corresponding
native ComfyUI optimizations during sampling.

BlockSwap is NOT included — it is officially deprecated by ComfyUI
(see comfy_extras/nodes_nop.py) as it breaks the native memory management.
"""

from .common import UmeBundle, log_node


class UmeAiRT_VideoOptimization:
    """Video Optimization middleware — toggles EasyCache, CFGZeroStar, and NAG.

    Sits on the UME_BUNDLE wire. The Video Generator reads these settings
    from ``model_bundle.overrides`` and applies them during sampling.

    - **EasyCache**: Caching optimization that skips redundant diffusion steps.
      Set threshold > 0 to enable (0.2 recommended). Higher = more aggressive.
    - **CFGZeroStar**: Improved CFG guidance that reduces artifacts.
      Enabled by default; disable for compatibility debugging.
    - **NAG** (Normalized Attention Guidance): Enables effective negative
      prompts on distilled/Lightning models. Uses the native ComfyUI
      implementation (comfy_extras/nodes_nag.py). Set nag_scale > 0 to enable.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "Model bundle from a WAN Loader or Bundle Auto-Loader node."}),
            },
            "optional": {
                "easy_cache": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": "EasyCache reuse threshold. 0 = disabled, 0.2 = recommended for ~40%% speed boost. Higher = more aggressive caching (faster but lower quality)."
                }),
                "cfg_zero_star": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable CFGZeroStar improved guidance. Reduces artifacts in video generation. Recommended ON."
                }),
                "nag_scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1,
                    "display": "slider",
                    "tooltip": "NAG guidance scale. 0 = disabled, 5.0 = recommended. Higher values push further from the negative prompt. Useful with distilled/Lightning models."
                }),
                "nag_alpha": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "NAG blending factor. 1.0 = full replacement, 0.0 = no effect. Only used when nag_scale > 0."
                }),
                "nag_tau": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 10.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "NAG normalization ceiling. Controls maximum attention scaling. Only used when nag_scale > 0."
                }),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Applies memory, performance, and guidance optimizations for video generation."

    def process(self, model_bundle, easy_cache=0.0, cfg_zero_star=True,
                nag_scale=0.0, nag_alpha=0.5, nag_tau=1.5):
        # Merge with existing overrides (from Lightning node, if chained)
        existing_overrides = getattr(model_bundle, "overrides", None) or {}
        new_overrides = dict(existing_overrides)

        # Inject optimization settings
        new_overrides["easy_cache"] = easy_cache
        new_overrides["cfg_zero_star"] = cfg_zero_star
        new_overrides["nag_scale"] = nag_scale
        new_overrides["nag_alpha"] = nag_alpha
        new_overrides["nag_tau"] = nag_tau

        # Create a new bundle with merged overrides
        new_bundle = UmeBundle(
            model=model_bundle.model,
            model_low_noise=getattr(model_bundle, "model_low_noise", None),
            clip=model_bundle.clip,
            vae=model_bundle.vae,
            model_name=model_bundle.model_name,
            bundle_type=model_bundle.bundle_type,
            loader_type=model_bundle.loader_type,
            shift=model_bundle.shift,
            overrides=new_overrides,
            clip_vision=getattr(model_bundle, "clip_vision", None),
        )

        opts = []
        if easy_cache > 0:
            opts.append(f"EasyCache({easy_cache:.2f})")
        if cfg_zero_star:
            opts.append("CFGZeroStar")
        if nag_scale > 0:
            opts.append(f"NAG(scale={nag_scale:.1f}, α={nag_alpha:.2f}, τ={nag_tau:.2f})")

        if opts:
            log_node(f"🔧 Video Optimization: {', '.join(opts)}", color="GREEN")
        else:
            log_node("🔧 Video Optimization: all optimizations disabled", color="YELLOW")

        return (new_bundle,)

