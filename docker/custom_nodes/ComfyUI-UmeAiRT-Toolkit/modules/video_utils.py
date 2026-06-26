"""
UmeAiRT Toolkit - Video Utilities
-----------------------------------
Shared helper functions for WAN video generation pipelines.

Used by: wan_sampler.py, wan_extender.py, video_looper.py
"""

import torch
import comfy.utils
import comfy.patcher_extension
from .common import log_node


def patch_wan_model(model, shift, overrides, label=""):
    """Apply ModelSamplingSD3, CFGZeroStar, EasyCache, and NAG patches to a WAN model.

    Args:
        model: The diffusion model to patch.
        shift: ModelSamplingSD3 shift value (0 = skip).
        overrides: Dict of optimization overrides (cfg_zero_star, easy_cache, nag_scale, etc.).
        label: Prefix for log messages (e.g. "High-Noise", "Low-Noise").

    Returns:
        The patched model.
    """
    prefix = f"  [{label}] " if label else "  "

    # ModelSamplingSD3 (shift)
    if shift > 0:
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        model = ModelSamplingSD3().patch(model, shift)[0]
        log_node(f"{prefix}ModelSamplingSD3: shift={shift}", color="GREEN")

    # CFGZeroStar
    cfg_zero_star = overrides.get("cfg_zero_star", True)
    if cfg_zero_star:
        from comfy_extras.nodes_cfg import optimized_scale
        m = model.clone()
        def cfg_zero_star_fn(args):
            guidance_scale = args['cond_scale']
            x = args['input']
            cond_p = args['cond_denoised']
            uncond_p = args['uncond_denoised']
            out = args["denoised"]
            alpha = optimized_scale(x - cond_p, x - uncond_p)
            return out + uncond_p * (alpha - 1.0) + guidance_scale * uncond_p * (1.0 - alpha)
        m.set_model_sampler_post_cfg_function(cfg_zero_star_fn)
        model = m
        log_node(f"{prefix}CFGZeroStar: enabled", color="GREEN")

    # EasyCache
    easy_cache_threshold = overrides.get("easy_cache", 0.0)
    if easy_cache_threshold > 0:
        try:
            from comfy_extras.nodes_easycache import (
                EasyCacheHolder, easycache_sample_wrapper,
                easycache_calc_cond_batch_wrapper, easycache_forward_wrapper
            )
            model = model.clone()
            output_channels = model.model.latent_format.latent_channels
            model.model_options["transformer_options"]["easycache"] = EasyCacheHolder(
                easy_cache_threshold, 0.15, 0.95,
                subsample_factor=8, offload_cache_diff=False,
                verbose=False, output_channels=output_channels
            )
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "easycache", easycache_sample_wrapper)
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, "easycache", easycache_calc_cond_batch_wrapper)
            model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "easycache", easycache_forward_wrapper)
            log_node(f"{prefix}EasyCache: threshold={easy_cache_threshold}", color="GREEN")
        except Exception as e:
            log_node(f"{prefix}EasyCache: failed to apply ({e}), continuing without it.", color="YELLOW")

    # NAG (Normalized Attention Guidance)
    # Reproduces the native ComfyUI NAGuidance logic (comfy_extras/nodes_nag.py)
    nag_scale = overrides.get("nag_scale", 0.0)
    if nag_scale > 0:
        nag_alpha = overrides.get("nag_alpha", 0.5)
        nag_tau = overrides.get("nag_tau", 1.5)

        m = model.clone()

        def nag_attention_output_patch(out, extra_options):
            cond_or_uncond = extra_options.get("cond_or_uncond", None)
            if cond_or_uncond is None:
                return out
            if not (1 in cond_or_uncond and 0 in cond_or_uncond):
                return out
            img_slice = extra_options.get("img_slice", None)
            if img_slice is not None:
                orig_out = out
                out = out[:, img_slice[0]:img_slice[1]]
            batch_size = out.shape[0]
            half_size = batch_size // len(cond_or_uncond)
            ind_neg = cond_or_uncond.index(1)
            ind_pos = cond_or_uncond.index(0)
            z_pos = out[half_size * ind_pos:half_size * (ind_pos + 1)]
            z_neg = out[half_size * ind_neg:half_size * (ind_neg + 1)]
            guided = z_pos * nag_scale - z_neg * (nag_scale - 1.0)
            eps = 1e-6
            norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True).clamp_min(eps)
            norm_guided = torch.norm(guided, p=1, dim=-1, keepdim=True).clamp_min(eps)
            ratio = norm_guided / norm_pos
            scale_factor = torch.minimum(ratio, torch.full_like(ratio, nag_tau)) / ratio
            guided_normalized = guided * scale_factor
            z_final = guided_normalized * nag_alpha + z_pos * (1.0 - nag_alpha)
            if img_slice is not None:
                orig_out[half_size * ind_neg:half_size * (ind_neg + 1), img_slice[0]:img_slice[1]] = z_final
                orig_out[half_size * ind_pos:half_size * (ind_pos + 1), img_slice[0]:img_slice[1]] = z_final
                return orig_out
            else:
                out[half_size * ind_pos:half_size * (ind_pos + 1)] = z_final
            return out

        m.set_model_attn1_output_patch(nag_attention_output_patch)
        m.disable_model_cfg1_optimization()
        model = m
        log_node(f"{prefix}NAG: scale={nag_scale:.1f}, α={nag_alpha:.2f}, τ={nag_tau:.2f}", color="GREEN")

    return model


def apply_color_match(frames, reference_image, width, height):
    """Apply histogram-based color matching to align generated frames with the reference.

    Uses per-channel mean/std transfer (Reinhard color transfer) to correct
    the color drift that VACE generation can introduce.

    Args:
        frames: Generated frames tensor [N, H, W, C].
        reference_image: Reference image tensor [1, H, W, C].
        width: Target width for reference resize.
        height: Target height for reference resize.

    Returns:
        Color-matched frames tensor.
    """
    try:
        ref = comfy.utils.common_upscale(
            reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)[0]  # [H, W, C]

        # Reinhard color transfer: match mean and std per channel
        for c in range(min(3, frames.shape[-1])):
            ref_mean = ref[:, :, c].mean()
            ref_std = ref[:, :, c].std() + 1e-6
            for i in range(frames.shape[0]):
                frame_c = frames[i, :, :, c]
                frame_mean = frame_c.mean()
                frame_std = frame_c.std() + 1e-6
                frames[i, :, :, c] = ((frame_c - frame_mean) / frame_std) * ref_std + ref_mean

        frames = torch.clamp(frames, 0.0, 1.0)
        log_node("  ColorMatch: applied (Reinhard transfer)", color="GREEN")
    except Exception as e:
        log_node(f"  ColorMatch: failed ({e}), skipping.", color="YELLOW")
    return frames
