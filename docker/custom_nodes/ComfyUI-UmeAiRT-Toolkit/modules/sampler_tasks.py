"""
UmeAiRT Toolkit - Sampler Task Preparation
--------------------------------------------
Mode-specific image preparation functions for the sampling pipeline.
Each function handles one generation mode (txt2img, img2img, inpaint, outpaint)
and returns a standardized ImagePrepResult.
"""

import torch
import torchvision.transforms.functional as TF
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
from .common import resize_tensor, apply_outpaint_padding, log_node


@dataclass
class ImagePrepResult:
    """Standardized output from image preparation functions.

    Attributes:
        raw_image: Prepared source image tensor (or None for txt2img).
        source_mask: Mask tensor for inpaint/outpaint (or None).
        mode_str: Final mode string after preparation.
        is_outpaint: Whether this is an outpaint operation.
        denoise: Effective denoise strength.
        flux_fill_info: Dict with FLUX Fill-specific data if applicable, else None.
    """
    raw_image: Any = None
    source_mask: Any = None
    mode_str: str = "txt2img"
    is_outpaint: bool = False
    denoise: float = 1.0
    flux_fill_info: Optional[dict] = None


def prepare_txt2img(width, height, model, denoise):
    """Prepare an empty latent for text-to-image generation.

    Auto-detects latent channels from the model (SD=4, FLUX=16).

    Args:
        width: Target image width.
        height: Target image height.
        model: The diffusion model (used to detect latent_channels).
        denoise: Denoise strength (forced to 1.0 for txt2img).

    Returns:
        dict: Latent image dict with 'samples' key.
    """
    latent_channels = 4
    try:
        latent_channels = model.model.latent_format.latent_channels
    except Exception as e:
        log_node(f"Image Generator: Could not detect latent channels, defaulting to 4: {e}", color="YELLOW")

    latent = torch.zeros([1, latent_channels, height // 8, width // 8], device="cpu")
    return {"samples": latent}


def _auto_resize(raw_image, source_mask, height, width, auto_resize):
    """Resize source image and mask to target settings dimensions.

    Args:
        raw_image: Source image tensor [B, H, W, C].
        source_mask: Mask tensor or None.
        height: Target height.
        width: Target width.
        auto_resize: Whether to perform auto-resize.

    Returns:
        tuple: (raw_image, source_mask) potentially resized.
    """
    if not auto_resize or raw_image is None:
        return raw_image, source_mask

    raw_image = resize_tensor(raw_image, height, width, interp_mode="bilinear")
    if source_mask is not None:
        source_mask = resize_tensor(source_mask, height, width, interp_mode="nearest", is_mask=True)

    return raw_image, source_mask


def _align_mask_to_image(raw_image, source_mask):
    """Ensure mask dimensions match image dimensions.

    LoadImage returns 64x64 masks for RGB images without alpha,
    so we resize the mask to match.

    Args:
        raw_image: Source image tensor [B, H, W, C].
        source_mask: Mask tensor or None.

    Returns:
        Mask tensor resized to match image, or None.
    """
    if raw_image is None or source_mask is None:
        return source_mask

    H, W = raw_image.shape[1], raw_image.shape[2]
    if source_mask.shape[-2] != H or source_mask.shape[-1] != W:
        source_mask = resize_tensor(source_mask, H, W, interp_mode="nearest", is_mask=True)

    return source_mask


def prepare_img2img(images, height, width, vae_encode_fn):
    """Prepare source image for img2img generation.

    Args:
        images: UME_IMAGE input bundle.
        height: Target height from settings.
        width: Target width from settings.
        vae_encode_fn: Callable(vae, image) -> (latent_dict,) for encoding.

    Returns:
        ImagePrepResult with raw_image and latent info.
    """
    raw_image = images.image
    source_mask = images.mask
    auto_resize = images.auto_resize
    denoise = images.denoise

    raw_image, source_mask = _auto_resize(raw_image, source_mask, height, width, auto_resize)
    source_mask = _align_mask_to_image(raw_image, source_mask)

    return ImagePrepResult(
        raw_image=raw_image,
        source_mask=source_mask,
        mode_str="img2img",
        denoise=denoise,
    )


def prepare_inpaint(images, height, width, model_bundle):
    """Prepare source image and mask for inpainting.

    Detects FLUX Fill models for specialized pipeline delegation.

    Args:
        images: UME_IMAGE input bundle.
        height: Target height from settings.
        width: Target width from settings.
        model_bundle: UME_BUNDLE for model detection.

    Returns:
        ImagePrepResult with raw_image, source_mask, and flux_fill detection.
    """
    raw_image = images.image
    source_mask = images.mask
    auto_resize = images.auto_resize
    denoise = images.denoise

    raw_image, source_mask = _auto_resize(raw_image, source_mask, height, width, auto_resize)
    source_mask = _align_mask_to_image(raw_image, source_mask)

    # Detect FLUX Fill model
    is_flux_fill = _detect_flux_fill(model_bundle)

    return ImagePrepResult(
        raw_image=raw_image,
        source_mask=source_mask,
        mode_str="inpaint",
        denoise=denoise,
        flux_fill_info={"is_flux_fill": is_flux_fill} if is_flux_fill else None,
    )


def prepare_outpaint(images, height, width, model_bundle):
    """Prepare source image for outpainting with padding and mask generation.

    Handles alignment-based padding, aspect ratio preservation,
    mask blur, and FLUX Fill delegation.

    Args:
        images: UME_IMAGE input bundle.
        height: Target height from settings.
        width: Target width from settings.
        model_bundle: UME_BUNDLE for model detection.

    Returns:
        ImagePrepResult with padded image, generated mask, and outpaint metadata.
    """
    raw_image = images.image
    source_mask = images.mask
    auto_resize = images.auto_resize
    denoise = images.denoise

    raw_image, source_mask = _auto_resize(raw_image, source_mask, height, width, auto_resize)
    source_mask = _align_mask_to_image(raw_image, source_mask)

    if raw_image is None:
        return ImagePrepResult(mode_str="txt2img", denoise=1.0)

    target_w = images.outpaint_target_w
    target_h = images.outpaint_target_h
    h_align = images.outpaint_h_align
    v_align = images.outpaint_v_align
    mask_blur = images.outpaint_mask_blur

    B, src_h, src_w, C = raw_image.shape

    # Resize source to fit within target (maintain aspect ratio)
    if src_w > target_w or src_h > target_h:
        scale = min(target_w / src_w, target_h / src_h)
        fit_w = int(src_w * scale)
        fit_h = int(src_h * scale)
        raw_image = resize_tensor(raw_image, fit_h, fit_w, interp_mode="bilinear")
        if source_mask is not None:
            source_mask = resize_tensor(source_mask, fit_h, fit_w, interp_mode="nearest", is_mask=True)
        src_w, src_h = fit_w, fit_h

    # Compute padding from alignment
    pad_l, pad_r = _compute_padding(target_w - src_w, h_align, "left", "right")
    pad_t, pad_b = _compute_padding(target_h - src_h, v_align, "top", "bottom")

    # Removed forced denoise=1.0. We will now use mirror padding as a structural base,
    # which allows FLUX Fill to use denoise < 1.0 (e.g. 0.85) for much higher quality
    # outpaint generation without the raw TV static grain.

    # Detect FLUX Fill model
    is_flux_fill = _detect_flux_fill(model_bundle)

    if is_flux_fill:
        # FLUX Fill: store info for flux_sampler delegation
        return ImagePrepResult(
            raw_image=raw_image,
            source_mask=source_mask,
            mode_str="inpaint",
            is_outpaint=True,
            denoise=denoise,
            flux_fill_info={
                "is_flux_fill": True,
                "raw_image": raw_image.clone(),
                "pad_info": (pad_l, pad_t, pad_r, pad_b),
                "feathering": mask_blur if mask_blur > 0 else 80,
            },
        )
    else:
        # Standard outpaint: apply padding directly
        raw_image, source_mask = apply_outpaint_padding(
            raw_image, source_mask, pad_l, pad_t, pad_r, pad_b,
            overlap=48, feathering=48, skip_noise=False)

        if source_mask is not None and mask_blur > 0:
            source_mask = _apply_mask_blur(source_mask, mask_blur)

        return ImagePrepResult(
            raw_image=raw_image,
            source_mask=source_mask,
            mode_str="inpaint",
            is_outpaint=True,
            denoise=denoise,
        )


def composite_inpaint(image_out, raw_image, source_mask):
    """Blend sampled output with source image using the inpaint mask.

    Only the masked region is replaced; the unmasked area retains the
    original source image pixels.

    Args:
        image_out: Decoded output image tensor [B, H, W, C].
        raw_image: Original source image tensor.
        source_mask: Inpaint mask tensor.

    Returns:
        Blended image tensor [B, H, W, C].
    """
    try:
        B, H, W, C = image_out.shape
        source_resized = resize_tensor(raw_image, H, W, interp_mode="bilinear")
        mask_resized = resize_tensor(source_mask, H, W, interp_mode="bilinear", is_mask=True)
        m = mask_resized
        if len(m.shape) == 2:
            m = m.unsqueeze(0).unsqueeze(-1)
        elif len(m.shape) == 3:
            m = m.unsqueeze(-1)
        if m.shape[0] < B:
            m = m.repeat(B, 1, 1, 1)
        if source_resized.shape[0] < B:
            source_resized = source_resized.repeat(B, 1, 1, 1)
        m = torch.clamp(m, 0.0, 1.0)
        image_out = source_resized * (1.0 - m) + image_out * m
        log_node("Image Generator Inpaint: Auto-Composited.", color="GREEN")
    except Exception as e:
        log_node(f"Image Generator Inpaint Composite Failed: {e}", color="RED")
    return image_out


# --- Private helpers ---

def _detect_flux_fill(model_bundle):
    """Detect if the model bundle is a FLUX Fill (inpaint) model."""
    if getattr(model_bundle, "loader_type", "") != "flux":
        return False
    b_type = getattr(model_bundle, "bundle_type", "").lower()
    cat = getattr(model_bundle, "category", "").lower()
    return "inpaint" in b_type or "fill" in b_type or "fill" in cat


def _compute_padding(total_pad, align, side_start, side_end):
    """Compute left/right or top/bottom padding based on alignment.

    Args:
        total_pad: Total padding to distribute (may be negative → clamped to 0).
        align: Alignment string (side_start, side_end, or "center").
        side_start: Name for the start side (e.g. "left", "top").
        side_end: Name for the end side (e.g. "right", "bottom").

    Returns:
        tuple: (pad_start, pad_end)
    """
    total_pad = max(0, total_pad)
    if align == side_start:
        return 0, total_pad
    elif align == side_end:
        return total_pad, 0
    else:  # center
        pad_start = total_pad // 2
        return pad_start, total_pad - pad_start


def _apply_mask_blur(source_mask, mask_blur):
    """Apply Gaussian blur to an inpaint/outpaint mask.

    Args:
        source_mask: Mask tensor (2D, 3D, or 4D).
        mask_blur: Blur kernel size (must be odd, auto-corrected).

    Returns:
        Blurred mask tensor with same shape as input.
    """
    if len(source_mask.shape) == 2:
        m = source_mask.unsqueeze(0).unsqueeze(0)
    else:
        m = source_mask
    k = mask_blur if mask_blur % 2 == 1 else mask_blur + 1
    m = TF.gaussian_blur(m, kernel_size=k)
    if len(source_mask.shape) == 2:
        source_mask = m.squeeze(0).squeeze(0)
    else:
        source_mask = m
    return source_mask
