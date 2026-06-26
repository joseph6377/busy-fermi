"""
FLUX-specific sampling pipeline.

Uses the ACTUAL ComfyUI core nodes directly — no replication, no approximation.
Chain: ImagePadForOutpaint → InpaintModelConditioning → DifferentialDiffusion
      → BasicGuider → BasicScheduler → SamplerCustomAdvanced → VAEDecode
"""

import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import node_helpers
from .common import log_node

# Import the REAL ComfyUI nodes
try:
    from nodes import ImagePadForOutpaint as _PadNode
    from nodes import InpaintModelConditioning as _IMCNode
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False

try:
    from comfy_extras.nodes_custom_sampler import (
        Guider_Basic,
        BasicScheduler,
        KSamplerSelect,
        Noise_RandomNoise,
    )
    from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
    _HAS_SAMPLER = True
except ImportError:
    _HAS_SAMPLER = False


def is_flux_model(model):
    """Detect if a model is FLUX-type (16 latent channels, guidance-based).

    Args:
        model: The diffusion model to check.

    Returns:
        bool: True if the model uses 16 latent channels (FLUX signature).
    """
    try:
        return model.model.latent_format.latent_channels == 16
    except Exception:  # Duck-type check: missing attrs → not FLUX
        return False


def apply_flux_guidance(positive_cond, negative_cond, cfg):
    """Apply FLUX guidance embedding and override CFG.

    FLUX models use guidance values embedded in the conditioning
    rather than the standard CFG scale multiplier. The sampler
    CFG is forced to 1.0 so that only the embedded guidance
    controls the generation strength.

    Args:
        positive_cond: Positive conditioning.
        negative_cond: Negative conditioning.
        cfg: User-specified guidance value.

    Returns:
        tuple: (positive_cond, negative_cond, sampler_cfg=1.0)
    """
    positive_cond = node_helpers.conditioning_set_values(positive_cond, {"guidance": cfg})
    negative_cond = node_helpers.conditioning_set_values(negative_cond, {"guidance": cfg})
    log_node(f"Image Generator: FLUX Guidance set to {cfg}, Sampler CFG forced to 1.0", color="CYAN")
    return positive_cond, negative_cond, 1.0


def sample_flux_base(model, vae, positive_cond,
                     seed, steps, guidance, sampler_name, scheduler,
                     latent_image, denoise, vae_decode_fn):
    """Standard FLUX sampling using BasicGuider + SamplerCustomAdvanced.

    Uses the native FLUX pipeline (positive-only conditioning via BasicGuider)
    instead of KSampler's CFGGuider. Required for ControlNetFlux which doesn't
    work correctly through CFGGuider's dual positive/negative path.

    Args:
        model: The FLUX diffusion model.
        vae: VAE model for decode.
        positive_cond: Pre-encoded positive conditioning (with ControlNet already applied).
        seed: Random seed.
        steps: Number of sampling steps.
        guidance: FLUX guidance value (embedded in conditioning).
        sampler_name: Sampler algorithm name.
        scheduler: Noise scheduler name.
        latent_image: Starting latent dict (empty for txt2img, encoded for img2img).
        denoise: Denoise strength.
        vae_decode_fn: Callable(vae, latent_dict) -> image_tensor.

    Returns:
        tuple: (final_image, result_latent)
    """
    if not _HAS_SAMPLER:
        raise RuntimeError("flux_sampler: Required ComfyUI custom sampler nodes not available.")

    # Apply FLUX guidance (embedded in conditioning, not CFG scale)
    positive_guided = node_helpers.conditioning_set_values(positive_cond, {"guidance": guidance})

    # BasicGuider — positive-only conditioning (the proper FLUX way)
    guider = Guider_Basic(model)
    guider.set_conds(positive_guided)

    # Sampler & Scheduler
    sampler_obj = KSamplerSelect.execute(sampler_name)[0]
    sigmas = BasicScheduler.execute(model, scheduler, steps, denoise)[0]

    # Noise & Sampling
    noise_obj = Noise_RandomNoise(seed)

    latent = latent_image.copy()
    latent_img = latent["samples"]
    latent_img = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_img)
    latent["samples"] = latent_img

    noise_mask = latent.get("noise_mask", None)

    import latent_preview
    x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    log_node(f"FLUX Sampler: Sampling {steps} steps (guidance={guidance}, denoise={denoise})...", color="CYAN")
    samples = guider.sample(
        noise_obj.generate_noise(latent),
        latent_img,
        sampler_obj,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=noise_obj.seed
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    result_latent = latent.copy()
    result_latent["samples"] = samples

    # VAE Decode
    log_node("FLUX Sampler: Decoding VAE...", color="CYAN")
    decoded_image = vae_decode_fn(vae, result_latent)

    log_node("FLUX Sampler: ✅ Standard Sampling Complete", color="GREEN")
    return decoded_image, result_latent


def sample_flux_outpaint(model, vae, clip, positive_cond, negative_cond,
                         raw_image, pad_l, pad_t, pad_r, pad_b,
                         seed, steps, guidance, sampler_name, scheduler,
                         denoise, feathering, vae_decode_fn):
    """Full FLUX Fill outpaint pipeline using real ComfyUI nodes.
    
    Returns (final_image, result_latent)
    """
    if not _HAS_CORE or not _HAS_SAMPLER:
        raise RuntimeError("flux_sampler: Required ComfyUI nodes not available")

    # Removed forced denoise=1.0 to allow denoise < 1.0 for mirror padding.
    if denoise < 1.0:
        log_node(f"FLUX Sampler: Using structural base with denoise {denoise:.2f}", color="GREEN")

    # === Step 1: Padding ===
    # We use UmeAiRT's mirror padding instead of ImagePadForOutpaint (which uses flat gray).
    # This provides a structural base so FLUX Fill can use denoise < 1.0 without generating grain!
    log_node("FLUX Sampler: Applying Mirror Padding...", color="CYAN")
    from .common import apply_outpaint_padding
    padded_image, outpaint_mask = apply_outpaint_padding(
        raw_image, None, pad_l, pad_t, pad_r, pad_b,
        overlap=feathering, feathering=feathering, skip_noise=True, sharp_mirror=True
    )
    # apply_outpaint_padding returns [B, H, W], which is exactly what ImagePadForOutpaint returns
    # when B=1 (i.e. [1, H, W]).

    log_node(f"FLUX Sampler: Padded {raw_image.shape} → {padded_image.shape}, mask={outpaint_mask.shape}", color="CYAN")

    # === Step 2: InpaintModelConditioning (REAL node) ===
    log_node("FLUX Sampler: InpaintModelConditioning...", color="CYAN")
    imc_node = _IMCNode()
    imc_pos, imc_neg, imc_latent = imc_node.encode(
        positive_cond, negative_cond, padded_image, vae, outpaint_mask, noise_mask=True)
    log_node(f"FLUX Sampler: Latent samples={imc_latent['samples'].shape}, noise_mask={'yes' if 'noise_mask' in imc_latent else 'no'}", color="CYAN")

    # === Step 3: Apply FLUX Guidance ===
    imc_pos = node_helpers.conditioning_set_values(imc_pos, {"guidance": guidance})

    # === Step 4: DifferentialDiffusion (REAL node) ===
    log_node("FLUX Sampler: DifferentialDiffusion...", color="CYAN")
    diff_model = DifferentialDiffusion.execute(model)[0]

    # === Step 5: BasicGuider ===
    guider = Guider_Basic(diff_model)
    guider.set_conds(imc_pos)

    # === Step 6: Sampler & Scheduler ===
    sampler_obj = KSamplerSelect.execute(sampler_name)[0]
    sigmas = BasicScheduler.execute(diff_model, scheduler, steps, denoise)[0]

    # === Step 7: Noise & Sampling ===
    noise_obj = Noise_RandomNoise(seed)
    
    latent = imc_latent.copy()
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
    latent["samples"] = latent_image

    noise_mask = latent.get("noise_mask", None)

    import latent_preview
    x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    log_node("FLUX Sampler: Sampling...", color="CYAN")
    samples = guider.sample(
        noise_obj.generate_noise(latent),
        latent_image,
        sampler_obj,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=noise_obj.seed
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    result_latent = latent.copy()
    result_latent["samples"] = samples

    # === Step 10: VAE Decode ===
    log_node("FLUX Sampler: Decoding VAE...", color="CYAN")
    decoded_image = vae_decode_fn(vae, result_latent)

    # === Step 11: ImageCompositeMasked ===
    # Restore pristine original pixels in the non-outpaint zone.
    # VAE encode→decode loses quality — compositing pastes the sharp original
    # back where mask=0 (original area) and keeps generated pixels where mask=1.
    log_node("FLUX Sampler: Compositing...", color="CYAN")
    try:
        from comfy_extras.nodes_mask import ImageCompositeMasked as _CompNode
        comp_result = _CompNode.execute(
            destination=padded_image,
            source=decoded_image,
            x=0, y=0,
            resize_source=False,
            mask=outpaint_mask
        )
        final_image = comp_result[0]
    except Exception as e:
        log_node(f"FLUX Sampler: ImageCompositeMasked failed ({e}), using raw decode", color="YELLOW")
        final_image = decoded_image

    log_node("FLUX Sampler: ✅ Outpaint Complete", color="GREEN")
    return final_image, result_latent


def sample_flux_inpaint(model, vae, clip, positive_cond, negative_cond,
                        raw_image, mask, seed, steps, guidance,
                        sampler_name, scheduler, denoise, vae_decode_fn):
    """Full FLUX Fill inpaint pipeline using real ComfyUI nodes.

    Unlike outpaint, this does NOT pad the image — the mask defines
    the inpaint region directly on the source image.

    Returns (final_image, result_latent)
    """
    if not _HAS_CORE or not _HAS_SAMPLER:
        raise RuntimeError("flux_sampler: Required ComfyUI nodes not available")

    # Force denoise=1.0 for FLUX Fill
    if denoise < 1.0:
        log_node(f"FLUX Sampler Inpaint: Forcing denoise {denoise:.2f} → 1.0", color="YELLOW")
        denoise = 1.0

    # === Step 1: InpaintModelConditioning (REAL node) ===
    log_node("FLUX Sampler Inpaint: InpaintModelConditioning...", color="CYAN")
    imc_node = _IMCNode()
    imc_pos, imc_neg, imc_latent = imc_node.encode(
        positive_cond, negative_cond, raw_image, vae, mask, noise_mask=True)
    log_node(f"FLUX Sampler Inpaint: Latent samples={imc_latent['samples'].shape}, "
             f"noise_mask={'yes' if 'noise_mask' in imc_latent else 'no'}", color="CYAN")

    # === Step 2: Apply FLUX Guidance ===
    imc_pos = node_helpers.conditioning_set_values(imc_pos, {"guidance": guidance})

    # === Step 3: DifferentialDiffusion (REAL node) ===
    log_node("FLUX Sampler Inpaint: DifferentialDiffusion...", color="CYAN")
    diff_model = DifferentialDiffusion.execute(model)[0]

    # === Step 4: BasicGuider ===
    guider = Guider_Basic(diff_model)
    guider.set_conds(imc_pos)

    # === Step 5: BasicScheduler ===
    sigmas = BasicScheduler.execute(diff_model, scheduler, steps, denoise)[0]

    # === Step 6: KSamplerSelect ===
    sampler_obj = KSamplerSelect.execute(sampler_name)[0]

    # === Step 7: RandomNoise ===
    noise_obj = Noise_RandomNoise(seed)

    # === Step 8: SamplerCustomAdvanced (inline) ===
    log_node(f"FLUX Sampler Inpaint: Sampling {steps} steps (guidance={guidance})", color="CYAN")

    latent = imc_latent.copy()
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
    latent["samples"] = latent_image

    noise_mask = latent.get("noise_mask", None)

    import latent_preview
    x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(
        noise_obj.generate_noise(latent),
        latent_image,
        sampler_obj,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=noise_obj.seed
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    result_latent = latent.copy()
    result_latent["samples"] = samples

    # === Step 9: VAE Decode ===
    log_node("FLUX Sampler Inpaint: Decoding VAE...", color="CYAN")
    decoded_image = vae_decode_fn(vae, result_latent)

    # No compositing needed here — block_sampler handles the inpaint composite
    # via composite_inpaint() which blends using the mask

    log_node("FLUX Sampler: ✅ Inpaint Complete", color="GREEN")
    return decoded_image, result_latent


def is_kontext_model(model_name):
    """Detect if a model filename indicates a FLUX Kontext model.

    Args:
        model_name (str): The model filename or name string.

    Returns:
        bool: True if "kontext" appears in the filename (case-insensitive).
    """
    return "kontext" in (model_name or "").lower()


# Dynamic imports for Kontext are handled inside sample_flux_kontext()


def sample_flux_kontext(model, vae, clip, positive_cond,
                        raw_image, reference_image,
                        seed, steps, guidance,
                        sampler_name, scheduler,
                        width, height, use_custom_size, vae_decode_fn):
    """Full FLUX Kontext image editing pipeline using real ComfyUI nodes.

    Chain: ImageStitch → FluxKontextImageScale → VAEEncode
           → ReferenceLatent → BasicGuider → SamplerCustomAdvanced → VAEDecode

    Unlike FLUX Fill, Kontext does NOT use InpaintModelConditioning or
    DifferentialDiffusion. It injects reference images via ReferenceLatent
    into the conditioning, and the model edits based on the text prompt.

    Args:
        model: The FLUX Kontext diffusion model.
        vae: VAE model for encode/decode.
        clip: Not used directly (conditioning is pre-encoded).
        positive_cond: Pre-encoded positive conditioning.
        raw_image: Primary source image tensor [B, H, W, C].
        reference_image: Optional second reference image tensor, or None.
        seed: Random seed for noise generation.
        steps: Number of sampling steps.
        guidance: FLUX guidance value (embedded in conditioning).
        sampler_name: Name of the sampler algorithm.
        scheduler: Name of the noise scheduler.
        width: Target width from generation settings.
        height: Target height from generation settings.
        use_custom_size: If True, use settings dimensions; if False, use source image dims.
        vae_decode_fn: Callable(vae, latent_dict) -> image_tensor.

    Returns:
        tuple: (final_image, result_latent)
    """
    if not _HAS_SAMPLER:
        raise RuntimeError("flux_sampler: Required ComfyUI custom sampler nodes not available.")

    try:
        from comfy_extras.nodes_images import ImageStitch as _ImageStitch
        from comfy_extras.nodes_flux import FluxKontextImageScale as _KontextScale
        from comfy_extras.nodes_sd3 import EmptySD3LatentImage as _EmptySD3Latent
        from comfy_extras.nodes_edit_model import ReferenceLatent as _ReferenceLatent
    except ImportError as e:
        raise RuntimeError(f"flux_sampler: Required Kontext ComfyUI nodes not available. Update ComfyUI. Details: {e}")

    # === Step 1: ImageStitch (combine 1-2 images) ===
    log_node("FLUX Kontext: Stitching reference image(s)...", color="CYAN")
    stitch_node = _ImageStitch()
    stitched = stitch_node.execute(
        image1=raw_image,
        direction="right",
        match_image_size=True,
        spacing_width=0,
        spacing_color="white",
        image2=reference_image
    )[0]
    log_node(f"FLUX Kontext: Stitched → {stitched.shape}", color="CYAN")

    # === Step 2: FluxKontextImageScale ===
    log_node("FLUX Kontext: Scaling for Kontext...", color="CYAN")
    scale_node = _KontextScale()
    scaled_image = scale_node.execute(image=stitched)[0]
    log_node(f"FLUX Kontext: Scaled → {scaled_image.shape}", color="CYAN")

    # === Step 3: VAE Encode the reference image ===
    log_node("FLUX Kontext: Encoding reference to latent...", color="CYAN")
    import nodes as comfy_nodes
    vae_enc = comfy_nodes.VAEEncode()
    ref_latent = vae_enc.encode(vae, scaled_image)[0]

    # === Step 4: Apply FLUX Guidance ===
    positive_guided = node_helpers.conditioning_set_values(positive_cond, {"guidance": guidance})

    # Kontext uses zeroed conditioning as negative, not a text-encoded negative prompt.
    # BasicGuider handles this internally — no explicit negative needed.

    # === Step 6: ReferenceLatent (inject image context into conditioning) ===
    log_node("FLUX Kontext: Injecting ReferenceLatent...", color="CYAN")
    ref_node = _ReferenceLatent()
    ref_conditioning = ref_node.execute(
        conditioning=positive_guided,
        latent=ref_latent
    )[0]

    # === Step 7: BasicGuider ===
    guider = Guider_Basic(model)
    guider.set_conds(ref_conditioning)

    # === Step 8: Prepare latent (empty or from source) ===
    if use_custom_size:
        # Use Generation Settings dimensions
        log_node(f"FLUX Kontext: Using custom size {width}x{height}", color="CYAN")
        empty_node = _EmptySD3Latent()
        latent = empty_node.generate(width, height, 1)[0]
    else:
        # Use source image dimensions → encode source as starting latent
        log_node("FLUX Kontext: Using source image dimensions", color="CYAN")
        latent = vae_enc.encode(vae, raw_image)[0]

    # === Step 9: Sampler & Scheduler ===
    sampler_obj = KSamplerSelect.execute(sampler_name)[0]
    sigmas = BasicScheduler.execute(model, scheduler, steps, 1.0)[0]

    # === Step 10: Noise & Sampling ===
    noise_obj = Noise_RandomNoise(seed)

    latent_copy = latent.copy()
    latent_image = latent_copy["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
    latent_copy["samples"] = latent_image

    noise_mask = latent_copy.get("noise_mask", None)

    import latent_preview
    x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    log_node(f"FLUX Kontext: Sampling {steps} steps (guidance={guidance})...", color="CYAN")
    samples = guider.sample(
        noise_obj.generate_noise(latent_copy),
        latent_image,
        sampler_obj,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=noise_obj.seed
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    result_latent = latent_copy.copy()
    result_latent["samples"] = samples

    # === Step 11: VAE Decode ===
    log_node("FLUX Kontext: Decoding VAE...", color="CYAN")
    decoded_image = vae_decode_fn(vae, result_latent)

    log_node("FLUX Kontext: ✅ Complete", color="GREEN")
    return decoded_image, result_latent

