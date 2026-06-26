"""
UmeAiRT Toolkit - QWEN Image Edit Sampler
-------------------------------------------
Dedicated sampling pipeline for QWEN Image Edit models.
Encapsulates the specialized TextEncodeQwenImageEditPlus encoding
and CFGGuider + SamplerCustomAdvanced sampling required by the
QWEN Image Edit architecture.

Called by the BlockSampler when mode="qwen_edit" is detected.
"""

import torch
import nodes as comfy_nodes
from .common import log_node


def sample_qwen_edit(model, vae, clip, images, pos_text, neg_text,
                     seed, steps, cfg, sampler_name, scheduler,
                     width, height, loras=None):
    """Execute the QWEN Image Edit pipeline.

    This replicates the legacy QWEN_Edit workflow internally:
    1. Scale images to ~1 megapixel (ImageScaleToTotalPixels)
    2. Apply ModelSamplingAuraFlow (shift=3) to the model
    3. Apply CFGNorm (strength=1) for gradient normalization
    4. Encode prompts with TextEncodeQwenImageEditPlus (embeds images into conditioning)
    5. VAEEncode the primary image → latent
    6. Sample with CFGGuider + SamplerCustomAdvanced
    7. VAEDecode → output image

    Args:
        model: The diffusion model from the bundle.
        vae: The QWEN VAE.
        clip: The QWEN CLIP (qwen2.5-vl).
        images: List of IMAGE tensors [image1, image2?, image3?].
        pos_text: Positive prompt (edit instruction).
        neg_text: Negative prompt.
        seed: Random seed.
        steps: Number of sampling steps.
        cfg: CFG scale.
        sampler_name: Sampler algorithm name.
        scheduler: Scheduler name.
        width: Target width.
        height: Target height.
        loras: Optional list of LoRA tuples already applied to model.

    Returns:
        tuple: (image_out, result_latent) — decoded image and raw latent.
    """
    # --- 1. Scale images to ~1 megapixel ---
    from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
    scaler = ImageScaleToTotalPixels()
    scaled_images = []
    for img in images:
        if img is not None:
            if hasattr(scaler, "upscale"):
                scaled = scaler.upscale(img, "lanczos", 1.0)[0]
            else:
                scaled = ImageScaleToTotalPixels.execute(img, "lanczos", 1.0, 1)[0]
            scaled_images.append(scaled)
        else:
            scaled_images.append(None)

    image1 = scaled_images[0] if len(scaled_images) > 0 else None
    image2 = scaled_images[1] if len(scaled_images) > 1 else None
    image3 = scaled_images[2] if len(scaled_images) > 2 else None

    if image1 is None:
        raise ValueError("QWEN Edit: At least one source image is required.")

    # --- 2. Apply ModelSamplingAuraFlow (shift=3) ---
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
    model = ModelSamplingAuraFlow().patch_aura(model, shift=3.0)[0]
    log_node("QWEN Edit: Applied ModelSamplingAuraFlow (shift=3.0).", color="CYAN")

    # --- 3. Apply CFGNorm (strength=1) ---
    try:
        from comfy_extras.nodes_cfg import CFGNorm
        if hasattr(CFGNorm, "execute"):
            model = CFGNorm.execute(model, strength=1.0)[0]
        else:
            model = CFGNorm().patch(model, strength=1.0)[0]
        log_node("QWEN Edit: Applied CFGNorm (strength=1.0).", color="CYAN")
    except ImportError:
        log_node("QWEN Edit: CFGNorm not available, skipping.", color="YELLOW")

    # --- 4. Encode prompts with TextEncodeQwenImageEditPlus ---
    try:
        from comfy_extras.nodes_qwen_vl import TextEncodeQwenImageEditPlus
    except ImportError:
        from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus

    log_node("QWEN Edit: Encoding prompts with TextEncodeQwenImageEditPlus...", color="CYAN")
    
    if hasattr(TextEncodeQwenImageEditPlus, "execute"):
        positive_cond = TextEncodeQwenImageEditPlus.execute(
            clip=clip, prompt=pos_text,
            vae=vae,
            image1=image1, image2=image2, image3=image3
        )[0]
        negative_cond = TextEncodeQwenImageEditPlus.execute(
            clip=clip, prompt=neg_text,
            vae=vae,
            image1=image1, image2=image2, image3=image3
        )[0]
    else:
        encoder = TextEncodeQwenImageEditPlus()
        positive_cond = encoder.execute(
            clip=clip, prompt=pos_text,
            vae=vae,
            image1=image1, image2=image2, image3=image3
        )[0]
        negative_cond = encoder.execute(
            clip=clip, prompt=neg_text,
            vae=vae,
            image1=image1, image2=image2, image3=image3
        )[0]

    # --- 5. VAEEncode primary image → latent ---
    vae_encode = comfy_nodes.VAEEncode()
    latent_image = vae_encode.encode(vae, image1)[0]

    # --- 6. Sampling: BasicScheduler + KSamplerSelect + RandomNoise + CFGGuider + SamplerCustomAdvanced ---
    from comfy_extras.nodes_custom_sampler import (
        BasicScheduler, KSamplerSelect, RandomNoise,
        CFGGuider, SamplerCustomAdvanced
    )

    sigmas = BasicScheduler().get_sigmas(model, scheduler, steps, 1.0)[0]
    sampler_obj = KSamplerSelect().get_sampler(sampler_name)[0]
    noise = RandomNoise().get_noise(seed)[0]
    guider = CFGGuider().get_guider(model, positive_cond, negative_cond, cfg)[0]

    num_images = sum(1 for img in [image1, image2, image3] if img is not None)
    log_node(f"QWEN Edit: {num_images} image(s) | {width}x{height} | Steps: {steps} | CFG: {cfg}", color="CYAN")

    if hasattr(SamplerCustomAdvanced, "execute"):
        result = SamplerCustomAdvanced.execute(noise=noise, guider=guider, sampler=sampler_obj, sigmas=sigmas, latent_image=latent_image)
    else:
        result = SamplerCustomAdvanced().sample(noise, guider, sampler_obj, sigmas, latent_image)
    result_latent = result[0]

    # --- 7. VAEDecode ---
    vae_decode = comfy_nodes.VAEDecode()
    image_out = vae_decode.decode(vae, result_latent)[0]

    log_node("QWEN Edit: ✅ Generation complete.", color="GREEN")
    return image_out, result_latent
