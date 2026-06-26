"""
UmeAiRT Toolkit - Image Generator (Block Sampler)
---------------------------------------------------
Central hub node: receives models + settings + prompts, orchestrates the
full generation pipeline, and produces the UME_PIPELINE output.

This is the thin orchestrator — all heavy logic is delegated to:
- sampler_tasks.py     (mode-specific image preparation)
- sampler_cache.py     (prompt caching)
- sampler_controlnet.py (ControlNet application)
- flux_sampler.py      (FLUX-specific sampling)
"""

import nodes as comfy_nodes
from .common import GenerationContext, UmeBundle, UmeSettings, UmeImage, log_node, validate_bundle, resize_tensor
from .optimization_utils import SamplerContext, warmup_vae
from .sampler_cache import PromptCache, build_zero_cond
from .sampler_controlnet import apply_controlnets
from .sampler_tasks import (
    ImagePrepResult,
    prepare_txt2img,
    prepare_img2img,
    prepare_inpaint,
    prepare_outpaint,
    composite_inpaint,
)
from typing import Tuple, Optional, List

try:
    from . import flux_sampler as _flux
except ImportError:
    _flux = None

try:
    from .facedetailer_core import detector, logic as fd_logic
except ImportError:
    pass


# --- Processor Blocks ---

class UmeAiRT_BlockSampler:
    """Central hub: receives models + settings + prompts as side-inputs,
    creates the GenerationContext pipeline, samples, and stores the image inside.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "Model bundle from a Loader node."}),
                "positive": ("POSITIVE", {"forceInput": True, "tooltip": "Describe what you want in the image. Connect a Prompt Input or CLIP Text Encode node."}),
                "settings": ("UME_SETTINGS", {"tooltip": "Settings from Generation Settings node."}),
            },
            "optional": {
                "negative": ("NEGATIVE", {"forceInput": True, "tooltip": "Describe what to avoid in the image. Optional for FLUX models. Connect a Prompt Input or CLIP Text Encode node."}),
                "loras": ("UME_LORA_STACK", {"tooltip": "Connect a LoRA Block node to apply style/character modifications to the model."}),
                "images": ("UME_IMAGE", {"tooltip": "Connect an Image Process node for img2img, inpaint, or outpaint workflows."}),
            }
        }
    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Sampler"
    DESCRIPTION = "Central hub node that generates images using the provided models, settings, and prompts."

    def __init__(self):
        self.lora_loader = comfy_nodes.LoraLoader()
        self.cnet_loader = comfy_nodes.ControlNetLoader()
        self.cnet_apply = comfy_nodes.ControlNetApplyAdvanced()
        self._vae_encode = comfy_nodes.VAEEncode()
        self._ksampler = comfy_nodes.KSampler()
        self._vae_decode = comfy_nodes.VAEDecode()
        self._controlnet_cache = {}
        self._prompt_cache = PromptCache()

    def process(self,
                model_bundle: UmeBundle,
                positive: Optional[str] = None,
                negative: Optional[str] = None,
                settings: UmeSettings = None,
                loras: Optional[List[Tuple[str, float, float]]] = None,
                images: Optional[UmeImage] = None) -> Tuple[GenerationContext]:
        # 1. Validate and unpack model_bundle
        validate_bundle(model_bundle, ["model", "clip", "vae"], context="Image Generator")
        model = model_bundle.model
        clip = model_bundle.clip
        vae = model_bundle.vae

        ctx = GenerationContext()
        ctx.model = model
        ctx.clip = clip
        ctx.vae = vae
        ctx.model_name = model_bundle.model_name
        ctx.loader_type = getattr(model_bundle, "loader_type", "")

        # 2. Apply settings
        ctx.width = settings.width
        ctx.height = settings.height
        ctx.steps = settings.steps
        ctx.cfg = settings.cfg
        ctx.sampler_name = settings.sampler_name
        ctx.scheduler = settings.scheduler
        ctx.seed = settings.seed

        # 2b. Apply bundle overrides (e.g. from Lightning Accelerator)
        bundle_overrides = getattr(model_bundle, "overrides", None)
        if bundle_overrides and isinstance(bundle_overrides, dict):
            if "steps" in bundle_overrides:
                ctx.steps = bundle_overrides["steps"]
            if "cfg" in bundle_overrides:
                ctx.cfg = bundle_overrides["cfg"]
            if "sampler_name" in bundle_overrides:
                ctx.sampler_name = bundle_overrides["sampler_name"]
            if "scheduler" in bundle_overrides:
                ctx.scheduler = bundle_overrides["scheduler"]
            log_node(
                f"Image Generator: ⚡ Bundle overrides applied — "
                f"CFG={ctx.cfg}, Steps={ctx.steps}, "
                f"Sampler={ctx.sampler_name}, Scheduler={ctx.scheduler}",
                color="CYAN"
            )

        controlnets = []
        if images:
            controlnets = images.controlnets if images.controlnets else []

        # 3. Apply LoRAs
        if loras:
            if not model or not clip:
                raise ValueError("Image Generator: No base Model/CLIP for LoRAs.")
            loaded_loras_meta = []
            for lora_def in loras:
                name, str_model, str_clip = lora_def
                if name != "None":
                    try:
                         model, clip = self.lora_loader.load_lora(model, clip, name, str_model, str_clip)
                         loaded_loras_meta.append({"name": name, "strength": str_model})
                    except Exception as e:
                        log_node(f"Image Generator LoRA Error ({name}): {e}", color="RED")
            ctx.model = model
            ctx.clip = clip
            ctx.loras = loaded_loras_meta

        if not model or not vae or not clip:
            raise ValueError("Image Generator: Missing Model/VAE/CLIP.")

        width, height = ctx.width, ctx.height
        steps, cfg = ctx.steps, ctx.cfg
        sampler_name, scheduler = ctx.sampler_name, ctx.scheduler
        seed = ctx.seed

        denoise = images.denoise if images else ctx.denoise
        ctx.denoise = denoise

        # 3b. Apply model sampling overrides (architecture-specific)
        actual_shift = model_bundle.shift
        if actual_shift is not None and actual_shift < 0:
            actual_shift = steps / 6.0  # Dynamic shift proportional to steps

        if actual_shift and actual_shift > 0:
            if model_bundle.loader_type == "zimg":
                import comfy.model_sampling
                m = model.clone()
                sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
                sampling_type = comfy.model_sampling.CONST
                class ModelSamplingAdvanced(sampling_base, sampling_type):
                    pass
                model_sampling = ModelSamplingAdvanced(model.model.model_config)
                model_sampling.set_parameters(shift=actual_shift, multiplier=1.0)
                m.add_object_patch("model_sampling", model_sampling)
                ctx.model = m
                log_node(f"Image Generator: Applied Z-IMG Sampling (shift={actual_shift:.2f}, multiplier=1.0).", color="CYAN")
            else:
                try:
                    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
                    model = ModelSamplingSD3().patch(model, shift=actual_shift)[0]
                    ctx.model = model
                    log_node(f"Image Generator: Applied ModelSamplingSD3 (shift={actual_shift:.2f}).", color="CYAN")
                except ImportError:
                    log_node("Image Generator: Warning — ModelSamplingSD3 not available.", color="YELLOW")

        # 4. Handle Prompts
        pos_text = positive if positive is not None else ""
        neg_text = negative if negative is not None else ""
        
        if neg_text.strip():
            if model_bundle.loader_type == "zimg" and "turbo" in (model_bundle.model_name or "").lower():
                log_node("Image Generator: Warning — Z-IMG Turbo ignores negative prompts. Use Z-IMG Normal instead.", color="YELLOW")
            elif cfg <= 1.0:
                log_node("Image Generator: Warning — Negative prompt is ignored when CFG <= 1.0.", color="YELLOW")
        ctx.positive_prompt = pos_text
        ctx.negative_prompt = neg_text

        # 5. Prepare image based on mode
        prep = ImagePrepResult(denoise=denoise)
        latent_image = None

        if images:
            mode_str = images.mode or "img2img"

            if mode_str == "kontext":
                # Kontext: early delegation — completely different pipeline
                pass  # handled after prompt encoding below
            elif mode_str == "outpaint":
                prep = prepare_outpaint(images, height, width, model_bundle)
            elif mode_str == "inpaint":
                prep = prepare_inpaint(images, height, width, model_bundle)
            else:
                prep = prepare_img2img(images, height, width, self._vae_encode)

            if mode_str != "kontext":
                raw_image = prep.raw_image
                source_mask = prep.source_mask
                ctx.source_image = raw_image
                ctx.source_mask = source_mask

                # Encode latent from source image
                if prep.mode_str in ["inpaint", "outpaint"] and source_mask is not None:
                    if prep.flux_fill_info and prep.flux_fill_info.get("is_flux_fill"):
                        # FLUX Fill: skip latent setup — flux_sampler handles everything
                        pass
                    else:
                        latent_image = self._vae_encode.encode(vae, raw_image)[0]
                        latent_image["noise_mask"] = source_mask
                elif prep.denoise < 1.0:
                    latent_image = self._vae_encode.encode(vae, raw_image)[0]

        if latent_image is None and ctx.latent is not None:
            latent_image = ctx.latent

        # 6. Empty latent for txt2img
        if latent_image is None:
            latent_image = prepare_txt2img(width, height, model, denoise)
            denoise = 1.0

        # 7. Encode prompts (with cache)
        # Detect model architecture from bundle metadata (manifest-driven)
        is_flux = _flux is not None and model_bundle.loader_type == "flux"

        cached = self._prompt_cache.try_get_cached(pos_text, neg_text, clip, loras, controlnets)
        if cached:
            positive_cond, negative_cond = cached
        else:
            log_node("Image Generator: Encoding Prompts...")
            tokens = clip.tokenize(pos_text)
            positive_cond = clip.encode_from_tokens_scheduled(tokens)

            if is_flux and not neg_text:
                log_node("Image Generator: FLUX detected — using zero conditioning (no negative prompt needed).", color="CYAN")
                negative_cond = build_zero_cond(positive_cond)
            else:
                tokens = clip.tokenize(neg_text)
                negative_cond = clip.encode_from_tokens_scheduled(tokens)

            self._prompt_cache.update(pos_text, neg_text, clip, loras, controlnets, positive_cond, negative_cond)

        # 8. Resize ControlNet images to generation dimensions
        if controlnets:
            resized_cnets = []
            for cnet_def in controlnets:
                if len(cnet_def) == 6:
                    c_name, c_image, c_str, c_start, c_end, c_type = cnet_def
                else:
                    c_name, c_image, c_str, c_start, c_end = cnet_def
                    c_type = None
                if c_image is not None and c_name != "None":
                    c_image = resize_tensor(c_image, height, width, interp_mode="bilinear")
                if c_type is not None:
                    resized_cnets.append((c_name, c_image, c_str, c_start, c_end, c_type))
                else:
                    resized_cnets.append((c_name, c_image, c_str, c_start, c_end))
            controlnets = resized_cnets

        # 9. Apply ControlNets
        positive_cond, negative_cond = apply_controlnets(
            controlnets, positive_cond, negative_cond,
            self.cnet_loader, self.cnet_apply, self._controlnet_cache, vae=vae, is_flux=is_flux)

        # 10. FLUX Kontext delegation (image editing pipeline)
        if images and images.mode == "kontext" and _flux is not None:
            kontext_image = images.image
            # Auto-resize source image to settings dimensions if requested
            if images.auto_resize and kontext_image is not None:
                kontext_image = resize_tensor(kontext_image, height, width, interp_mode="bilinear")

            ctx.source_image = kontext_image
            log_node("Image Generator: Delegating to FLUX Kontext pipeline.", color="CYAN")
            try:
                with SamplerContext():
                    image_out, result_latent = _flux.sample_flux_kontext(
                        model, vae, clip, positive_cond,
                        kontext_image, images.reference_image,
                        seed, steps, cfg, sampler_name, scheduler,
                        width, height, images.auto_resize,
                        vae_decode_fn=lambda v, lat: self._vae_decode.decode(v, lat)[0])
            except Exception as e:
                raise RuntimeError(f"FLUX Kontext Sampling Failed: {e}")

            ctx.image = image_out
            ctx.latent = result_latent
            return (ctx,)

        # 10b. QWEN Image Edit delegation (multi-image editing pipeline)
        if images and images.mode == "qwen_edit":
            from . import qwen_sampler as _qwen
            # Collect all edit images (primary + up to 2 extra)
            edit_images = [images.image]
            if images.edit_images:
                edit_images.extend([img for img in images.edit_images if img is not None])

            ctx.source_image = images.image
            log_node("Image Generator: Delegating to QWEN Image Edit pipeline.", color="CYAN")
            try:
                with SamplerContext():
                    image_out, result_latent = _qwen.sample_qwen_edit(
                        model, vae, clip, edit_images,
                        pos_text, neg_text,
                        seed, steps, cfg, sampler_name, scheduler,
                        width, height)
            except Exception as e:
                raise RuntimeError(f"QWEN Image Edit Sampling Failed: {e}")

            ctx.image = image_out
            ctx.latent = result_latent
            return (ctx,)

        # 11. FLUX Fill delegation (outpaint OR inpaint)
        if prep.flux_fill_info and prep.flux_fill_info.get("is_flux_fill") and _flux is not None:
            ffi = prep.flux_fill_info

            if prep.is_outpaint:
                # FLUX Fill Outpaint: uses ImagePadForOutpaint + InpaintModelConditioning
                pad_l, pad_t, pad_r, pad_b = ffi["pad_info"]
                flux_feather = ffi["feathering"]
                log_node("Image Generator: Delegating to FLUX Fill Outpaint pipeline.", color="CYAN")
                try:
                    with SamplerContext():
                        image_out, result_latent = _flux.sample_flux_outpaint(
                            model, vae, clip, positive_cond, negative_cond,
                            ffi["raw_image"], pad_l, pad_t, pad_r, pad_b,
                            seed, steps, cfg, sampler_name, scheduler, denoise,
                            flux_feather,
                            vae_decode_fn=lambda v, lat: self._vae_decode.decode(v, lat)[0])
                except Exception as e:
                    raise RuntimeError(f"FLUX Fill Outpaint Sampling Failed: {e}")
            else:
                # FLUX Fill Inpaint: uses InpaintModelConditioning directly (no padding)
                log_node("Image Generator: Delegating to FLUX Fill Inpaint pipeline.", color="CYAN")
                try:
                    with SamplerContext():
                        image_out, result_latent = _flux.sample_flux_inpaint(
                            model, vae, clip, positive_cond, negative_cond,
                            prep.raw_image, prep.source_mask,
                            seed, steps, cfg, sampler_name, scheduler, denoise,
                            vae_decode_fn=lambda v, lat: self._vae_decode.decode(v, lat)[0])
                except Exception as e:
                    raise RuntimeError(f"FLUX Fill Inpaint Sampling Failed: {e}")

            ctx.image = image_out
            ctx.latent = result_latent
            return (ctx,)

        # 12. FLUX + ControlNet: use BasicGuider pipeline (ControlNetFlux needs it)
        if is_flux and controlnets and _flux is not None:
            log_node("Image Generator: FLUX + ControlNet → Delegating to BasicGuider pipeline.", color="CYAN")
            warmup_vae(vae, latent_image)
            try:
                with SamplerContext():
                    image_out, result_latent = _flux.sample_flux_base(
                        model, vae, positive_cond,
                        seed, steps, cfg, sampler_name, scheduler,
                        latent_image, denoise,
                        vae_decode_fn=lambda v, lat: self._vae_decode.decode(v, lat)[0])
            except Exception as e:
                raise RuntimeError(f"FLUX ControlNet Sampling Failed: {e}")

            if prep.mode_str == "inpaint" and not prep.is_outpaint and prep.raw_image is not None and prep.source_mask is not None:
                image_out = composite_inpaint(image_out, prep.raw_image, prep.source_mask)

            ctx.image = image_out
            ctx.latent = result_latent
            return (ctx,)

        # 13. Apply FLUX guidance (or standard CFG)
        sampler_cfg = cfg
        if is_flux:
            positive_cond, negative_cond, sampler_cfg = _flux.apply_flux_guidance(
                positive_cond, negative_cond, cfg)

        # 14. Standard sampling
        log_node(f"Image Generator: {prep.mode_str} | {width}x{height} | Steps: {steps} | CFG: {sampler_cfg}")
        warmup_vae(vae, latent_image)

        try:
             with SamplerContext():
                 result_latent = self._ksampler.sample(model, seed, steps, sampler_cfg, sampler_name, scheduler, positive_cond, negative_cond, latent_image, denoise)[0]
        except Exception as e:
             raise RuntimeError(f"Sampling Failed: {e}")

        # 15. VAE Decode + Inpaint composite
        log_node("Image Generator: Decoding VAE")
        image_out = self._vae_decode.decode(vae, result_latent)[0]

        if prep.mode_str == "inpaint" and not prep.is_outpaint and prep.raw_image is not None and prep.source_mask is not None:
            image_out = composite_inpaint(image_out, prep.raw_image, prep.source_mask)

        ctx.image = image_out
        ctx.latent = result_latent
        return (ctx,)
