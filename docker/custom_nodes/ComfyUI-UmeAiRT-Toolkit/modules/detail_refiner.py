"""
UmeAiRT Toolkit - Detail Refiner
----------------------------------
Pipeline-aware second-pass sampler that re-samples the generated image
with reduced denoise to add fine detail. Supports progressive multi-pass
refinement with automatic denoise decay (each pass halves the denoise).

Architecture-aware: auto-detects FLUX models and applies embedded guidance
via apply_flux_guidance. Z-IMG and standard SD/SDXL models use the standard
KSampler path (the model sampling patch is already applied by BlockSampler).
"""

import nodes as comfy_nodes
import torch
from .common import log_node, extract_pipeline_params
from .optimization_utils import SamplerContext, warmup_vae

try:
    from . import flux_sampler as _flux
except ImportError:
    _flux = None

try:
    from .sampler_cache import build_zero_cond
except ImportError:
    build_zero_cond = None


class UmeAiRT_DetailRefiner:
    """Detail Refiner — re-samples the pipeline image with reduced denoise
    to add fine detail and resolve soft areas.

    Supports progressive multi-pass refinement where each successive pass
    uses half the denoise of the previous one, producing diminishing but
    accumulative detail enhancement.

    Architecture-aware: auto-detects Z-IMG (Lumina2), FLUX, and standard
    SD/SDXL models and applies the correct sampling strategy.
    """

    @classmethod
    def INPUT_TYPES(s):
        import comfy.samplers
        samplers = ["Pipeline"] + comfy.samplers.KSampler.SAMPLERS
        schedulers = ["Pipeline"] + comfy.samplers.KSampler.SCHEDULERS
        return {
            "required": {
                "gen_pipe": ("UME_PIPELINE", {
                    "tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "enabled": ("BOOLEAN", {
                    "default": True, "label_on": "Active", "label_off": "Passthrough",
                    "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                "denoise": ("FLOAT", {
                    "default": 0.40, "min": 0.05, "max": 1.0, "step": 0.01, "display": "slider",
                    "tooltip": "How much the AI redraws on the first pass. Recommended 0.3–0.5. Lower = preserves more, higher = more creative freedom."}),
                "passes": ("INT", {
                    "default": 1, "min": 1, "max": 5, "step": 1, "display": "slider",
                    "tooltip": "Number of refinement passes. Each pass halves the denoise (e.g. pass 1=0.50, pass 2=0.25, pass 3=0.12). More passes = subtler cumulative detail."}),
            },
            "optional": {
                "steps": ("INT", {
                    "default": 0, "min": 0, "max": 150, "advanced": True,
                    "tooltip": "Override sampling steps. Leave at 0 to use the pipeline's step count."}),
                "cfg": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5, "advanced": True,
                    "tooltip": "Override CFG guidance. Leave at 0 to use the pipeline's CFG value."}),
                "sampler_name": (samplers, {
                    "default": "Pipeline", "advanced": True,
                    "tooltip": "Override the sampler algorithm, or use the pipeline's."}),
                "scheduler": (schedulers, {
                    "default": "Pipeline", "advanced": True,
                    "tooltip": "Override the noise scheduler, or use the pipeline's."}),
                "seed_offset": ("INT", {
                    "default": 1, "min": 0, "max": 1000, "advanced": True,
                    "tooltip": "Offset added to the seed between passes. Default 1 = new seed for independent noise (prevents color deep-frying). 0 = use original seed (not recommended)."}),
            }
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Post-Process"
    DESCRIPTION = "Refines details on detected bounding boxes using a focused secondary pass."

    def __init__(self):
        self._ksampler = comfy_nodes.KSampler()
        self._vae_encode = comfy_nodes.VAEEncode()
        self._vae_decode = comfy_nodes.VAEDecode()

    def process(self, gen_pipe, enabled, denoise, passes,
                steps=0, cfg=0.0, sampler_name="Pipeline", scheduler="Pipeline", seed_offset=1):
        if not enabled:
            return (gen_pipe,)

        image = gen_pipe.image
        if image is None:
            raise ValueError("Detail Refiner: No image in pipeline.")

        pp = extract_pipeline_params(gen_pipe)
        model, vae, clip = pp.model, pp.vae, pp.clip

        # Resolve overrides (0 = use pipeline value)
        is_turbo = "turbo" in getattr(gen_pipe, "model_name", "").lower()
        
        if steps == 0:
            if is_turbo:
                final_steps = max(pp.steps, 20)
                log_node(f"Detail Refiner: Turbo model detected. Auto-inflating schedule resolution to {final_steps} to maintain quality.", color="YELLOW")
            else:
                final_steps = pp.steps
        else:
            final_steps = steps
        final_cfg = pp.cfg if cfg == 0.0 else cfg
        final_sampler = pp.sampler_name if sampler_name == "Pipeline" else sampler_name
        final_scheduler = pp.scheduler if scheduler == "Pipeline" else scheduler

        # Detect architecture from pipeline context
        loader_type = getattr(gen_pipe, "loader_type", "")
        is_flux = _flux is not None and loader_type == "flux"

        # Encode prompts (using scheduled encoding for FLUX/Lumina2 compat)
        log_node("Detail Refiner: Encoding prompts...", color="CYAN")
        tokens = clip.tokenize(pp.pos_text)
        positive_cond = clip.encode_from_tokens_scheduled(tokens)

        if is_flux and not pp.neg_text.strip():
            if build_zero_cond is not None:
                negative_cond = build_zero_cond(positive_cond)
            else:
                neg_tokens = clip.tokenize("")
                negative_cond = clip.encode_from_tokens_scheduled(neg_tokens)
        else:
            neg_tokens = clip.tokenize(pp.neg_text)
            negative_cond = clip.encode_from_tokens_scheduled(neg_tokens)

        # Apply FLUX guidance embedding (sets sampler CFG to 1.0)
        sampler_cfg = final_cfg
        if is_flux:
            positive_cond, negative_cond, sampler_cfg = _flux.apply_flux_guidance(
                positive_cond, negative_cond, final_cfg)

        # Progressive refinement loop
        current_image = image
        current_denoise = denoise
        result_latent = None

        for pass_idx in range(passes):
            # Skip passes where denoise is negligible
            if current_denoise < 0.02:
                log_node(f"Detail Refiner: Skipping pass {pass_idx + 1} (denoise {current_denoise:.3f} < 0.02)", color="YELLOW")
                break

            current_seed = pp.seed + seed_offset * (pass_idx + 1)
            pass_label = f"Pass {pass_idx + 1}/{passes}" if passes > 1 else "Single Pass"
            current_steps = max(1, int(final_steps * current_denoise))

            log_node(
                f"Detail Refiner: {pass_label} | "
                f"Denoise: {current_denoise:.3f} | Steps: {current_steps} | "
                f"CFG: {sampler_cfg} | Seed: {current_seed}",
                color="CYAN"
            )

            # VAE Encode (strip alpha channel if present) or bypass if latent is available
            if pass_idx == 0 and getattr(gen_pipe, "latent", None) is not None:
                log_node("Detail Refiner: Using original pipeline latent to bypass VAE Encode degradation.", color="CYAN")
                latent_image = gen_pipe.latent
            else:
                latent_image = self._vae_encode.encode(vae, current_image[:, :, :, :3])[0]

            # Warmup VAE for Triton JIT (no-op if already warmed up)
            warmup_vae(vae, latent_image)

            # Sample
            try:
                with SamplerContext():
                    import comfy.sample
                    noise = torch.randn(
                        latent_image["samples"].size(),
                        dtype=latent_image["samples"].dtype,
                        layout=latent_image["samples"].layout,
                        generator=torch.manual_seed(current_seed),
                        device="cpu"
                    )
                    
                    # Clone model to disable live previews but keep progress bar
                    m = model.clone()
                    if "transformer_options" in m.model_options:
                        m.model_options["transformer_options"] = m.model_options["transformer_options"].copy()
                        if "callback" in m.model_options["transformer_options"]:
                            del m.model_options["transformer_options"]["callback"]

                    import comfy.utils
                    pbar = comfy.utils.ProgressBar(current_steps)
                    def progress_callback(step, x0, x, total_steps):
                        pbar.update_absolute(step + 1, total_steps, None)

                    result_samples = comfy.sample.sample(
                        m, noise, current_steps, sampler_cfg,
                        final_sampler, final_scheduler,
                        positive_cond, negative_cond,
                        latent_image["samples"], denoise=current_denoise,
                        disable_pbar=False, seed=current_seed,
                        callback=progress_callback
                    )
                    result_latent = latent_image.copy()
                    result_latent["samples"] = result_samples
            except Exception as e:
                raise RuntimeError(f"Detail Refiner: Sampling failed on {pass_label}: {e}")

            # VAE Decode
            current_image = self._vae_decode.decode(vae, result_latent)[0]

            # Progressive denoise decay: halve for next pass
            current_denoise = current_denoise / 2.0

        log_node(f"Detail Refiner: ✅ Complete ({passes} pass{'es' if passes > 1 else ''})", color="GREEN")

        ctx = gen_pipe.clone()
        ctx.image = current_image
        ctx.latent = result_latent
        return (ctx,)
