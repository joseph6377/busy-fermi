"""
Vendored from ComfyUI-LTXVideo/easy_samplers.py (Apache 2.0)

Only the classes needed by the LoopingSampler and UmeAiRT pipelines:
- LTXVBaseSampler
- LTXVExtendSampler
- LTXVInContextSampler
- LinearOverlapLatentTransition
"""

import copy

import comfy
import comfy_extras
import nodes
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SplitSigmas
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVAddGuide, LTXVCropGuides

from .guide import blur_internal
from .latent_norm import LTXVAdainLatent
from .latents import LTXVAddLatentGuide, LTXVSelectLatents


def _get_raw_conds_from_guider(guider):
    """Extract raw positive/negative conditioning from a guider."""
    if not hasattr(guider, "raw_conds"):
        if "negative" not in guider.original_conds:
            raise ValueError(
                "Guider does not have negative conds, cannot use it as a guider."
            )
        raw_pos = guider.original_conds["positive"]
        positive = [[raw_pos[0]["cross_attn"], copy.deepcopy(raw_pos[0])]]
        raw_neg = guider.original_conds["negative"]
        negative = [[raw_neg[0]["cross_attn"], copy.deepcopy(raw_neg[0])]]
        guider.raw_conds = (positive, negative)
    return guider.raw_conds


# ---------------------------------------------------------------------------
# LTXVBaseSampler
# ---------------------------------------------------------------------------

class LTXVBaseSampler:
    """Base T2V/I2V sampler with optional keyframe conditioning."""

    def sample(
        self,
        model,
        vae,
        width,
        height,
        num_frames,
        guider,
        sampler,
        sigmas,
        noise,
        optional_cond_images=None,
        optional_cond_indices=None,
        strength=0.9,
        crop="disabled",
        crf=35,
        blur=0,
        optional_negative_index_latents=None,
        optional_negative_index=-1,
        optional_negative_index_strength=1.0,
        optional_initialization_latents=None,
        guiding_start_step=0,
        guiding_end_step=1000,
    ):
        guider = copy.copy(guider)
        guider.original_conds = copy.deepcopy(guider.original_conds)
        positive, negative = _get_raw_conds_from_guider(guider)

        if optional_cond_images is not None:
            optional_cond_images = (
                comfy.utils.common_upscale(
                    optional_cond_images.movedim(-1, 1),
                    width, height, "bilinear", crop=crop,
                )
                .movedim(1, -1)
                .clamp(0, 1)
            )
            optional_cond_images = comfy_extras.nodes_lt.LTXVPreprocess.execute(
                optional_cond_images, crf
            )[0]
            for i in range(optional_cond_images.shape[0]):
                optional_cond_images[i] = blur_internal(
                    optional_cond_images[i].unsqueeze(0), blur
                )

        if optional_cond_indices is not None and optional_cond_images is not None:
            optional_cond_indices = optional_cond_indices.split(",")
            optional_cond_indices = [int(i) for i in optional_cond_indices]
            assert len(optional_cond_indices) == len(optional_cond_images), \
                "Number of optional cond images must match number of optional cond indices"

        if optional_initialization_latents is None:
            (latents,) = EmptyLTXVLatentVideo().execute(width, height, num_frames, 1)
        else:
            latents = optional_initialization_latents

        if optional_cond_images is not None and 0 in optional_cond_indices:
            # Apply classical i2v conditioning on the first frame
            idx_0 = optional_cond_indices.index(0)
            encode_pixels = optional_cond_images[idx_0: idx_0 + 1, :, :, :3]
            t = vae.encode(encode_pixels)
            latents["samples"][:, :, :t.shape[2]] = t

            if "noise_mask" not in latents:
                conditioning_latent_frames_mask = torch.ones(
                    (1, 1, latents["samples"].shape[2], 1, 1),
                    dtype=torch.float32, device=latents["samples"].device,
                )
                conditioning_latent_frames_mask[:, :, :t.shape[2]] = 1.0 - strength
                latents["noise_mask"] = conditioning_latent_frames_mask
            else:
                latents["noise_mask"][:, :, :t.shape[2]] = 1.0 - strength
                conditioning_latent_frames_mask = latents["noise_mask"]
        else:
            conditioning_latent_frames_mask = None

        high_sigmas, rest_sigmas = SplitSigmas().get_sigmas(sigmas, guiding_start_step)
        middle_sigmas, low_sigmas = SplitSigmas().get_sigmas(
            rest_sigmas, guiding_end_step - guiding_start_step
        )

        if len(high_sigmas) > 1:
            (_, new_latents) = SamplerCustomAdvanced().sample(
                noise=noise, guider=guider, sampler=sampler,
                sigmas=high_sigmas, latent_image=latents,
            )

        if optional_cond_images is not None:
            for cond_image, cond_idx in zip(optional_cond_images, optional_cond_indices):
                if cond_idx == 0:
                    continue
                (positive, negative, latents,) = LTXVAddGuide.execute(
                    positive=positive, negative=negative, vae=vae,
                    latent=latents, image=cond_image.unsqueeze(0),
                    frame_idx=cond_idx, strength=strength,
                )

        if optional_negative_index_latents is not None:
            (positive, negative, latents,) = LTXVAddLatentGuide().generate(
                vae=vae, positive=positive, negative=negative,
                latent=latents, guiding_latent=optional_negative_index_latents,
                latent_idx=optional_negative_index,
                strength=optional_negative_index_strength,
            )

        guider.set_conds(positive, negative)

        (output_latents, denoised_output_latents) = SamplerCustomAdvanced().sample(
            noise=noise, guider=guider, sampler=sampler,
            sigmas=middle_sigmas, latent_image=latents,
        )

        positive, negative, denoised_output_latents = LTXVCropGuides.execute(
            positive=positive, negative=negative, latent=denoised_output_latents,
        )

        denoised_output_latents["noise_mask"] = conditioning_latent_frames_mask

        if len(low_sigmas) > 1:
            (_, denoised_output_latents) = SamplerCustomAdvanced().sample(
                noise=noise, guider=guider, sampler=sampler,
                sigmas=low_sigmas, latent_image=denoised_output_latents,
            )

        return (denoised_output_latents, positive, negative)


# ---------------------------------------------------------------------------
# LTXVExtendSampler
# ---------------------------------------------------------------------------

class LTXVExtendSampler:
    """Extends a video latent by generating new frames conditioned on the last overlap frames."""

    def sample(
        self,
        model,
        vae,
        latents,
        num_new_frames,
        frame_overlap,
        guider,
        sampler,
        sigmas,
        noise,
        strength=0.5,
        guiding_strength=1.0,
        cond_image_strength=1.0,
        optional_guiding_latents=None,
        optional_cond_images=None,
        optional_cond_indices=None,
        optional_reference_latents=None,
        optional_initialization_latents=None,
        adain_factor=0.0,
        optional_negative_index_latents=None,
        optional_negative_index=-1,
        optional_negative_index_strength=1.0,
        guiding_start_step=0,
        guiding_end_step=1000,
        normalize_per_frame=False,
    ):
        guider = copy.copy(guider)
        guider.original_conds = copy.deepcopy(guider.original_conds)

        if optional_cond_indices is not None and optional_cond_images is not None:
            optional_cond_indices = optional_cond_indices.split(",")
            optional_cond_indices = [int(i) for i in optional_cond_indices]
            assert len(optional_cond_indices) == len(optional_cond_images), \
                "Number of optional cond images must match number of optional cond indices"

        positive, negative = _get_raw_conds_from_guider(guider)

        samples = latents["samples"]
        batch, channels, frames, height, width = samples.shape
        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )
        overlap = frame_overlap // time_scale_factor

        if num_new_frames == -1 and optional_guiding_latents is not None:
            num_new_frames = (
                optional_guiding_latents["samples"].shape[2] - overlap
            ) * time_scale_factor

        (last_overlap_latents,) = LTXVSelectLatents().select_latents(
            latents, -overlap, -1
        )

        if optional_initialization_latents is None:
            new_latents = EmptyLTXVLatentVideo.execute(
                width=width * width_scale_factor,
                height=height * height_scale_factor,
                length=overlap * time_scale_factor + num_new_frames,
                batch_size=1,
            )[0]
        else:
            new_latents = optional_initialization_latents

        last_overlap_latents["samples"] = last_overlap_latents["samples"].to(
            new_latents["samples"].device
        )

        (positive, negative, new_latents,) = LTXVAddLatentGuide().generate(
            vae=vae, positive=positive, negative=negative,
            latent=new_latents, guiding_latent=last_overlap_latents,
            latent_idx=0, strength=strength,
        )

        high_sigmas, rest_sigmas = SplitSigmas().get_sigmas(sigmas, guiding_start_step)
        middle_sigmas, low_sigmas = SplitSigmas().get_sigmas(
            rest_sigmas, guiding_end_step - guiding_start_step
        )

        if optional_cond_images is not None:
            for cond_image, cond_idx in zip(optional_cond_images, optional_cond_indices):
                if optional_guiding_latents is not None and cond_idx % 8 == 1:
                    raise ValueError(
                        f"Conditioning image index {cond_idx} is a multiple of 8 + 1 "
                        "and guiding latents are used. Please provide other indices."
                    )
                (positive, negative, new_latents,) = LTXVAddGuide.execute(
                    positive=positive, negative=negative, vae=vae,
                    latent=new_latents, image=cond_image.unsqueeze(0),
                    frame_idx=cond_idx, strength=cond_image_strength,
                )

        if len(high_sigmas) > 1:
            guider.set_conds(positive, negative)
            (_, new_latents) = SamplerCustomAdvanced().sample(
                noise=noise, guider=guider, sampler=sampler,
                sigmas=high_sigmas, latent_image=new_latents,
            )

        if optional_guiding_latents is not None:
            optional_guiding_latents = LTXVSelectLatents().select_latents(
                optional_guiding_latents, overlap, -1
            )[0]
            (positive, negative, new_latents,) = LTXVAddLatentGuide().generate(
                vae=vae, positive=positive, negative=negative,
                latent=new_latents, guiding_latent=optional_guiding_latents,
                latent_idx=last_overlap_latents["samples"].shape[2],
                strength=guiding_strength,
            )

        if optional_negative_index_latents is not None:
            (positive, negative, new_latents,) = LTXVAddLatentGuide().generate(
                vae=vae, positive=positive, negative=negative,
                latent=new_latents, guiding_latent=optional_negative_index_latents,
                latent_idx=optional_negative_index,
                strength=optional_negative_index_strength,
            )

        guider.set_conds(positive, negative)

        (output_latents, denoised_output_latents) = SamplerCustomAdvanced().sample(
            noise=noise, guider=guider, sampler=sampler,
            sigmas=middle_sigmas, latent_image=new_latents,
        )

        positive, negative, denoised_output_latents = LTXVCropGuides.execute(
            positive=positive, negative=negative, latent=denoised_output_latents,
        )

        if len(low_sigmas) > 1:
            (positive, negative, denoised_output_latents,) = LTXVAddLatentGuide().generate(
                vae=vae, positive=positive, negative=negative,
                latent=denoised_output_latents, guiding_latent=last_overlap_latents,
                latent_idx=0, strength=strength,
            )

            if optional_cond_images is not None:
                for cond_image, cond_idx in zip(optional_cond_images, optional_cond_indices):
                    if optional_guiding_latents is not None and cond_idx % 8 == 1:
                        raise ValueError(
                            f"Conditioning image index {cond_idx} is a multiple of 8 + 1 "
                            "and guiding latents are used."
                        )
                    (positive, negative, denoised_output_latents,) = LTXVAddGuide.execute(
                        positive=positive, negative=negative, vae=vae,
                        latent=denoised_output_latents, image=cond_image.unsqueeze(0),
                        frame_idx=cond_idx, strength=cond_image_strength,
                    )

            guider.set_conds(positive, negative)
            (_, denoised_output_latents) = SamplerCustomAdvanced().sample(
                noise=noise, guider=guider, sampler=sampler,
                sigmas=low_sigmas, latent_image=denoised_output_latents,
            )
            positive, negative, denoised_output_latents = LTXVCropGuides.execute(
                positive=positive, negative=negative, latent=denoised_output_latents,
            )

        if optional_reference_latents is not None:
            denoised_output_latents = LTXVAdainLatent().batch_normalize(
                latents=denoised_output_latents,
                reference=optional_reference_latents,
                factor=adain_factor,
                per_frame=normalize_per_frame,
            )[0]

        # Drop first output latent (reinterpreted 8-frame latent as 1-frame)
        truncated_denoised_output_latents = LTXVSelectLatents().select_latents(
            denoised_output_latents, 1, -1
        )[0]

        # Fuse new frames with old ones
        (latents,) = LinearOverlapLatentTransition().process(
            latents, truncated_denoised_output_latents, overlap - 1, axis=2
        )
        return (latents, positive, negative)


# ---------------------------------------------------------------------------
# LTXVInContextSampler
# ---------------------------------------------------------------------------

class LTXVInContextSampler:
    """IC-LoRA guided sampler for in-context video generation."""

    def sample(
        self,
        vae,
        guider,
        sampler,
        sigmas,
        noise,
        guiding_latents,
        optional_cond_images=None,
        optional_cond_indices=None,
        num_frames=0,
        optional_initialization_latents=None,
        optional_negative_index_latents=None,
        optional_negative_index=-1,
        optional_negative_index_strength=1.0,
        cond_image_strength=1.0,
        guiding_strength=1.0,
        guiding_start_step=0,
        guiding_end_step=1000,
    ):
        guider = copy.copy(guider)
        guider.original_conds = copy.deepcopy(guider.original_conds)
        if optional_cond_images is None:
            optional_cond_indices = None

        if optional_cond_indices is not None and optional_cond_images is not None:
            optional_cond_indices = optional_cond_indices.split(",")
            optional_cond_indices = [int(i) for i in optional_cond_indices]
            assert len(optional_cond_indices) == len(optional_cond_images), \
                "Number of optional cond images must match number of optional cond indices"

        positive, negative = _get_raw_conds_from_guider(guider)

        time_scale_factor, width_scale_factor, height_scale_factor = (
            vae.downscale_index_formula
        )

        batch, channels, frames, height, width = guiding_latents["samples"].shape
        if num_frames != -1:
            frames = (num_frames - 1) // time_scale_factor + 1

        if optional_initialization_latents is not None:
            new_latents = optional_initialization_latents
        else:
            new_latents = EmptyLTXVLatentVideo.execute(
                width=width * width_scale_factor,
                height=height * height_scale_factor,
                length=(frames - 1) * time_scale_factor + 1,
                batch_size=1,
            )[0]

        high_sigmas, rest_sigmas = SplitSigmas().get_sigmas(sigmas, guiding_start_step)
        middle_sigmas, low_sigmas = SplitSigmas().get_sigmas(
            rest_sigmas, guiding_end_step - guiding_start_step
        )

        if len(high_sigmas) > 1:
            (_, new_latents) = SamplerCustomAdvanced().sample(
                noise=noise, guider=guider, sampler=sampler,
                sigmas=high_sigmas, latent_image=new_latents,
            )

        if optional_cond_indices is not None and 0 in optional_cond_indices:
            guiding_latents = LTXVSelectLatents().select_latents(
                guiding_latents, 1, -1
            )[0]
            skip_one_guiding_latent = True
        else:
            skip_one_guiding_latent = False

        (positive, negative, new_latents,) = LTXVAddLatentGuide().generate(
            vae=vae, positive=positive, negative=negative,
            latent=new_latents, guiding_latent=guiding_latents,
            latent_idx=1 if skip_one_guiding_latent else 0,
            strength=guiding_strength,
        )

        if optional_cond_images is not None:
            for cond_image, cond_idx in zip(optional_cond_images, optional_cond_indices):
                if cond_idx % 8 == 1:
                    raise ValueError(
                        f"Conditioning image index {cond_idx} is a multiple of 8 + 1 "
                        "and guiding latents are used."
                    )
                (positive, negative, new_latents,) = LTXVAddGuide.execute(
                    positive=positive, negative=negative, vae=vae,
                    latent=new_latents, image=cond_image.unsqueeze(0),
                    frame_idx=cond_idx, strength=cond_image_strength,
                )

        if optional_negative_index_latents is not None:
            (positive, negative, new_latents,) = LTXVAddLatentGuide().generate(
                vae=vae, positive=positive, negative=negative,
                latent=new_latents, guiding_latent=optional_negative_index_latents,
                latent_idx=optional_negative_index,
                strength=optional_negative_index_strength,
            )

        guider.set_conds(positive, negative)

        (_, denoised_output_latents) = SamplerCustomAdvanced().sample(
            noise=noise, guider=guider, sampler=sampler,
            sigmas=middle_sigmas, latent_image=new_latents,
        )

        positive, negative, denoised_output_latents = LTXVCropGuides.execute(
            positive=positive, negative=negative, latent=denoised_output_latents,
        )

        if len(low_sigmas) > 1:
            guider.set_conds(positive, negative)
            (_, denoised_output_latents) = SamplerCustomAdvanced().sample(
                noise=noise, guider=guider, sampler=sampler,
                sigmas=low_sigmas, latent_image=denoised_output_latents,
            )
            positive, negative, denoised_output_latents = LTXVCropGuides.execute(
                positive=positive, negative=negative, latent=denoised_output_latents,
            )

        return (denoised_output_latents, positive, negative)


# ---------------------------------------------------------------------------
# LinearOverlapLatentTransition
# ---------------------------------------------------------------------------

class LinearOverlapLatentTransition:
    """Linear blending transition between two overlapping latents."""

    def _get_subbatch(self, samples):
        s = samples.copy()
        return s["samples"]

    def process(self, samples1, samples2, overlap, axis=0):
        samples1 = self._get_subbatch(samples1)
        samples2 = self._get_subbatch(samples2)

        # Create transition coefficients
        alpha = torch.linspace(1, 0, overlap + 2)[1:-1].to(samples1.device)

        # Create shape for broadcasting based on the axis
        shape = [1] * samples1.dim()
        shape[axis] = alpha.size(0)
        alpha = alpha.reshape(shape)

        # Create slices for the overlap regions
        slice_all = [slice(None)] * samples1.dim()
        slice_overlap1 = slice_all.copy()
        slice_overlap1[axis] = slice(-overlap, None)
        slice_overlap2 = slice_all.copy()
        slice_overlap2[axis] = slice(0, overlap)
        slice_rest1 = slice_all.copy()
        slice_rest1[axis] = slice(None, -overlap)
        slice_rest2 = slice_all.copy()
        slice_rest2[axis] = slice(overlap, None)

        # Combine samples
        parts = [
            samples1[tuple(slice_rest1)],
            alpha * samples1[tuple(slice_overlap1)]
            + (1 - alpha) * samples2[tuple(slice_overlap2)],
            samples2[tuple(slice_rest2)],
        ]

        combined_samples = torch.cat(parts, dim=axis)
        combined_batch_index = torch.arange(0, combined_samples.shape[0]).to(
            dtype=torch.float32
        )

        return (
            {
                "samples": combined_samples,
                "batch_index": combined_batch_index,
            },
        )
