"""
Vendored from ComfyUI-LTXVideo/latents.py (Apache 2.0)

Only the classes needed by the UmeAiRT Video Extender & Enhancer:
- LTXVSelectLatents
- LTXVAddLatents
- LTXVDilateLatent
- LTXVAddLatentGuide
- LTXVSetAudioVideoMaskByTime
- get_video_latent_blend_coefficients
"""

from typing import Optional

import comfy.utils
import comfy_extras.nodes_lt as nodes_lt
import numpy as np
import torch
from comfy.ldm.lightricks.vae.audio_vae import LATENT_DOWNSAMPLE_FACTOR
from comfy.nested_tensor import NestedTensor

from .iclora_attention import append_guide_attention_entry


# ---------------------------------------------------------------------------
# LTXVSelectLatents
# ---------------------------------------------------------------------------

class LTXVSelectLatents:
    """Selects a range of frames from a video latent.

    Supports positive and negative indexing. Preserves noise masks.
    """

    def select_latents(self, samples: dict, start_index: int, end_index: int) -> tuple:
        s = samples.copy()
        video_latent = s["samples"]
        batch, channels, frames, height, width = video_latent.shape

        # Handle negative indices
        start_idx = frames + start_index if start_index < 0 else start_index
        end_idx = frames + end_index if end_index < 0 else end_index

        # Validate and clamp indices
        start_idx = max(0, min(start_idx, frames - 1))
        end_idx = max(0, min(end_idx, frames - 1))
        if start_idx > end_idx:
            start_idx = min(start_idx, end_idx)

        # Select frames while maintaining 5D format
        s["samples"] = video_latent[:, :, start_idx: end_idx + 1, :, :]

        # Handle noise mask if present
        if "noise_mask" in s and s["noise_mask"] is not None:
            s["noise_mask"] = s["noise_mask"][:, :, start_idx: end_idx + 1, :, :]

        return (s,)


# ---------------------------------------------------------------------------
# LTXVAddLatents
# ---------------------------------------------------------------------------

class LTXVAddLatents:
    """Concatenates two video latents along the frames dimension."""

    def add_latents(self, latents1, latents2):
        s = latents1.copy()
        video_latent1 = latents1["samples"]
        video_latent2 = latents2["samples"]

        target_device = video_latent1.device
        video_latent2 = video_latent2.to(target_device)

        self._validate_dimensions(video_latent1, video_latent2)
        s["samples"] = torch.cat([video_latent1, video_latent2], dim=2)
        s["noise_mask"] = self._merge_noise_masks(
            latents1, latents2, video_latent1.shape[2], video_latent2.shape[2]
        )
        return (s,)

    def _validate_dimensions(self, latent1, latent2):
        if latent1.ndim == 5 and latent2.ndim == 5:
            b1, c1, f1, h1, w1 = latent1.shape
            b2, c2, f2, h2, w2 = latent2.shape
            if not (b1 == b2 and c1 == c2 and h1 == h2 and w1 == w2):
                raise ValueError(
                    f"Latent dimensions must match (except frames).\n"
                    f"Got shapes {latent1.shape} and {latent2.shape}"
                )
        elif latent1.ndim == 4 and latent2.ndim == 4:
            b1, c1, f1, s1 = latent1.shape
            b2, c2, f2, s2 = latent2.shape
            if not (b1 == b2 and c1 == c2 and s1 == s2):
                raise ValueError(
                    f"Latent dimensions must match (except frames).\n"
                    f"Got shapes {latent1.shape} and {latent2.shape}"
                )
        else:
            raise ValueError(
                f"Latent dimensions must be 4 (audio) or 5 (video).\n"
                f"Got shapes {latent1.shape} and {latent2.shape}"
            )

    def _merge_noise_masks(self, latents1, latents2, frames1, frames2) -> Optional[torch.Tensor]:
        if "noise_mask" in latents1 and "noise_mask" in latents2:
            return torch.cat([latents1["noise_mask"], latents2["noise_mask"]], dim=2)
        elif "noise_mask" in latents1 and latents1["noise_mask"] is not None:
            zeros = torch.zeros_like(latents1["noise_mask"][:, :, :frames2, :, :])
            return torch.cat([latents1["noise_mask"], zeros], dim=2)
        elif "noise_mask" in latents2 and latents2["noise_mask"] is not None:
            zeros = torch.zeros_like(latents2["noise_mask"][:, :, :frames1, :, :])
            return torch.cat([zeros, latents2["noise_mask"]], dim=2)
        return None


# ---------------------------------------------------------------------------
# LTXVDilateLatent
# ---------------------------------------------------------------------------

class LTXVDilateLatent:
    """Dilates a latent by a grid size for IC-LoRA conditioning."""

    def dilate_latent(self, latent: dict, horizontal_scale: int, vertical_scale: int) -> tuple:
        if horizontal_scale == 1 and vertical_scale == 1:
            return (latent,)

        samples = latent["samples"]
        mask = latent.get("noise_mask", None)
        dilated_shape = samples.shape[:3] + (
            samples.shape[3] * vertical_scale,
            samples.shape[4] * horizontal_scale,
        )

        dilated_samples = torch.zeros(
            dilated_shape, device=samples.device, dtype=samples.dtype,
            requires_grad=False,
        )
        dilated_samples[..., ::vertical_scale, ::horizontal_scale] = samples

        dilated_mask_shape = (
            dilated_samples.shape[0], 1,
            dilated_samples.shape[2], dilated_samples.shape[3], dilated_samples.shape[4],
        )
        dilated_mask = torch.full(
            dilated_mask_shape, -1.0,
            device=samples.device, dtype=samples.dtype, requires_grad=False,
        )
        dilated_mask[..., ::vertical_scale, ::horizontal_scale] = (
            mask if mask is not None else 1.0
        )
        latent = {"samples": dilated_samples, "noise_mask": dilated_mask}
        return (latent,)


# ---------------------------------------------------------------------------
# LTXVAddLatentGuide
# ---------------------------------------------------------------------------

class LTXVAddLatentGuide:
    """Adds a keyframe or video segment at a specific frame index via latent guide."""

    def generate(self, vae, positive, negative, latent, guiding_latent, latent_idx, strength):
        noise_mask = nodes_lt.get_noise_mask(latent)
        latent = latent["samples"]
        guide = guiding_latent["samples"]

        # Record original (pre-dilation) guide latent shape
        guide_orig_shape = list(guide.shape[2:])  # [F, H_small, W_small]

        assert (
            latent.shape[4] % guide.shape[4] == 0
            and latent.shape[3] % guide.shape[3] == 0
        ), "The ratio of height/width of latents and guiding_latents must be an integer"

        guiding_latent = LTXVDilateLatent().dilate_latent(
            guiding_latent,
            horizontal_scale=latent.shape[4] // guide.shape[4],
            vertical_scale=latent.shape[3] // guide.shape[3],
        )[0]

        guide = guiding_latent["samples"]
        guide_mask = guiding_latent.get("noise_mask", None)

        iclora_tokens_added = guide.shape[2] * guide.shape[3] * guide.shape[4]
        scale_factors = vae.downscale_index_formula

        if latent_idx <= 0:
            frame_idx = latent_idx * scale_factors[0]
        else:
            frame_idx = 1 + (latent_idx - 1) * scale_factors[0]

        positive, negative, latent, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
            positive=positive,
            negative=negative,
            frame_idx=frame_idx,
            latent_image=latent,
            noise_mask=noise_mask,
            guiding_latent=guide,
            strength=strength,
            scale_factors=scale_factors,
            guide_mask=guide_mask,
        )

        positive = append_guide_attention_entry(
            positive, iclora_tokens_added, guide_orig_shape
        )
        negative = append_guide_attention_entry(
            negative, iclora_tokens_added, guide_orig_shape
        )

        return (
            positive,
            negative,
            {"samples": latent, "noise_mask": noise_mask},
        )


# ---------------------------------------------------------------------------
# get_video_latent_blend_coefficients
# ---------------------------------------------------------------------------

def get_video_latent_blend_coefficients(
    video_frame_index_start,
    video_frame_index_end,
    video_frame_count,
    slope_len=3,
):
    """Returns blend coefficient lists for video latent and pixel frames.

    Generates a trapezoidal blend mask:
    - 0.0 outside the range
    - Ramp up over slope_len frames at start
    - 1.0 plateau during [start, end]
    - Ramp down over slope_len frames at end
    """
    coeffs = [0.0] * video_frame_count

    video_frame_index_start = max(0, min(video_frame_count - 1, video_frame_index_start))
    video_frame_index_end = max(video_frame_index_start, min(video_frame_count - 1, video_frame_index_end))
    slope_len = max(1, slope_len)

    # Ramp up
    ramp_start = max(0, video_frame_index_start - slope_len)
    for i in range(ramp_start, video_frame_index_start):
        coeffs[i] = (i - ramp_start + 1) / slope_len

    # Plateau
    for i in range(video_frame_index_start, video_frame_index_end + 1):
        coeffs[i] = 1.0

    # Ramp down
    ramp_end = min(video_frame_count, video_frame_index_end + slope_len + 1)
    for i in range(video_frame_index_end + 1, ramp_end):
        coeffs[i] = 1.0 - ((i - (video_frame_index_end + 1) + 1) / slope_len)
        coeffs[i] = max(0.0, coeffs[i])

    num_coeffs = len(coeffs)
    pixel_frame_length = (num_coeffs - 1) * 8 + 1

    xp = np.array([0] + list(range(1, pixel_frame_length, 8)))
    fp = np.array(coeffs)
    pixel_frame_positions = np.arange(pixel_frame_length)
    pixel_frame_coefficients = np.interp(pixel_frame_positions, xp, fp).tolist()

    return coeffs, pixel_frame_coefficients


# ---------------------------------------------------------------------------
# LTXVSetAudioVideoMaskByTime
# ---------------------------------------------------------------------------

class LTXVSetAudioVideoMaskByTime:
    """Sets audio and video noise masks by time range on AV latents."""

    def run(
        self,
        av_latent,
        positive,
        negative,
        model,
        vae,
        audio_vae,
        start_time,
        end_time,
        video_fps,
        mask_video,
        mask_audio,
        mask_init_value_video,
        mask_init_value_audio,
        slope_len,
        spatial_mask=None,
    ):
        from comfy.ldm.lightricks.av_model import LTXAVModel

        if model.model.diffusion_model.__class__.__name__ != "LTXAVModel":
            raise ValueError("LTXVSetAudioVideoMaskByTime requires an LTXAVModel")

        ltxav: LTXAVModel = model.model.diffusion_model

        # Extract configuration from the audio VAE
        sampling_rate = audio_vae.autoencoder.sampling_rate
        mel_hop_length = audio_vae.autoencoder.mel_hop_length
        audio_latents_per_second = (
            sampling_rate / mel_hop_length / LATENT_DOWNSAMPLE_FACTOR
        )

        time_scale_factor = vae.downscale_index_formula[0]
        video_latents_per_second = video_fps / time_scale_factor

        if not isinstance(av_latent["samples"], NestedTensor):
            raise ValueError("av_latent must be a NestedTensor")

        video_samples, audio_samples = ltxav.separate_audio_and_video_latents(
            av_latent["samples"].tensors, None,
        )
        video_mask = torch.full(
            video_samples.shape, fill_value=mask_init_value_video,
        )
        audio_mask = torch.full(
            audio_samples.shape, fill_value=mask_init_value_audio,
        )
        if spatial_mask is not None:
            if spatial_mask.ndim == 3:
                spatial_mask = spatial_mask.unsqueeze(0)
            if spatial_mask.ndim == 2:
                spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(0)
            spatial_mask = torch.nn.functional.interpolate(
                spatial_mask,
                size=(video_samples.shape[3], video_samples.shape[4]),
                mode="bilinear",
                align_corners=False,
            )

        video_latent_frame_count = video_samples.shape[2]
        audio_latent_frame_count = audio_samples.shape[2]
        video_pixel_frame_count = (video_latent_frame_count - 1) * time_scale_factor + 1

        xp = np.array(
            [0] + list(range(1, video_pixel_frame_count + time_scale_factor, time_scale_factor))
        )
        video_pixel_frame_start_raw = int(round(start_time * video_fps))
        video_latent_frame_index_start = np.searchsorted(xp, video_pixel_frame_start_raw, side="left")
        video_pixel_frame_end_raw = int(round(end_time * video_fps))
        video_latent_frame_index_end = np.searchsorted(xp, video_pixel_frame_end_raw, side="right") - 1
        audio_latent_frame_index_start = int(round(start_time * audio_latents_per_second))
        audio_latent_frame_index_end = int(round(end_time * audio_latents_per_second)) + 1

        # Clamping
        video_latent_frame_index_start = max(0, video_latent_frame_index_start)
        video_latent_frame_index_end = min(video_latent_frame_index_end, video_latent_frame_count)
        audio_latent_frame_index_start = max(0, audio_latent_frame_index_start)
        audio_latent_frame_index_end = min(audio_latent_frame_index_end, audio_latent_frame_count)

        if mask_video:
            if spatial_mask is not None:
                video_mask[
                    :, :, video_latent_frame_index_start:video_latent_frame_index_end, :, :,
                ] = spatial_mask
            else:
                video_mask[
                    :, :, video_latent_frame_index_start:video_latent_frame_index_end
                ] = 1.0
        if mask_audio:
            audio_mask[
                :, :, audio_latent_frame_index_start:audio_latent_frame_index_end
            ] = 1.0

        if "noise_mask" in av_latent:
            base_mask = av_latent["noise_mask"].tensors[0].clone()
            if (
                base_mask.shape[0] == base_mask.shape[1] == 1
                == base_mask.shape[3] == base_mask.shape[4]
            ):
                for frame in range(base_mask.shape[2]):
                    video_mask[:, :, frame, :, :] *= base_mask[0, 0, frame, 0, 0]

        av_latent["noise_mask"] = NestedTensor(
            ltxav.recombine_audio_and_video_latents(video_mask, audio_mask)
        )

        video_latent_blend_coefficients, video_pixel_blend_coefficients = (
            get_video_latent_blend_coefficients(
                video_latent_frame_index_start,
                video_latent_frame_index_end,
                video_latent_frame_count,
                slope_len=slope_len,
            )
        )
        return (
            positive,
            negative,
            av_latent,
            video_latent_blend_coefficients,
            video_pixel_blend_coefficients,
        )
