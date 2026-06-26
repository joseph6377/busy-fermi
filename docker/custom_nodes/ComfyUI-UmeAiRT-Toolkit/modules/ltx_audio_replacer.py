"""
UmeAiRT Toolkit - LTX Audio Replacer
---------------------------------------
Replaces the audio track in an existing video pipeline.

Two modes:
- Replace from File: swap in an external AUDIO input (from ComfyUI's LoadAudio)
- Regenerate from Video: re-run diffusion for audio only (video latent masked)
"""

import torch
import node_helpers
import comfy.samplers
import comfy.model_management
import comfy.nested_tensor
from .ltx_sampler import _parse_sigmas
from .common import VideoGenerationContext, UmeBundle, log_node, validate_bundle
from typing import Optional


class UmeAiRT_LTXAudioReplacer:
    """LTX Audio Replacer — replaces or regenerates the audio track.

    Two modes:
    - Replace from File: takes a ComfyUI AUDIO input and swaps it in
    - Regenerate from Video: encodes video frames, runs diffusion for
      audio-only (video latent fully masked), decodes new audio
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_pipe": ("UME_VIDEO_PIPELINE", {"tooltip": "Video pipeline whose audio to replace."}),
                "model_bundle": ("UME_BUNDLE", {"tooltip": "From LTX Loader (needed for audio VAE and model)."}),
                "mode": (["Replace from File", "Regenerate from Video"], {
                    "default": "Replace from File",
                    "tooltip": "Replace: swap in external audio. Regenerate: re-run diffusion for audio only.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "External audio input (from ComfyUI's LoadAudio node). Used in 'Replace' mode.",
                }),
                "positive": ("POSITIVE", {
                    "forceInput": True,
                    "tooltip": "Audio generation prompt (for 'Regenerate' mode). Describes the desired audio.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed for audio regeneration.",
                    "advanced": True,
                }),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "replace_audio"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Replaces the audio track in a video pipeline. Can swap in an external audio file or regenerate audio from video frames."

    def replace_audio(self,
                      video_pipe: VideoGenerationContext,
                      model_bundle: UmeBundle,
                      mode: str = "Replace from File",
                      audio=None,
                      positive: str = None,
                      seed: int = 0):
        """Replace or regenerate the audio track."""

        if video_pipe.frames is None or video_pipe.frames.shape[0] == 0:
            raise ValueError("Audio Replacer: No frames in the video pipeline.")

        if mode == "Replace from File":
            return self._replace_from_file(video_pipe, audio)
        else:
            return self._regenerate(video_pipe, model_bundle, positive, seed)

    def _replace_from_file(self, video_pipe, audio):
        """Replace audio with an external AUDIO input."""

        if audio is None:
            raise ValueError("Audio Replacer: 'Replace from File' mode requires an AUDIO input. "
                             "Connect a LoadAudio node.")

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        fps = video_pipe.fps
        video_duration = video_pipe.frames.shape[0] / fps
        target_samples = int(video_duration * sample_rate)

        # Trim or pad waveform to match video duration
        current_samples = waveform.shape[-1]
        if current_samples > target_samples:
            # Trim
            waveform = waveform[..., :target_samples]
            log_node(f"  Trimmed audio: {current_samples} → {target_samples} samples "
                     f"({current_samples / sample_rate:.1f}s → {video_duration:.1f}s)", color="GREEN")
        elif current_samples < target_samples:
            # Pad with silence
            pad_size = target_samples - current_samples
            padding = torch.zeros(
                *waveform.shape[:-1], pad_size,
                dtype=waveform.dtype, device=waveform.device,
            )
            waveform = torch.cat([waveform, padding], dim=-1)
            log_node(f"  Padded audio: {current_samples} → {target_samples} samples "
                     f"(+{pad_size / sample_rate:.1f}s silence)", color="GREEN")

        video_pipe.audio = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

        log_node(f"🔊 Audio Replacer: Replaced audio — {target_samples} samples @ "
                 f"{sample_rate}Hz ({video_duration:.1f}s)", color="GREEN")

        return (video_pipe,)

    def _regenerate(self, video_pipe, model_bundle, positive, seed):
        """Regenerate audio by running diffusion with video latent masked."""

        validate_bundle(model_bundle, ["model", "clip", "vae"], context="Audio Replacer (Regenerate)")

        audio_vae = model_bundle.audio_vae
        if audio_vae is None:
            raise ValueError("Audio Replacer: 'Regenerate' mode requires an audio VAE in the model bundle. "
                             "Make sure to use the LTX Loader with an audio VAE.")

        model = model_bundle.model
        clip = model_bundle.clip
        vae = model_bundle.vae
        fps = video_pipe.fps
        frames = video_pipe.frames

        pos_text = positive if isinstance(positive, str) else ""

        log_node(f"🔊 Audio Replacer (Regenerate): {frames.shape[0]} frames, "
                 f"encoding video + generating audio...", color="CYAN")

        # --- 1. Encode prompts ---
        tokens_pos = clip.tokenize(pos_text)
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
        positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]

        # ConditioningZeroOut for negative
        negative_cond = []
        for t in positive_cond:
            d = t[1].copy()
            pooled = d.get("pooled_output", None)
            if pooled is not None:
                d["pooled_output"] = torch.zeros_like(pooled)
            negative_cond.append([torch.zeros_like(t[0]), d])

        positive_cond = node_helpers.conditioning_set_values(positive_cond, {"frame_rate": float(fps)})
        negative_cond = node_helpers.conditioning_set_values(negative_cond, {"frame_rate": float(fps)})

        # --- 2. Encode video frames to latent (fixed, no denoise) ---
        log_node("  Encoding video frames to latent...", color="CYAN")
        video_latent = vae.encode(frames[:, :, :, :3])
        log_node(f"  Video latent: {video_latent.shape}", color="GREEN")

        # Create video noise mask = 0 (fully fixed)
        video_noise_mask = torch.zeros(
            (1, 1, video_latent.shape[2], 1, 1),
            dtype=torch.float32,
            device=video_latent.device,
        )
        video_latent_dict = {"samples": video_latent, "noise_mask": video_noise_mask}

        # --- 3. Create empty audio latent (full denoise) ---
        frame_count = frames.shape[0]
        z_channels = audio_vae.latent_channels
        audio_freq = audio_vae.first_stage_model.latent_frequency_bins
        num_audio_latents = audio_vae.first_stage_model.num_of_latents_from_frames(frame_count, fps)
        audio_latent = torch.zeros(
            (1, z_channels, num_audio_latents, audio_freq),
            device=comfy.model_management.intermediate_device(),
        )
        audio_latent_dict = {"samples": audio_latent, "type": "audio"}
        log_node(f"  Audio latent: {audio_latent.shape}", color="GREEN")

        # --- 4. Combine AV latent ---
        av_samples = comfy.nested_tensor.NestedTensor(
            (video_latent_dict["samples"], audio_latent_dict["samples"])
        )
        av_latent = {"samples": av_samples}

        # Video mask = 0 (fixed), audio mask = 1 (denoise)
        audio_noise_mask = torch.ones_like(audio_latent)
        av_latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (video_noise_mask.expand_as(video_latent), audio_noise_mask)
        )

        # --- 5. Diffusion (audio only) ---
        sigmas = _parse_sigmas("standard", "pass1", "")
        log_node(f"  Running diffusion for audio ({len(sigmas) - 1} steps)...", color="CYAN")

        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive=positive_cond, negative=negative_cond)
        guider.set_cfg(1.0)

        sampler = comfy.samplers.sampler_object("euler")
        noise = comfy.samplers.prepare_noise(av_samples, seed)

        sampled = guider.sample(
            noise, av_samples, sampler, sigmas,
            denoise_mask=av_latent.get("noise_mask", None),
            disable_pbar=False,
        )

        log_node("  Diffusion: ✅ Complete", color="GREEN")

        # --- 6. Extract and decode audio ---
        all_latents = sampled.unbind()
        audio_lat = all_latents[1] if len(all_latents) > 1 else None

        if audio_lat is None:
            raise RuntimeError("Audio Replacer: Failed to extract audio latent from diffusion output.")

        log_node("  Decoding audio...", color="CYAN")
        audio_decoded = audio_vae.decode(audio_lat).movedim(-1, 1).to(audio_lat.device)
        output_sample_rate = audio_vae.first_stage_model.output_sample_rate

        video_pipe.audio = {
            "waveform": audio_decoded,
            "sample_rate": int(output_sample_rate),
        }
        video_pipe.audio_vae = audio_vae

        log_node(f"🔊 Audio Replacer: ✅ Regenerated — {audio_decoded.shape}, "
                 f"{output_sample_rate}Hz", color="GREEN")

        return (video_pipe,)
