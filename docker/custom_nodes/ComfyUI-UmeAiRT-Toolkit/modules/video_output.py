"""
UmeAiRT Toolkit - Video Output
---------------------------------
Final output node for the video pipeline. Takes a VideoGenerationContext
(from the Video Generator) and:

1. Saves the video using native PyAV
2. Embeds generation metadata (A1111-compatible params, video-specific info)
3. Embeds ComfyUI workflow JSON for drag & drop reload
4. Returns frames for ComfyUI preview

Metadata is stored in the video container's metadata fields via PyAV.
Note: MP4 only supports a few predefined metadata keys (comment, description).
All metadata is packed into 'comment' as JSON for cross-format compatibility.
"""

import os
import json
import av
import torch
import folder_paths
from fractions import Fraction
from datetime import datetime
from .common import VideoGenerationContext, log_node


# Version stamp for metadata
try:
    from .. import __version__ as TOOLKIT_VERSION
except Exception:  # Non-critical: version stamp fallback
    TOOLKIT_VERSION = "unknown"


def _resolve_filename_prefix(prefix, ctx):
    """Replace %placeholder tokens in the filename prefix."""
    placeholders = {
        "%date": datetime.now().strftime("%Y-%m-%d"),
        "%time": datetime.now().strftime("%H%M%S"),
        "%seed": str(ctx.seed),
        "%width": str(ctx.width),
        "%height": str(ctx.height),
        "%steps": str(ctx.steps),
        "%cfg": str(ctx.cfg),
        "%model": os.path.splitext(ctx.model_name)[0] if ctx.model_name else "unknown",
        "%sampler": ctx.sampler_name,
        "%scheduler": ctx.scheduler,
        "%duration": str(ctx.duration),
    }
    
    for key, value in placeholders.items():
        prefix = prefix.replace(key, value)
    return prefix


def _build_generation_metadata(ctx):
    """Build a complete generation metadata dict from the VideoGenerationContext."""
    meta = {
        "prompt": ctx.positive_prompt,
        "negative_prompt": ctx.negative_prompt,
        "model": ctx.model_name,
        "seed": ctx.seed,
        "steps": ctx.steps,
        "cfg": ctx.cfg,
        "shift": ctx.shift,
        "sampler": ctx.sampler_name,
        "scheduler": ctx.scheduler,
        "width": ctx.width,
        "height": ctx.height,
        "duration": ctx.duration,
        "fps": ctx.fps,
        "frame_count": ctx.frame_count,
        "denoise": ctx.denoise,
        "loader_type": ctx.loader_type,
        "toolkit_version": TOOLKIT_VERSION,
    }
    if ctx.loras:
        meta["loras"] = [{"name": n, "strength": s} for n, s in ctx.loras]
    return meta


def _build_a1111_params(ctx):
    """Build a lightweight A1111-compatible params string (no hashing, no network)."""
    pos = str(ctx.positive_prompt or "").strip()
    neg = str(ctx.negative_prompt or "").strip()
    basemodelname = os.path.splitext(os.path.basename(ctx.model_name or ""))[0]
    lora_str = ""
    if ctx.loras:
        lora_parts = [f"{n}:{s:.2f}" for n, s in ctx.loras]
        lora_str = f", Lora: {', '.join(lora_parts)}"
    return (
        f"{pos}\n"
        f"Negative prompt: {neg}\n"
        f"Steps: {ctx.steps}, Sampler: {ctx.sampler_name}, "
        f"Scheduler: {ctx.scheduler}, CFG scale: {ctx.cfg}, "
        f"Seed: {ctx.seed}, Size: {ctx.width}x{ctx.height}, "
        f"Model: {basemodelname}{lora_str}, Version: ComfyUI UmeAiRT v{TOOLKIT_VERSION}"
    )


class UmeAiRT_VideoOutput:
    """Video output node — saves generated video with optional frame interpolation
    and generation metadata embedding.

    Supports WebM (VP9/AV1) and MP4 output via native PyAV.
    RIFE frame interpolation doubles the frame rate for smoother playback.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_pipe": ("UME_VIDEO_PIPELINE", {"tooltip": "Video pipeline from the Video Generator node."}),
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "%date/%time_WAN_%seed",
                    "tooltip": "Output filename with placeholders: %date, %time, %seed, %model, %steps, %cfg, etc."
                }),
                "format": (["mp4", "webm"], {
                    "default": "mp4", "advanced": True,
                    "tooltip": "Video container format."
                }),
                "quality": (["Visually Lossless", "High Quality", "Standard"], {
                    "default": "Visually Lossless", "advanced": True,
                    "tooltip": "Video Quality compression level."
                }),
                "embed_metadata": ("BOOLEAN", {
                    "default": True, "advanced": True,
                    "tooltip": "Embed generation parameters and ComfyUI workflow data in the video file metadata. Required for drag & drop workflow reload."
                }),
                "save_last_frame": ("BOOLEAN", {
                    "default": False, "advanced": True,
                    "tooltip": "Save the very last frame of the video as an image file. Useful for chaining Image-to-Video generations."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Video"
    OUTPUT_NODE = True
    DESCRIPTION = "Encodes and saves generated frames into a video file (e.g. MP4, WebM)."

    def process(self, video_pipe: VideoGenerationContext,
                filename_prefix="%date/%time_WAN_%seed",
                format="mp4", quality="Visually Lossless",
                embed_metadata=True, save_last_frame=False,
                prompt=None, extra_pnginfo=None):
        """Save the generated video."""

        ctx = video_pipe
        frames = ctx.frames

        if frames is None or frames.shape[0] == 0:
            raise ValueError("Video Output: No frames in the video pipeline. Check your Video Generator.")

        output_fps = ctx.fps
        log_node(f"📹 Video Output: {frames.shape[0]} frames, {ctx.width}x{ctx.height}", color="CYAN")
        
        # Map quality to CRF
        crf_map = {
            "Visually Lossless": 19,
            "High Quality": 23,
            "Standard": 28
        }
        crf = crf_map.get(quality, 19)

        # --- 2. Resolve output path ---
        resolved_prefix = _resolve_filename_prefix(filename_prefix, ctx)
        ext = "webm" if format == "webm" else "mp4"
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            resolved_prefix, folder_paths.get_output_directory(),
            frames.shape[2], frames.shape[1]
        )
        file = f"{filename}_{counter:05}_.{ext}"
        full_path = os.path.join(full_output_folder, file)

        # --- 3. Save video with PyAV ---
        log_node(f"  Encoding: {file} ({format}, CRF={crf})...", color="CYAN")

        container = av.open(full_path, mode="w")

        # Embed generation metadata
        if embed_metadata:
            a111_params = _build_a1111_params(ctx)

            # MP4 only supports a few predefined metadata keys (comment, description)
            # and silently discards custom keys. WebM supports arbitrary tags.
            # Strategy: pack everything into 'comment' as JSON for cross-format
            # compatibility, then also write native keys where supported.
            meta_bundle = {
                "parameters": a111_params,
                "umeairt": _build_generation_metadata(ctx),
            }
            if prompt is not None:
                meta_bundle["prompt"] = prompt
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    meta_bundle[k] = v

            # 'comment' is the only reliable free-text field across all containers
            container.metadata["comment"] = json.dumps(meta_bundle, separators=(',', ':'))

            if format == "webm":
                # WebM supports arbitrary keys natively
                container.metadata["parameters"] = a111_params
                container.metadata["umeairt"] = json.dumps(_build_generation_metadata(ctx))
                if prompt is not None:
                    container.metadata["prompt"] = json.dumps(prompt)
                if extra_pnginfo is not None:
                    for k, v in extra_pnginfo.items():
                        container.metadata[k] = json.dumps(v)
            else:
                # MP4: 'description' also survives
                container.metadata["description"] = a111_params

        # Configure codec
        if format == "webm":
            codec = "libvpx-vp9"
            pix_fmt = "yuv420p"
        else:  # mp4
            codec = "libx264"
            pix_fmt = "yuv420p"

        stream = container.add_stream(codec, rate=Fraction(round(output_fps * 1000), 1000))
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = pix_fmt
        stream.bit_rate = 0
        stream.options = {"crf": str(crf)}

        for frame_tensor in frames:
            frame_np = torch.clamp(frame_tensor[..., :3] * 255, min=0, max=255).to(
                device=torch.device("cpu"), dtype=torch.uint8
            ).numpy()
            video_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        # Flush video
        for packet in stream.encode():
            container.mux(packet)

        # --- Audio muxing (LTX-2.3) ---
        if ctx.audio is not None:
            try:
                import numpy as np
                waveform = ctx.audio["waveform"]  # [1, channels, samples]
                sample_rate = ctx.audio["sample_rate"]

                # Ensure waveform is on CPU and convert to numpy
                if hasattr(waveform, "cpu"):
                    waveform = waveform.cpu()
                if hasattr(waveform, "numpy"):
                    waveform_np = waveform.squeeze(0).numpy()  # [channels, samples]
                else:
                    waveform_np = waveform

                # Configure audio stream
                audio_codec = "aac" if format != "webm" else "libopus"
                audio_stream = container.add_stream(audio_codec, rate=sample_rate)

                # PyAV expects audio frames as [samples, channels]
                if waveform_np.ndim == 1:
                    waveform_np = waveform_np.reshape(-1, 1)
                elif waveform_np.ndim == 2:
                    waveform_np = waveform_np.T  # [channels, samples] → [samples, channels]

                # Encode audio in chunks
                chunk_size = 1024
                for i in range(0, waveform_np.shape[0], chunk_size):
                    chunk = waveform_np[i:i + chunk_size]
                    audio_frame = av.AudioFrame.from_ndarray(
                        chunk.T.astype(np.float32),  # PyAV wants [channels, samples]
                        format="fltp",
                        layout="mono" if chunk.shape[1] == 1 else "stereo"
                    )
                    audio_frame.sample_rate = sample_rate
                    for packet in audio_stream.encode(audio_frame):
                        container.mux(packet)

                # Flush audio
                for packet in audio_stream.encode():
                    container.mux(packet)

                log_node(f"  🔊 Audio muxed: {waveform_np.shape[0]} samples @ {sample_rate}Hz", color="GREEN")
            except Exception as e:
                log_node(f"  ⚠️ Audio muxing failed: {e}", color="YELLOW")

        container.close()

        log_node(f"  ✅ Saved: {file} ({frames.shape[0]} frames @ {output_fps}fps)", color="GREEN")

        # Save last frame if requested
        if save_last_frame:
            from PIL import Image
            import numpy as np
            last_frame_tensor = frames[-1]
            last_frame_np = torch.clamp(last_frame_tensor[..., :3] * 255, min=0, max=255).to(
                device=torch.device("cpu"), dtype=torch.uint8
            ).numpy()
            img = Image.fromarray(last_frame_np)
            img_file = f"{filename}_{counter:05}__last_frame.png"
            img_path = os.path.join(full_output_folder, img_file)
            img.save(img_path)
            log_node(f"  🖼️ Saved last frame: {img_file}", color="GREEN")

        # --- 4. Return preview frames ---
        format_mime = "video/webm" if format == "webm" else "video/mp4"
        return {"ui": {
                    "images": [{"filename": file, "subfolder": subfolder, "type": "output", "format": format_mime}],
                    "animated": [True]
                },
                "result": (frames,)}
