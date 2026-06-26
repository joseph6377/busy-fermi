"""
UmeAiRT Toolkit - Video Slicer
---------------------------------
Generic post-processing node that extracts a frame range from an existing
video pipeline. Works with both LTX and WAN pipelines.

Trims both video frames and audio waveform proportionally.
"""

import torch
from .common import VideoGenerationContext, log_node, resize_tensor


class UmeAiRT_VideoSlicer:
    """Video Slicer — trims a video pipeline to a specific time range.

    Extracts frames between start_time and end_time (in seconds),
    and trims the audio waveform proportionally if present.
    Pure post-processing — no diffusion, no models required.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_pipe": ("UME_VIDEO_PIPELINE", {"tooltip": "Video pipeline to slice."}),
                "start_time": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                    "tooltip": "Start time in seconds. Frames before this are removed.",
                }),
                "end_time": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 3600.0, "step": 0.1,
                    "tooltip": "End time in seconds. Use -1 for end of video. Frames after this are removed.",
                }),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "slice"
    CATEGORY = "UmeAiRT/Post-Process"
    DESCRIPTION = "Trims a video to a specific time range (start/end in seconds). Works with both LTX and WAN pipelines."

    def slice(self, video_pipe: VideoGenerationContext, start_time: float = 0.0, end_time: float = -1.0):
        """Slice the video pipeline to the specified time range."""

        frames = video_pipe.frames
        if frames is None or frames.shape[0] == 0:
            raise ValueError("Video Slicer: No frames in the video pipeline.")

        fps = video_pipe.fps
        total_frames = frames.shape[0]
        total_duration = total_frames / fps

        # --- Compute frame indices ---
        start_frame = max(0, int(start_time * fps))
        if end_time < 0 or end_time > total_duration:
            end_frame = total_frames
        else:
            end_frame = min(total_frames, int(end_time * fps))

        # Validate range
        if start_frame >= end_frame:
            raise ValueError(
                f"Video Slicer: Invalid range — start ({start_time:.1f}s = frame {start_frame}) "
                f">= end ({end_time:.1f}s = frame {end_frame}). "
                f"Video has {total_frames} frames ({total_duration:.1f}s)."
            )

        if start_frame >= total_frames:
            raise ValueError(
                f"Video Slicer: start_time ({start_time:.1f}s) exceeds video duration ({total_duration:.1f}s)."
            )

        # --- Slice frames ---
        sliced_frames = frames[start_frame:end_frame]
        new_frame_count = sliced_frames.shape[0]
        new_duration = new_frame_count / fps

        log_node(f"✂️ Video Slicer: frames [{start_frame}:{end_frame}] of {total_frames} "
                 f"({start_time:.1f}s–{end_time:.1f}s → {new_duration:.1f}s, {new_frame_count} frames)",
                 color="GREEN")

        # --- Slice audio ---
        if video_pipe.audio is not None:
            try:
                waveform = video_pipe.audio["waveform"]
                sample_rate = video_pipe.audio["sample_rate"]
                total_audio_samples = waveform.shape[-1]

                audio_start = int(start_time * sample_rate)
                if end_time < 0:
                    audio_end = total_audio_samples
                else:
                    audio_end = min(total_audio_samples, int(end_time * sample_rate))

                audio_start = max(0, min(audio_start, total_audio_samples))
                audio_end = max(audio_start, min(audio_end, total_audio_samples))

                sliced_waveform = waveform[..., audio_start:audio_end]
                video_pipe.audio = {
                    "waveform": sliced_waveform,
                    "sample_rate": sample_rate,
                }
                log_node(f"  🔊 Audio sliced: {audio_end - audio_start} samples "
                         f"({(audio_end - audio_start) / sample_rate:.1f}s)", color="GREEN")
            except Exception as e:
                log_node(f"  ⚠️ Audio slicing failed: {e}", color="YELLOW")

        # --- Update context ---
        video_pipe.frames = sliced_frames
        video_pipe.frame_count = new_frame_count
        video_pipe.duration = new_duration

        return (video_pipe,)


class UmeAiRT_VideoConcat:
    """Video Concatenate — merges two video pipelines together.

    Appends video_pipe_2 after video_pipe_1.
    If the videos have different resolutions, the second video will be resized to match the first.
    If the videos have different frame rates, the second video is played at the frame rate determined by fps_source.
    Audio is combined and padded with silence if only one video has audio.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_pipe_1": ("UME_VIDEO_PIPELINE", {"tooltip": "The first video pipeline (plays first)."}),
                "video_pipe_2": ("UME_VIDEO_PIPELINE", {"tooltip": "The second video pipeline (plays second)."}),
                "fps_source": (["first_video", "second_video"], {
                    "default": "first_video",
                    "tooltip": "Which video's FPS to use for the combined video. Playing the other video at this FPS may speed it up or slow it down.",
                }),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "concat"
    CATEGORY = "UmeAiRT/Post-Process"
    DESCRIPTION = "Merges two videos together, playing the second video after the first. Automatically resizes mismatched resolutions and handles audio concatenation/padding."

    def concat(self, video_pipe_1: VideoGenerationContext, video_pipe_2: VideoGenerationContext, fps_source: str = "first_video"):
        """Concatenate two video pipelines."""

        frames_1 = video_pipe_1.frames
        frames_2 = video_pipe_2.frames

        if frames_1 is None or frames_1.shape[0] == 0:
            raise ValueError("Video Concatenate: First video pipeline has no frames.")
        if frames_2 is None or frames_2.shape[0] == 0:
            raise ValueError("Video Concatenate: Second video pipeline has no frames.")

        fps_1 = video_pipe_1.fps
        fps_2 = video_pipe_2.fps

        # Determine output FPS
        final_fps = fps_1 if fps_source == "first_video" else fps_2
        if abs(fps_1 - fps_2) > 0.001:
            log_node(
                f"⚠️ Video Concatenate: Videos have different frame rates (Video 1: {fps_1:.2f} FPS, "
                f"Video 2: {fps_2:.2f} FPS). Combined video will use {final_fps:.2f} FPS.",
                color="YELLOW"
            )

        # Check resolution differences and resize video 2 if needed
        h1, w1 = frames_1.shape[1], frames_1.shape[2]
        h2, w2 = frames_2.shape[1], frames_2.shape[2]

        if h1 != h2 or w1 != w2:
            log_node(
                f"🔄 Video Concatenate: Resolution mismatch (Video 1: {w1}x{h1}, Video 2: {w2}x{h2}). "
                f"Resizing Video 2 to match Video 1 ({w1}x{h1}).",
                color="CYAN"
            )
            # resize_tensor expects (tensor, target_h, target_w)
            resized_frames_2 = resize_tensor(frames_2, h1, w1, interp_mode="bilinear")
        else:
            resized_frames_2 = frames_2

        # Concatenate frames along batch dimension
        merged_frames = torch.cat([frames_1, resized_frames_2], dim=0)
        new_frame_count = merged_frames.shape[0]
        new_duration = new_frame_count / final_fps

        # Audio Concatenation
        merged_audio = None
        audio_1 = video_pipe_1.audio
        audio_2 = video_pipe_2.audio

        duration_1 = frames_1.shape[0] / fps_1
        duration_2 = frames_2.shape[0] / fps_2

        if audio_1 is not None and audio_2 is not None:
            try:
                waveform_1 = audio_1["waveform"]
                sample_rate_1 = audio_1["sample_rate"]

                waveform_2 = audio_2["waveform"]
                sample_rate_2 = audio_2["sample_rate"]

                # 1. Resample waveform_2 if sample rates differ
                if sample_rate_1 != sample_rate_2:
                    log_node(
                        f"🔄 Video Concatenate: Audio sample rate mismatch (Video 1: {sample_rate_1}Hz, "
                        f"Video 2: {sample_rate_2}Hz). Resampling Video 2 audio to {sample_rate_1}Hz.",
                        color="CYAN"
                    )
                    # waveform is [1, channels, samples], interpolate expects [batch, channels, length]
                    target_samples = int(waveform_2.shape[-1] * sample_rate_1 / sample_rate_2)
                    waveform_2 = torch.nn.functional.interpolate(
                        waveform_2,
                        size=target_samples,
                        mode="linear",
                        align_corners=False
                    )
                    sample_rate_2 = sample_rate_1

                # 2. Match channel counts (mono [1, 1, S] <-> stereo [1, 2, S])
                c1 = waveform_1.shape[1]
                c2 = waveform_2.shape[1]
                if c1 != c2:
                    if c1 == 1 and c2 == 2:
                        waveform_1 = waveform_1.repeat(1, 2, 1)
                        c1 = 2
                    elif c1 == 2 and c2 == 1:
                        waveform_2 = waveform_2.repeat(1, 2, 1)
                        c2 = 2
                    else:
                        # Fallback for incompatible channel counts (e.g. 5.1 and stereo): pad/truncate
                        min_channels = min(c1, c2)
                        waveform_1 = waveform_1[:, :min_channels, :]
                        waveform_2 = waveform_2[:, :min_channels, :]
                        c1 = c2 = min_channels

                # 3. Concatenate waveforms along the sample dimension (dim=-1)
                merged_waveform = torch.cat([waveform_1, waveform_2], dim=-1)
                merged_audio = {
                    "waveform": merged_waveform,
                    "sample_rate": sample_rate_1
                }
                log_node("🔊 Video Concatenate: Audio streams merged successfully.", color="GREEN")
            except Exception as e:
                log_node(f"⚠️ Video Concatenate: Audio merging failed: {e}", color="YELLOW")
                # Fallback to no audio or audio_1
                merged_audio = audio_1

        elif audio_1 is not None:
            # Only video 1 has audio -> pad video 2 with silence
            try:
                waveform_1 = audio_1["waveform"]
                sample_rate_1 = audio_1["sample_rate"]
                channels = waveform_1.shape[1]
                silence_samples = int(duration_2 * sample_rate_1)

                silence = torch.zeros(
                    (1, channels, silence_samples),
                    dtype=waveform_1.dtype,
                    device=waveform_1.device
                )
                merged_waveform = torch.cat([waveform_1, silence], dim=-1)
                merged_audio = {
                    "waveform": merged_waveform,
                    "sample_rate": sample_rate_1
                }
                log_node(f"🔊 Video Concatenate: Pad silent audio for second video ({duration_2:.1f}s).", color="GREEN")
            except Exception as e:
                log_node(f"⚠️ Video Concatenate: Silent audio padding for Video 2 failed: {e}", color="YELLOW")
                merged_audio = audio_1

        elif audio_2 is not None:
            # Only video 2 has audio -> pad video 1 with silence at start
            try:
                waveform_2 = audio_2["waveform"]
                sample_rate_2 = audio_2["sample_rate"]
                channels = waveform_2.shape[1]
                silence_samples = int(duration_1 * sample_rate_2)

                silence = torch.zeros(
                    (1, channels, silence_samples),
                    dtype=waveform_2.dtype,
                    device=waveform_2.device
                )
                merged_waveform = torch.cat([silence, waveform_2], dim=-1)
                merged_audio = {
                    "waveform": merged_waveform,
                    "sample_rate": sample_rate_2
                }
                log_node(f"🔊 Video Concatenate: Pad silent audio for first video ({duration_1:.1f}s).", color="GREEN")
            except Exception as e:
                log_node(f"⚠️ Video Concatenate: Silent audio padding for Video 1 failed: {e}", color="YELLOW")
                merged_audio = audio_2

        # Create output context
        ctx = VideoGenerationContext()
        # Copy fields from first video
        for key, val in video_pipe_1.__dict__.items():
            setattr(ctx, key, val)

        ctx.frames = merged_frames
        ctx.frame_count = new_frame_count
        ctx.fps = final_fps
        ctx.duration = new_duration
        ctx.audio = merged_audio

        # Update width/height to match video_pipe_1 resolution
        ctx.width = w1
        ctx.height = h1

        log_node(f"🎬 Video Concatenate: Combined {frames_1.shape[0]} and {frames_2.shape[0]} frames -> "
                 f"{new_frame_count} frames @ {final_fps:.2f}fps ({w1}x{h1}, {new_duration:.2f}s)", color="GREEN")

        return (ctx,)

