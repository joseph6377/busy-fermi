"""
UmeAiRT Toolkit - Video ControlNet Apply (FunControl)
-------------------------------------------------------
Prepares a FunControl conditioning bundle (UME_FUNCONTROL) for WAN video
generation. Accepts a source image and a control video (e.g. pose/depth/canny)
and optionally applies a preprocessor to the control frames.

Design pattern: mirrors UmeAiRT_ControlNetImageApply (block_inputs.py)
but operates on video frames instead of single images, and outputs a
UME_FUNCONTROL bundle instead of modifying a UME_IMAGE bundle.

Usage in the workflow:
  LoadImage(source) ──┐
  LoadVideo(pose)  ──→ VideoControlNetApply ──→ VideoGenerator
                         ↑                          ↑
                    preprocessor: DWPose      model_bundle (I2V)
                    strength: 1.0             video_settings
"""

from .common import UmeFunControl, log_node
from typing import Optional, Any


class UmeAiRT_VideoControlNetApply:
    """Prepares FunControl conditioning for WAN video generation.

    Takes a source image (for CLIP Vision I2V conditioning) and a control
    video (batch of pose/depth/canny frames) for motion guidance. Optionally
    applies a native preprocessor to the control frames.

    Outputs a UME_FUNCONTROL bundle that plugs into the Video Generator's
    optional 'funcontrol' input.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE", {"tooltip": "Source image for I2V conditioning. This image provides the visual content/style."}),
                "control_video": ("IMAGE", {"tooltip": "Control frames batch (e.g. from a loaded video). Each frame guides the motion/structure of the corresponding generated frame."}),
                "preprocessor": (["None", "UmeAiRT_DWPose", "UmeAiRT_Canny", "UmeAiRT_Depth"], {
                    "default": "None",
                    "tooltip": "Apply a native preprocessor to the control_video frames. Use DWPose for pose-driven animation, Canny for edge-driven, Depth for depth-driven."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider",
                    "tooltip": "How strongly the control video guides the output. 1.0 = full control, 0.5 = soft guidance."
                }),
            },
        }

    RETURN_TYPES = ("UME_FUNCONTROL", "IMAGE")
    RETURN_NAMES = ("funcontrol", "control_preview")
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Prepares FunControl conditioning (source image + control video) for WAN video generation."

    def process(self, source_image, control_video, preprocessor="None", strength=1.0):
        """Build a UME_FUNCONTROL bundle from source image and control video."""

        if source_image is None:
            raise ValueError("Video ControlNet Apply: source_image is required.")
        if control_video is None:
            raise ValueError("Video ControlNet Apply: control_video is required.")

        control_frames = control_video
        log_node(f"🎮 Video ControlNet: {control_frames.shape[0]} control frames, "
                 f"strength={strength:.2f}", color="CYAN")

        # --- Apply preprocessor if requested ---
        if preprocessor != "None":
            control_frames = self._apply_preprocessor(control_frames, preprocessor)

        # --- Build bundle ---
        bundle = UmeFunControl(
            source_image=source_image,
            control_video=control_frames,
            strength=strength,
        )

        log_node(f"🎮 Video ControlNet: ✅ Bundle ready "
                 f"({control_frames.shape[0]} frames, preprocessor={preprocessor})", color="GREEN")

        return (bundle, control_frames)

    def _apply_preprocessor(self, frames, preprocessor):
        """Apply a native preprocessor to each frame of the control video.

        Falls back to the raw frames if the preprocessor fails.
        """
        if preprocessor == "UmeAiRT_Canny":
            try:
                from .preprocessors.canny_core import apply_canny
                log_node("  Running native Canny on control frames...", color="CYAN")
                processed = apply_canny(frames, low_threshold=100, high_threshold=200)
                log_node(f"  Canny: processed {processed.shape[0]} frames", color="GREEN")
                return processed
            except Exception as e:
                log_node(f"  Canny preprocessor failed: {e}. Using raw frames.", color="YELLOW")

        elif preprocessor == "UmeAiRT_Depth":
            try:
                import os
                import folder_paths
                from .manifest import download_bundle_files
                log_node("  Fetching Depth Model...", color="CYAN")
                resolved, _, _, _, errs = download_bundle_files("PREPROCESSORS/Depth", "Zoe-N")
                if errs:
                    log_node(f"  Failed to download Depth Model: {errs}", color="RED")
                else:
                    model_path = os.path.join(folder_paths.models_dir, "preprocessors", "depth", "Intel-zoedepth-nyu-kitti")
                    from .preprocessors.depth_core import apply_zoedepth
                    log_node("  Running native Depth on control frames...", color="CYAN")
                    processed = apply_zoedepth(frames, model_path)
                    log_node(f"  Depth: processed {processed.shape[0]} frames", color="GREEN")
                    return processed
            except Exception as e:
                log_node(f"  Depth preprocessor failed: {e}. Using raw frames.", color="YELLOW")

        elif preprocessor == "UmeAiRT_DWPose":
            try:
                import os
                import folder_paths
                from .manifest import download_bundle_files
                log_node("  Fetching DWPose Model...", color="CYAN")
                resolved, _, _, _, errs = download_bundle_files("PREPROCESSORS/Pose", "DWPose")
                if errs:
                    log_node(f"  Failed to download DWPose Model: {errs}", color="RED")
                else:
                    model_path_det = os.path.join(folder_paths.models_dir, "models_base", "yolox_l.onnx")
                    model_path_pose = os.path.join(folder_paths.models_dir, "models_base", "dw-ll_ucoco_384.onnx")
                    from .preprocessors.dwpose_core import apply_dwpose
                    log_node("  Running native DWPose on control frames...", color="CYAN")
                    processed = apply_dwpose(frames, model_path_det, model_path_pose)
                    log_node(f"  DWPose: processed {processed.shape[0]} frames", color="GREEN")
                    return processed
            except Exception as e:
                log_node(f"  DWPose preprocessor failed: {e}. Using raw frames.", color="YELLOW")

        return frames
