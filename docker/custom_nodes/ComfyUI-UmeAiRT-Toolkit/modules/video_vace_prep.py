"""
UmeAiRT Toolkit - Video VACE Prep
------------------------------------
Preparation node for VACE (Video All-in-One Creation and Editing) workflows.

Takes a start frame and optional end frame, packages them into a
UME_VACE_FRAMES bundle for the Video Generator, which will construct
the actual control_video/control_masks at sampling time.

Pattern: same as ImageProcess_Kontext → BlockSampler.
"""

from .common import UmeVaceFrames, log_node


class UmeAiRT_VideoVacePrep:
    """Prepares Start+End frame conditioning for VACE video generation.

    Takes a start frame and optional end frame, and outputs a
    UME_VACE_FRAMES bundle ready for the Video Generator.

    The Video Generator automatically constructs the VACE control_video
    and control_masks tensors at sampling time when the target
    resolution and frame count are known.

    When only a start frame is provided, the model generates a video
    starting from that frame with free-form continuation.
    When both start and end frames are provided, the model generates
    a transition between the two frames.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_image": ("IMAGE", {"tooltip": "The first frame of the video. The generated video will begin from this image."}),
            },
            "optional": {
                "end_image": ("IMAGE", {"tooltip": "The last frame of the video. When connected, the model generates a smooth transition from start to end frame."}),
                "color_match": ("BOOLEAN", {"default": True, "advanced": True,
                    "tooltip": "Match output video colors to the start frame reference. Corrects color drift that can occur during VACE generation. Recommended ON."}),
            }
        }
    RETURN_TYPES = ("UME_VACE_FRAMES",)
    RETURN_NAMES = ("vace_frames",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Video"
    DESCRIPTION = "Prepares start and end frames for VACE video generation. Connect to the Video Generator's vace_frames input."

    def process(self, start_image, end_image=None, color_match=True):
        """Package start/end images into a UmeVaceFrames bundle.

        Images are stored as-is; resizing to target dimensions is deferred
        to the Video Generator which knows the resolution from VideoSettings.
        """
        mode = "Start+End" if end_image is not None else "Start Only"
        log_node(
            f"🎬 Video VACE Prep: {mode} mode | "
            f"ColorMatch={'ON' if color_match else 'OFF'}",
            color="CYAN"
        )

        return (UmeVaceFrames(
            start_image=start_image,
            end_image=end_image,
            color_match=color_match,
        ),)

