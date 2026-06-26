"""
UmeAiRT Toolkit - Block Nodes (re-export shim)
-----------------------------------------------
This module re-exports all Block node classes from their sub-modules.
The actual implementations live in:
  - block_inputs.py:  LoRA, ControlNet, Settings, Image, Prompts
  - block_loaders.py: Model Loaders, BundleAutoLoader
  - block_sampler.py: BlockSampler (hub node)
"""

from .block_inputs import (
    UmeAiRT_LoraBlock_1, UmeAiRT_LoraBlock_3, UmeAiRT_LoraBlock_5, UmeAiRT_LoraBlock_10,
    UmeAiRT_WanLoraBlock_1, UmeAiRT_WanLoraBlock_3, UmeAiRT_WanLoraBlock_5, UmeAiRT_WanLoraBlock_10,
    UmeAiRT_ControlNetImageApply,
    UmeAiRT_GenerationSettings,
    UmeAiRT_VideoSettings,
    UmeAiRT_LTXVideoSettings,
    UmeAiRT_BlockImageLoader, UmeAiRT_BlockVideoLoader, UmeAiRT_BlockImageProcess,
    UmeAiRT_ImageProcess_Img2Img, UmeAiRT_ImageProcess_Inpaint, UmeAiRT_ImageProcess_Outpaint, UmeAiRT_ImageProcess_Kontext, UmeAiRT_ImageProcess_Edit,
    UmeAiRT_Positive_Input, UmeAiRT_Negative_Input,
)

from .block_loaders import (
    UmeAiRT_FilesSettings_Checkpoint,
    UmeAiRT_FilesSettings_FLUX,
    UmeAiRT_FilesSettings_ZIMG,
    UmeAiRT_FilesSettings_QWEN,
    UmeAiRT_FilesSettings_ANIMA,
    UmeAiRT_FilesSettings_HiDream,
    UmeAiRT_FilesSettings_WAN,
    UmeAiRT_FilesSettings_LTX,
    UmeAiRT_BundleLoader,
)

from .block_sampler import (
    UmeAiRT_BlockSampler,
)

from .block_passthrough import (
    UmeAiRT_PackPipeline,
)

from .block_lightning import (
    UmeAiRT_LightningAccelerator,
)

from .video_sampler import (
    UmeAiRT_VideoGenerator,
)

# Backward-compatible alias: LTXVideoGenerator → unified VideoGenerator
UmeAiRT_LTXVideoGenerator = UmeAiRT_VideoGenerator

from .video_vace_prep import (
    UmeAiRT_VideoVacePrep,
)

from .video_lightning import (
    UmeAiRT_VideoLightningAccelerator,
)

from .video_optimization import (
    UmeAiRT_VideoOptimization,
)

from .video_output import (
    UmeAiRT_VideoOutput,
)

from .video_postprod import (
    UmeAiRT_VideoFrameInterpolation,
)

from .video_extender import (
    UmeAiRT_VideoExtender,
)

# Backward-compatible alias: LTXVideoExtender → unified VideoExtender
UmeAiRT_LTXVideoExtender = UmeAiRT_VideoExtender

from .video_looper import (
    UmeAiRT_VideoLooper,
)

from .video_funcontrol import (
    UmeAiRT_VideoControlNetApply,
)

