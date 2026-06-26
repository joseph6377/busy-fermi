"""
UmeAiRT Toolkit - Common Shared State & Utils
---------------------------------------------
Core utilities, typed bundle definitions, and the GenerationContext pipeline object.
"""

import copy
import torch
import torchvision.transforms.functional as TF
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, List, Tuple
from .logger import log_node


# --- Pipeline Parameter Extraction ---

PipelineParams = namedtuple("PipelineParams", [
    "model", "vae", "clip", "steps", "cfg",
    "sampler_name", "scheduler", "seed", "pos_text", "neg_text"
])


def extract_pipeline_params(gen_pipe):
    """Extracts common parameters from a GenerationContext into a PipelineParams tuple.

    Centralizes the repetitive pattern of reading model/vae/clip/steps/cfg/sampler/scheduler/seed/prompts
    from a GenerationContext, used by every pipeline-aware post-processing node.

    Args:
        gen_pipe (GenerationContext): The generation pipeline object.

    Returns:
        PipelineParams: A named tuple with all extracted values.

    Raises:
        ValueError: If model, vae, or clip are missing from the pipeline.
    """
    model = gen_pipe.model
    vae = gen_pipe.vae
    clip = gen_pipe.clip

    if not model or not vae or not clip:
        raise ValueError("Pipeline is missing Model, VAE, or CLIP.")

    return PipelineParams(
        model=model,
        vae=vae,
        clip=clip,
        steps=int(gen_pipe.steps or 20),
        cfg=float(gen_pipe.cfg or 8.0),
        sampler_name=gen_pipe.sampler_name or "euler",
        scheduler=gen_pipe.scheduler or "normal",
        seed=int(gen_pipe.seed or 0),
        pos_text=str(gen_pipe.positive_prompt or ""),
        neg_text=str(gen_pipe.negative_prompt or ""),
    )


def validate_bundle(bundle, required_attrs, context=""):
    """Validates that a dataclass bundle has all required attributes set (non-None).

    Args:
        bundle: The dataclass instance to validate.
        required_attrs (list[str]): Attribute names that must be non-None.
        context (str): Name of the calling node for error messages.

    Raises:
        ValueError: If the bundle is missing required attributes.
    """
    missing = [a for a in required_attrs if getattr(bundle, a, None) is None]
    if missing:
        raise ValueError(f"{context}: Bundle is missing required attributes: {', '.join(missing)}.")


# --- SeedVR2 Known Models ---

KNOWN_DIT_MODELS = [
    "seedvr2_ema_3b-Q4_K_M.gguf", "seedvr2_ema_3b-Q8_0.gguf",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_7b-Q4_K_M.gguf", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    "seedvr2_ema_7b_fp16.safetensors", "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", "seedvr2_ema_7b_sharp_fp16.safetensors",
]


# --- Typed Bundle Definitions ---

@dataclass
class UmeBundle:
    """Type contract for UME_BUNDLE — produced by all Loader nodes."""
    model: Any = None
    model_low_noise: Any = None  # WAN 2.2 MoE: Low-noise expert model (second pass)
    clip: Any = None
    vae: Any = None
    model_name: str = ""
    bundle_type: str = ""
    loader_type: str = ""
    shift: float = 0.0  # ModelSampling shift override (e.g. 2.0 for HiDream)
    overrides: Optional[dict] = None  # Settings overrides injected by middleware nodes (e.g. Lightning Accelerator)
    clip_vision: Any = None  # CLIP Vision model for video I2V pipelines
    audio_vae: Any = None  # LTX-2.3: Audio VAE for audio generation
    latent_upscale_model: Any = None  # LTX-2.3: Spatial upscaler 2x for dual-pass pipeline

    @property
    def is_moe(self) -> bool:
        """True if this bundle uses a Mixture-of-Experts (dual model) architecture (WAN 2.2 14B)."""
        return self.model_low_noise is not None


@dataclass
class UmeVideoSettings:
    """Type contract for UME_VIDEO_SETTINGS — produced by VideoSettings node."""
    width: int = 848
    height: int = 480
    duration: float = 3.0       # Duration in seconds (user-facing; frame_count = int(duration * fps) + 1)
    steps: int = 20
    cfg: float = 6.0
    shift: float = 6.0          # ModelSamplingSD3 shift for WAN models
    sampler_name: str = "uni_pc"
    scheduler: str = "simple"
    seed: int = 0
    frame_rate: int = 16              # LTX=25, WAN=16
    audio_enabled: bool = False       # LTX=True, WAN=False
    sigmas_preset: str = ""           # "", "standard", "fast", "custom"
    custom_sigmas: str = ""           # Comma-separated sigma values for custom preset


@dataclass
class UmeSettings:
    """Type contract for UME_SETTINGS — produced by GenerationSettings."""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 8.0
    sampler_name: str = "euler"
    scheduler: str = "normal"
    seed: int = 0


@dataclass
class UmeImage:
    """Type contract for UME_IMAGE — produced by Image Loader/Process nodes."""
    image: Any = None
    mask: Any = None
    reference_image: Any = None  # Optional 2nd image for Kontext workflows
    edit_images: Optional[List[Any]] = None  # QWEN Edit: additional reference images (up to 2 extra)
    mode: str = "img2img"
    denoise: float = 1.0
    auto_resize: bool = False
    controlnets: List[Tuple] = field(default_factory=list)
    # Outpaint-specific (Solution C: target dimensions + alignment)
    outpaint_target_w: int = 0
    outpaint_target_h: int = 0
    outpaint_h_align: str = "center"
    outpaint_v_align: str = "center"
    outpaint_mask_blur: int = 10


@dataclass
class UmeVaceFrames:
    """Type contract for UME_VACE_FRAMES — produced by VideoVacePrep node.

    Stores raw start/end images; control_video and control_masks are built
    by the VideoGenerator at sampling time when target dimensions are known.
    """
    start_image: Any = None        # Required: first frame (IMAGE tensor)
    end_image: Any = None          # Optional: last frame (IMAGE tensor, None = start-only)
    color_match: bool = True       # Match output colors to start frame


@dataclass
class UmeFunControl:
    """Type contract for UME_FUNCONTROL — produced by VideoControlNetApply node.

    Stores the source image for I2V conditioning and preprocessed control video
    frames (e.g. DWPose, Canny, Depth) for motion guidance via WanFunControlToVideo.
    """
    source_image: Any = None        # Source image for CLIP Vision I2V conditioning
    control_video: Any = None       # Preprocessed control frames [N, H, W, 3]
    strength: float = 1.0           # Control strength applied to the conditioning


# --- GenerationContext (UME_PIPELINE) ---

class GenerationContext:
    """Encapsulates all state for a single generation pipeline.

    Created by the BlockSampler, this object carries models, settings,
    prompts, and the generated image through the post-processing chain.
    """
    def __init__(self):
        # Models
        self.model = None
        self.clip = None
        self.vae = None
        self.model_name = ""

        # Settings
        self.width = 1024
        self.height = 1024
        self.steps = 20
        self.cfg = 8.0
        self.sampler_name = "euler"
        self.scheduler = "normal"
        self.seed = 0
        self.denoise = 1.0
        self.loader_type = ""

        # Prompts
        self.positive_prompt = ""
        self.negative_prompt = ""

        # Generated output
        self.image = None
        self.latent = None

        # Extras
        self.loras = []
        self.controlnets = []
        self.source_image = None
        self.source_mask = None

    def clone(self):
        """Create an independent copy for branched workflows."""
        ctx = copy.copy(self)
        ctx.loras = list(self.loras)
        ctx.controlnets = list(self.controlnets)
        return ctx

    def is_ready(self):
        """Validates that minimum required data is set for sampling."""
        return self.model is not None and self.vae is not None and self.clip is not None


# --- VideoGenerationContext (UME_VIDEO_PIPELINE) ---

class VideoGenerationContext:
    """Encapsulates all state for a single video generation pipeline.

    Created by the VideoGenerator, this object carries models, settings,
    prompts, and the generated video frames through the output chain.
    """
    def __init__(self):
        # Models
        self.model = None
        self.model_low_noise = None
        self.clip = None
        self.vae = None
        self.clip_vision = None
        self.model_name = ""

        # Settings
        self.width = 848
        self.height = 480
        self.duration = 3.0
        self.fps = 16
        self.frame_count = 49  # int(duration * fps) + 1
        self.steps = 20
        self.cfg = 6.0
        self.shift = 6.0
        self.sampler_name = "uni_pc"
        self.scheduler = "simple"
        self.seed = 0
        self.denoise = 1.0
        self.loader_type = "wan"

        # Prompts
        self.positive_prompt = ""
        self.negative_prompt = ""

        # Generated output
        self.frames = None   # IMAGE tensor [N, H, W, C] where N = frame_count

        # Extras
        self.loras = []
        self.source_image = None
        self.audio = None        # {"waveform": tensor, "sample_rate": int} — LTX-2.3
        self.audio_vae = None    # Audio VAE stored for post-prod decode needs

    def clone(self):
        """Create an independent copy for branched workflows."""
        ctx = copy.copy(self)
        ctx.loras = list(self.loras)
        return ctx

    def is_ready(self):
        """Validates that minimum required data is set for video sampling."""
        return self.model is not None and self.vae is not None and self.clip is not None


# --- Tensor Utilities ---

def resize_tensor(tensor, target_h, target_w, interp_mode="bilinear", is_mask=False):
    """Resizes an image or mask tensor to the target dimensions.

    Handles dimension permutations between ComfyUI format (B, H, W, C) 
    and PyTorch format (B, C, H, W) before applying the interpolation.

    Args:
        tensor (torch.Tensor): The input tensor representing an image or a mask.
        target_h (int): The target height in pixels.
        target_w (int): The target width in pixels.
        interp_mode (str, optional): The interpolation mode used by torch.nn.functional.interpolate. Defaults to "bilinear".
        is_mask (bool, optional): If True, treats the input as a mask (B, H, W). Defaults to False.

    Returns:
        torch.Tensor: The resized tensor, returned in its original ComfyUI dimension format.
    """
    if is_mask:
        # Mask: can be [H, W] or [B, H, W]
        original_shape = tensor.shape
        if len(original_shape) == 2:
            t = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        else:
            t = tensor.unsqueeze(1)  # [B, 1, H, W]
    else:
        # Image: [B, H, W, C] -> [B, C, H, W]
        t = tensor.permute(0, 3, 1, 2)
    
    align_corners = False if interp_mode not in ("nearest", "nearest-exact") else None
    t_resized = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode=interp_mode, align_corners=align_corners)
    
    if is_mask:
        if len(original_shape) == 2:
            return t_resized.squeeze(0).squeeze(0)  # [H, W]
        else:
            return t_resized.squeeze(1)  # [B, H, W]
    else:
        # [B, C, H, W] -> [B, H, W, C]
        return t_resized.permute(0, 2, 3, 1)


def encode_prompts(clip, pos_text, neg_text):
    """Encode positive and negative text prompts into conditioning tensors.

    Centralizes the CLIP tokenize → encode → format pattern used across
    all pipeline-aware nodes (FaceDetailer, Upscaler, Detailer Daemon, etc.).

    Args:
        clip: The loaded CLIP model from ComfyUI.
        pos_text (str): The positive prompt text.
        neg_text (str): The negative prompt text.

    Returns:
        tuple: (positive_cond, negative_cond) ready for KSampler.
    """
    tokens = clip.tokenize(pos_text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    positive = [[cond, {"pooled_output": pooled}]]

    tokens = clip.tokenize(neg_text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    negative = [[cond, {"pooled_output": pooled}]]
    return positive, negative


def apply_outpaint_padding(image, mask, pad_l, pad_t, pad_r, pad_b, overlap=8, feathering=40, skip_noise=False, sharp_mirror=False):
    """Apply outpaint padding to an image and generate the corresponding mask.

    Stretches edge pixels outward using 'replicate' padding, creates a mask marking
    the padded regions (with overlap into the original area), and applies Gaussian
    feathering for smooth transitions.

    Args:
        image (torch.Tensor): Input image tensor [B, H, W, C].
        mask (torch.Tensor or None): Optional existing mask [B, H, W].
        pad_l (int): Left padding in pixels.
        pad_t (int): Top padding in pixels.
        pad_r (int): Right padding in pixels.
        pad_b (int): Bottom padding in pixels.
        overlap (int, optional): Overlap into original image in pixels. Defaults to 8.
        feathering (int, optional): Gaussian blur kernel size for mask feathering. Defaults to 40.
        skip_noise (bool, optional): Skip noise injection (crucial for FLUX Fill). Defaults to False.
        sharp_mirror (bool, optional): Skip blurring the padded regions to provide a sharp base. Defaults to False.

    Returns:
        tuple: (padded_image, padded_mask) with the same tensor formats.
    """
    if pad_l <= 0 and pad_t <= 0 and pad_r <= 0 and pad_b <= 0:
        return image, mask

    B, H, W, C = image.shape

    # 1. Expand the canvas with replicate padding (edge pixel stretching)
    # Replicate padding is far superior to reflect padding for outpainting because
    # it prevents central subjects (e.g. a bright sun) from being mirrored into the sky,
    # which would corrupt the conditioning colors.
    img_p = image.permute(0, 3, 1, 2)
    img_padded = torch.nn.functional.pad(img_p, (pad_l, pad_r, pad_t, pad_b), mode='replicate')
    
    # 2. Aggressively blur the mirrored pixels to destroy symmetry patterns.
    # Old cap of 21px was far too weak for large padding (160-320px).
    # New formula: ~2/3 of max padding, capped at 99px.
    if sharp_mirror:
        blurred_padded = img_padded
    else:
        max_pad = max(pad_l, pad_t, pad_r, pad_b)
        kernel_size = max_pad // 3 * 2 + 1
        kernel_size = max(kernel_size, 3)
        kernel_size = min(kernel_size, 99)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred_padded = TF.gaussian_blur(img_padded, kernel_size=kernel_size)

    # 3. Inject light noise into the padded areas to break residual symmetry.
    # Gives the diffusion model a randomized starting point instead of a
    # recognizable (even if blurred) mirror of the source.
    if not skip_noise:
        noise_strength = 0.12
        if pad_t > 0:
            blurred_padded[:, :, :pad_t, :] += torch.randn_like(blurred_padded[:, :, :pad_t, :]) * noise_strength
        if pad_b > 0:
            blurred_padded[:, :, -(pad_b):, :] += torch.randn_like(blurred_padded[:, :, -(pad_b):, :]) * noise_strength
        if pad_l > 0:
            blurred_padded[:, :, :, :pad_l] += torch.randn_like(blurred_padded[:, :, :, :pad_l]) * noise_strength
        if pad_r > 0:
            blurred_padded[:, :, :, -(pad_r):] += torch.randn_like(blurred_padded[:, :, :, -(pad_r):]) * noise_strength
        blurred_padded = torch.clamp(blurred_padded, 0.0, 1.0)

    # 4. Restore the sharp original image exactly in its designated center spot
    # Instead of a hard stamp, we feather the sharp image over the blurred padding
    # within the overlap zone to prevent a sharp texture seam in the FLUX conditioning.
    blend = torch.ones((1, 1, H, W), device=image.device, dtype=torch.float32)
    if overlap > 0:
        ramp = torch.linspace(0.0, 1.0, overlap, device=image.device)
        if pad_t > 0: blend[:, :, :overlap, :] = torch.min(blend[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
        if pad_b > 0: blend[:, :, -overlap:, :] = torch.min(blend[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
        if pad_l > 0: blend[:, :, :, :overlap] = torch.min(blend[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
        if pad_r > 0: blend[:, :, :, -overlap:] = torch.min(blend[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    if skip_noise:
        # FLUX Fill requires a perfectly sharp transition from the 0.5 gray padding
        # directly into the pristine high-frequency image. Any alpha blending gradient
        # here confuses the flow matching and produces a square seam.
        blurred_padded[:, :, pad_t:pad_t+H, pad_l:pad_l+W] = img_p
    else:
        blurred_padded[:, :, pad_t:pad_t+H, pad_l:pad_l+W] = img_p * blend + blurred_padded[:, :, pad_t:pad_t+H, pad_l:pad_l+W] * (1.0 - blend)
    
    final_image = blurred_padded.permute(0, 2, 3, 1)

    # Build outpaint mask
    new_h = H + pad_t + pad_b
    new_w = W + pad_l + pad_r
    new_mask = torch.zeros((B, new_h, new_w), dtype=torch.float32, device=final_image.device)

    if mask is not None:
        if len(mask.shape) == 2:
            m_in = mask.unsqueeze(0)
        else:
            m_in = mask
        m_padded = torch.nn.functional.pad(m_in, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
        if len(mask.shape) == 2:
            new_mask = m_padded.squeeze(0)
        else:
            new_mask = m_padded

    # Mark padded regions (with overlap into original area)
    if pad_t > 0: new_mask[:, :pad_t + overlap, :] = 1.0
    if pad_b > 0: new_mask[:, -(pad_b + overlap):, :] = 1.0
    if pad_l > 0: new_mask[:, :, :pad_l + overlap] = 1.0
    if pad_r > 0: new_mask[:, :, -(pad_r + overlap):] = 1.0

    # Feathering (Gaussian blur)
    if feathering > 0:
        k = feathering
        if k % 2 == 0: k += 1
        sig = float(k) / 3.0
        if len(new_mask.shape) == 2:
            m_b = new_mask.unsqueeze(0).unsqueeze(0)
        else:
            m_b = new_mask.unsqueeze(1)
        m_b = TF.gaussian_blur(m_b, kernel_size=k, sigma=sig)
        if len(new_mask.shape) == 2:
            new_mask = m_b.squeeze(0).squeeze(0)
        else:
            new_mask = m_b.squeeze(1)

    return final_image, new_mask

