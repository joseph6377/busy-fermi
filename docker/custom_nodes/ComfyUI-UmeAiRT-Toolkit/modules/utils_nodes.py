import torch
import os
import json
import folder_paths
import nodes as comfy_nodes
import comfy.samplers
from .common import UmeBundle, log_node
from .logger import logger



class UmeAiRT_Bundle_Downloader:
    """Standalone model downloader utility node.

    Downloads model bundles from the remote model manifest to the correct ComfyUI
    model folders WITHOUT loading them into memory. Ideal for:
    - Pre-downloading models on RunPod/cloud before running workflows
    - Batch-downloading entire model families (FLUX, Z-IMG)
    - Ensuring all required files are present before generation

    Uses aria2c for multi-connection downloads when available, with urllib fallback.
    Supports HuggingFace token for authenticated/faster downloads.
    """

    @classmethod
    def INPUT_TYPES(s):
        from .manifest import get_bundle_dropdowns
        categories, versions_list = get_bundle_dropdowns()
        return {
            "required": {
                "category": (categories, {"tooltip": "Model family to download (e.g. FLUX, Z-IMAGE_TURBO)."}),
                "version": (versions_list, {"tooltip": "Quantization/precision variant (e.g. fp16, GGUF_Q4)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download"
    CATEGORY = "UmeAiRT/Utils"
    OUTPUT_NODE = True
    DESCRIPTION = "Standalone utility to download model bundles from the remote manifest."

    def download(self, category, version):
        """Download all files for the selected bundle without loading into memory."""
        from .manifest import download_bundle_files

        try:
            _, _, downloaded, skipped, errors = download_bundle_files(category, version)
        except ValueError as e:
            return (f"❌ {e}",)

        parts = [f"📥 {category}/{version}:"]
        if downloaded:
            parts.append(f"{downloaded} downloaded")
        if skipped:
            parts.append(f"{skipped} already present")
        if errors:
            parts.append(f"{len(errors)} failed ({', '.join(errors)})")
        status = " | ".join(parts)
        log_node(status, color="GREEN" if not errors else "RED")
        return (status,)


class UmeAiRT_Unpack_Settings:
    """Extracts multiple individual variables from a single UME_SETTINGS dictionary bundle."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"settings": ("UME_SETTINGS",)}}
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "*", "*", "INT")
    RETURN_NAMES = ("width", "height", "steps", "cfg", "sampler", "scheduler", "seed")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Interop"
    DESCRIPTION = "Unpacks a UME_SETTINGS bundle into individual native ComfyUI values."
    def unpack(self, settings):
        """Unpacks the provided settings dataclass."""
        return (
            settings.width, settings.height,
            settings.steps, settings.cfg,
            settings.sampler_name, settings.scheduler,
            settings.seed
        )

class UmeAiRT_Unpack_FilesBundle:
    """Deconstructs a unified UME_FILES bundle into standard ComfyUI data pipes.

    Outputs Model, Clip, VAE, and the readable Model Name separately for native nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "Connect a Model Loader output here to see its individual model components."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_name")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Interop"
    DESCRIPTION = "Unpacks a UME_BUNDLE into Model, Clip, and VAE outputs."

    def unpack(self, model_bundle):
        """Extracts model components from the UmeBundle dataclass."""
        return (
            model_bundle.model,
            model_bundle.clip,
            model_bundle.vae,
            model_bundle.model_name,
        )


class UmeAiRT_Pack_Bundle:
    """Packs native ComfyUI types (MODEL, CLIP, VAE) into a UME_BUNDLE.

    Use this to feed models from any native or community loader into the Block pipeline.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model."}),
                "clip": ("CLIP", {"tooltip": "The CLIP text encoder."}),
                "vae": ("VAE", {"tooltip": "The VAE model."}),
            },
            "optional": {
                "model_name": ("STRING", {"default": "", "tooltip": "Model name stored in the bundle metadata (useful for Image Saver)."}),
                "loader_type": (["auto", "flux", "zimg", "qwen", "hidream", "checkpoint"], {
                    "default": "auto",
                    "tooltip": "Model architecture type. Tells the sampler which pipeline to use (e.g. FLUX guidance, HiDream ModelSampling). 'auto' = standard KSampler with no overrides.",
                }),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "pack"
    CATEGORY = "UmeAiRT/Interop"
    DESCRIPTION = "Packs native Model, Clip, and VAE into a UME_BUNDLE."

    def pack(self, model, clip, vae, model_name="", loader_type="auto"):
        """Packs native ComfyUI models into a UmeBundle."""
        lt = "" if loader_type == "auto" else loader_type
        return (UmeBundle(model=model, clip=clip, vae=vae, model_name=model_name, loader_type=lt),)

class UmeAiRT_Unpack_ImageBundle:
    """Deconstructs a UME_IMAGE bundle into native ComfyUI types."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "Connect an Image process output here to see its individual components."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "mode", "denoise", "auto_resize")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Interop"
    DESCRIPTION = "Unpacks a UME_IMAGE bundle into Image, Mask, and configuration outputs."

    def unpack(self, image_bundle):
        """Extracts all fields from the UmeImage dataclass."""
        return (
            image_bundle.image,
            image_bundle.mask,
            image_bundle.mode,
            float(image_bundle.denoise),
            bool(image_bundle.auto_resize),
        )


class UmeAiRT_Unpack_Pipeline:
    """Deconstructs a UME_PIPELINE (GenerationContext) into native ComfyUI types.

    This enables full interoperability: connect any output to native or community nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("UME_PIPELINE", {"tooltip": "Connect a generation pipeline to extract all its values as individual outputs."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "*", "*", "INT", "FLOAT")
    RETURN_NAMES = ("image", "model", "clip", "vae", "model_name", "positive", "negative", "width", "height", "steps", "cfg", "sampler_name", "scheduler", "seed", "denoise")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Interop"
    DESCRIPTION = "Unpacks a UME_PIPELINE into all its native components."

    def unpack(self, pipeline):
        """Extracts all fields from the GenerationContext pipeline.

        Args:
            pipeline (GenerationContext): The pipeline object.

        Returns:
            tuple: All 15 native ComfyUI outputs.
        """
        return (
            pipeline.image,
            pipeline.model,
            pipeline.clip,
            pipeline.vae,
            str(pipeline.model_name or ""),
            str(pipeline.positive_prompt or ""),
            str(pipeline.negative_prompt or ""),
            int(pipeline.width or 1024),
            int(pipeline.height or 1024),
            int(pipeline.steps or 20),
            float(pipeline.cfg or 8.0),
            str(pipeline.sampler_name or "euler"),
            str(pipeline.scheduler or "normal"),
            int(pipeline.seed or 0),
            float(pipeline.denoise or 1.0),
        )


class UmeAiRT_Unpack_VideoPipeline:
    """Deconstructs a UME_VIDEO_PIPELINE (VideoGenerationContext) into native ComfyUI types.

    This enables full interoperability: connect any output to native or community nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_pipe": ("UME_VIDEO_PIPELINE", {"tooltip": "Connect a video pipeline to extract all its values."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "MODEL", "CLIP", "VAE", "CLIP_VISION", "STRING", "STRING", "STRING", "INT", "INT", "FLOAT", "INT", "INT", "INT", "FLOAT", "*", "*", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "model", "model_low_noise", "clip", "vae", "clip_vision", "model_name", "positive", "negative", "width", "height", "duration", "fps", "frame_count", "steps", "cfg", "sampler_name", "scheduler", "seed", "denoise")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Interop"
    DESCRIPTION = "Unpacks a UME_VIDEO_PIPELINE into all its native components."

    def unpack(self, video_pipe):
        """Extracts all fields from the VideoGenerationContext."""
        return (
            video_pipe.frames,
            video_pipe.model,
            getattr(video_pipe, "model_low_noise", None),
            video_pipe.clip,
            video_pipe.vae,
            video_pipe.clip_vision,
            str(video_pipe.model_name or ""),
            str(video_pipe.positive_prompt or ""),
            str(video_pipe.negative_prompt or ""),
            int(video_pipe.width or 848),
            int(video_pipe.height or 480),
            float(video_pipe.duration or 3.0),
            int(video_pipe.fps or 16),
            int(video_pipe.frame_count or 49),
            int(video_pipe.steps or 20),
            float(video_pipe.cfg or 6.0),
            str(video_pipe.sampler_name or "uni_pc"),
            str(video_pipe.scheduler or "simple"),
            int(video_pipe.seed or 0),
            float(video_pipe.denoise or 1.0),
        )


class UmeAiRT_Pack_VideoPipeline:
    """Packs native ComfyUI types into a UME_VIDEO_PIPELINE.

    Use this to construct a video pipeline manually for external interoperability.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "The batch of image tensors representing the video frames."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "The loaded Video Diffusion Model."}),
                "model_low_noise": ("MODEL", {"tooltip": "WAN 2.2 MoE: Low-Noise expert diffusion model."}),
                "clip": ("CLIP", {"tooltip": "The Text Encoder(s) used for prompt embedding."}),
                "vae": ("VAE", {"tooltip": "The Variational Auto-Encoder used for pixel/latent conversion."}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "The Image Encoder used for Image-to-Video generation."}),
                "model_name": ("STRING", {"default": "", "tooltip": "The architectural identifier of the model (e.g. 'WAN_2.1')."}),
                "positive": ("STRING", {"default": "", "multiline": True, "tooltip": "The main text prompt describing the video."}),
                "negative": ("STRING", {"default": "", "multiline": True, "tooltip": "The negative text prompt (what to avoid)."}),
                "width": ("INT", {"default": 848, "tooltip": "The horizontal resolution of the video."}),
                "height": ("INT", {"default": 480, "tooltip": "The vertical resolution of the video."}),
                "duration": ("FLOAT", {"default": 3.0, "tooltip": "The duration of the video in seconds."}),
                "fps": ("INT", {"default": 16, "tooltip": "The framerate of the video."}),
                "steps": ("INT", {"default": 20, "tooltip": "The number of denoising steps."}),
                "cfg": ("FLOAT", {"default": 6.0, "tooltip": "The Classifier-Free Guidance scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "uni_pc", "tooltip": "The sampling algorithm to use."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "The noise schedule to apply."}),
                "seed": ("INT", {"default": 0, "tooltip": "The random seed for generation."}),
                "audio": ("AUDIO", {"tooltip": "Optional audio track associated with the video."}),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "pack"
    CATEGORY = "UmeAiRT/Interop"
    DESCRIPTION = "Packs native components into a UME_VIDEO_PIPELINE."

    def pack(self, frames, model=None, model_low_noise=None, clip=None, vae=None, clip_vision=None, model_name="", positive="", negative="", width=848, height=480, duration=3.0, fps=16, steps=20, cfg=6.0, sampler_name="uni_pc", scheduler="simple", seed=0, audio=None):
        from .common import VideoGenerationContext
        ctx = VideoGenerationContext()
        ctx.frames = frames
        ctx.model = model
        ctx.model_low_noise = model_low_noise
        ctx.clip = clip
        ctx.vae = vae
        ctx.clip_vision = clip_vision
        ctx.model_name = model_name
        ctx.positive_prompt = positive
        ctx.negative_prompt = negative
        ctx.width = width
        ctx.height = height
        ctx.duration = duration
        ctx.fps = fps
        ctx.frame_count = frames.shape[0] if frames is not None else int(duration * fps) + 1
        ctx.steps = steps
        ctx.cfg = cfg
        ctx.sampler_name = sampler_name
        ctx.scheduler = scheduler
        ctx.seed = seed
        ctx.audio = audio
        return (ctx,)


class UmeAiRT_VideoToImagePipeline:
    """Converts a UME_VIDEO_PIPELINE into a UME_PIPELINE.

    This allows sending video frames (or a single generated frame for Text-to-Image)
    directly to the standard Image Saver nodes (like the one used for FLUX) which
    expect a generation pipeline with metadata.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_pipe": ("UME_VIDEO_PIPELINE", {"tooltip": "The video pipeline to convert."}),
            }
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("image_pipe",)
    FUNCTION = "convert"
    CATEGORY = "UmeAiRT/Interop"
    DESCRIPTION = "Converts a Video Pipeline to an Image Pipeline for compatibility with Image Savers."

    def convert(self, video_pipe):
        from .common import GenerationContext
        ctx = GenerationContext()
        ctx.image = video_pipe.frames
        ctx.model = video_pipe.model
        ctx.clip = video_pipe.clip
        ctx.vae = video_pipe.vae
        ctx.model_name = getattr(video_pipe, "model_name", "UmeAiRT_Video")
        ctx.positive_prompt = getattr(video_pipe, "positive_prompt", "")
        ctx.negative_prompt = getattr(video_pipe, "negative_prompt", "")
        ctx.width = getattr(video_pipe, "width", 848)
        ctx.height = getattr(video_pipe, "height", 480)
        ctx.steps = getattr(video_pipe, "steps", 20)
        ctx.cfg = getattr(video_pipe, "cfg", 6.0)
        ctx.sampler_name = getattr(video_pipe, "sampler_name", "uni_pc")
        ctx.scheduler = getattr(video_pipe, "scheduler", "simple")
        ctx.seed = getattr(video_pipe, "seed", 0)
        ctx.denoise = getattr(video_pipe, "denoise", 1.0)
        ctx.loras = getattr(video_pipe, "loras", [])
        return (ctx,)


# --- Legacy Unpack Nodes Restoration ---


class UmeAiRT_Signature:
    """A Node designed purely for aesthetic and branding purposes on the canvas.

    It renders a custom transparent PNG signature (`assets/signature.png`) via JavaScript.
    It has no inputs, and running the node yields an empty result.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}, # No inputs! Clean and simple.
        }

    RETURN_TYPES = ()
    FUNCTION = "display_signature"
    CATEGORY = "UmeAiRT/Utils"
    OUTPUT_NODE = True 
    DESCRIPTION = "Displays the UmeAiRT signature logo on the canvas."

    def display_signature(self):
        """Silently short-circuits execution context.

        Returns:
            dict: An empty UI images dictionary since rendering is entirely handled client-side.
        """
        # The node execution does nothing except return the path relative to ComfyUI for preview.
        # But this node is for frontend visual mostly!
        # If the user somehow executes it, we just return empty.
        # The real magic happens in umeairt_signature.js
        return {"ui": {"images": []}}

# Alias for legacy compatibility
UmeAiRT_Unpack_SettingsBundle = UmeAiRT_Unpack_Settings


