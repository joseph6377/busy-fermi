# --- Dynamic Sampler Registration ---
try:
    from .modules.extra_samplers import register_extra_samplers
    register_extra_samplers()
except Exception as e:
    print(f"[UmeAiRT-Toolkit] Failed to register extra samplers: {e}")
# ------------------------------------
from .modules.video_postprod import (
    UmeAiRT_VideoFrameInterpolation,
    UmeAiRT_VideoSmartUpscale
)
from .modules.ltx_enhancer import UmeAiRT_LTXVideoEnhancer
from .modules.video_slicer import UmeAiRT_VideoSlicer, UmeAiRT_VideoConcat
from .modules.ltx_audio_replacer import UmeAiRT_LTXAudioReplacer
from .modules.ltx_keyframe_generator import UmeAiRT_LTXKeyframeGenerator
from .modules.ltx_prompt_director import UmeAiRT_PromptSegment, UmeAiRT_LTXPromptDirector
from .modules.image_nodes import (
    UmeAiRT_PipelineImageSaver,
)
# Register 'bbox' folder for FaceDetailer
import folder_paths
import os
try:
    folder_paths.add_model_folder_path("bbox", os.path.join(folder_paths.models_dir, "bbox"))
except Exception as e:
    print(f"[UmeAiRT-Toolkit] bbox folder registration note: {e}")
if "bbox" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["bbox"] = ([os.path.join(folder_paths.models_dir, "bbox")], folder_paths.supported_pt_extensions)

from .modules.logic_nodes import (
    UmeAiRT_PipelineUltimateUpscale,
    UmeAiRT_PipelineSeedVR2Upscale,
    UmeAiRT_PipelineSubjectDetailer,
    UmeAiRT_Detailer_Daemon,
    UmeAiRT_DetailRefiner
)
from .modules.block_nodes import (
    UmeAiRT_LoraBlock_1, UmeAiRT_LoraBlock_3, UmeAiRT_LoraBlock_5, UmeAiRT_LoraBlock_10,
    UmeAiRT_WanLoraBlock_1, UmeAiRT_WanLoraBlock_3, UmeAiRT_WanLoraBlock_5, UmeAiRT_WanLoraBlock_10,
    UmeAiRT_ControlNetImageApply,
    UmeAiRT_GenerationSettings,
    UmeAiRT_VideoSettings,
    UmeAiRT_LTXVideoSettings,
    UmeAiRT_FilesSettings_Checkpoint,
    UmeAiRT_FilesSettings_FLUX,
    UmeAiRT_FilesSettings_ZIMG,
    UmeAiRT_FilesSettings_QWEN,
    UmeAiRT_FilesSettings_ANIMA,
    UmeAiRT_FilesSettings_HiDream,
    UmeAiRT_FilesSettings_WAN,
    UmeAiRT_FilesSettings_LTX,
    UmeAiRT_BlockImageLoader, UmeAiRT_BlockImageProcess,
    UmeAiRT_BlockVideoLoader,
    UmeAiRT_ImageProcess_Img2Img, UmeAiRT_ImageProcess_Inpaint, UmeAiRT_ImageProcess_Outpaint, UmeAiRT_ImageProcess_Kontext, UmeAiRT_ImageProcess_Edit,
    UmeAiRT_BlockSampler,
    UmeAiRT_PackPipeline,
    UmeAiRT_BundleLoader,
    UmeAiRT_LightningAccelerator,
    UmeAiRT_VideoGenerator,
    UmeAiRT_LTXVideoGenerator,
    UmeAiRT_VideoLightningAccelerator,
    UmeAiRT_VideoOptimization,
    UmeAiRT_VideoOutput,
    UmeAiRT_VideoVacePrep,
    UmeAiRT_VideoExtender,
    UmeAiRT_LTXVideoExtender,
    UmeAiRT_VideoLooper,
    UmeAiRT_VideoControlNetApply,
    UmeAiRT_Positive_Input, UmeAiRT_Negative_Input
)
from .modules.utils_nodes import (
    UmeAiRT_Bundle_Downloader,
    UmeAiRT_Unpack_SettingsBundle,
    UmeAiRT_Unpack_ImageBundle,
    UmeAiRT_Pack_Bundle,
    UmeAiRT_Unpack_Pipeline,
    UmeAiRT_Pack_VideoPipeline,
    UmeAiRT_Unpack_VideoPipeline,
    UmeAiRT_Unpack_FilesBundle,
    UmeAiRT_VideoToImagePipeline,
    UmeAiRT_Signature,
)
from .modules.image_analyze import UmeAiRT_ImageToPrompt

NODE_CLASS_MAPPINGS = {
    # Block Loaders
    "UmeAiRT_FilesSettings_Checkpoint": UmeAiRT_FilesSettings_Checkpoint,
    "UmeAiRT_FilesSettings_FLUX": UmeAiRT_FilesSettings_FLUX,
    "UmeAiRT_FilesSettings_ZIMG": UmeAiRT_FilesSettings_ZIMG,
    "UmeAiRT_FilesSettings_QWEN": UmeAiRT_FilesSettings_QWEN,
    "UmeAiRT_FilesSettings_ANIMA": UmeAiRT_FilesSettings_ANIMA,
    "UmeAiRT_FilesSettings_HiDream": UmeAiRT_FilesSettings_HiDream,
    "UmeAiRT_BundleLoader": UmeAiRT_BundleLoader,
    "UmeAiRT_LightningAccelerator": UmeAiRT_LightningAccelerator,
    "UmeAiRT_FilesSettings_WAN": UmeAiRT_FilesSettings_WAN,
    "UmeAiRT_FilesSettings_LTX": UmeAiRT_FilesSettings_LTX,

    # Block Settings & Image
    "UmeAiRT_GenerationSettings": UmeAiRT_GenerationSettings,
    "UmeAiRT_VideoSettings": UmeAiRT_VideoSettings,
    "UmeAiRT_LTXVideoSettings": UmeAiRT_LTXVideoSettings,
    "UmeAiRT_BlockImageLoader": UmeAiRT_BlockImageLoader,
    "UmeAiRT_BlockVideoLoader": UmeAiRT_BlockVideoLoader,
    "UmeAiRT_BlockImageProcess": UmeAiRT_BlockImageProcess,
    "UmeAiRT_ImageProcess_Img2Img": UmeAiRT_ImageProcess_Img2Img,
    "UmeAiRT_ImageProcess_Inpaint": UmeAiRT_ImageProcess_Inpaint,
    "UmeAiRT_ImageProcess_Outpaint": UmeAiRT_ImageProcess_Outpaint,
    "UmeAiRT_ImageProcess_Kontext": UmeAiRT_ImageProcess_Kontext,
    "UmeAiRT_ImageProcess_Edit": UmeAiRT_ImageProcess_Edit,
    "UmeAiRT_LoraBlock_1": UmeAiRT_LoraBlock_1,
    "UmeAiRT_LoraBlock_3": UmeAiRT_LoraBlock_3,
    "UmeAiRT_LoraBlock_5": UmeAiRT_LoraBlock_5,
    "UmeAiRT_LoraBlock_10": UmeAiRT_LoraBlock_10,
    "UmeAiRT_WanLoraBlock_1": UmeAiRT_WanLoraBlock_1,
    "UmeAiRT_WanLoraBlock_3": UmeAiRT_WanLoraBlock_3,
    "UmeAiRT_WanLoraBlock_5": UmeAiRT_WanLoraBlock_5,
    "UmeAiRT_WanLoraBlock_10": UmeAiRT_WanLoraBlock_10,
    "UmeAiRT_ControlNetImageApply": UmeAiRT_ControlNetImageApply,

    # Prompt Editors
    "UmeAiRT_Positive_Input": UmeAiRT_Positive_Input,
    "UmeAiRT_Negative_Input": UmeAiRT_Negative_Input,

    # Sampler & Post-Process (Pipeline)
    "UmeAiRT_BlockSampler": UmeAiRT_BlockSampler,
    "UmeAiRT_PackPipeline": UmeAiRT_PackPipeline,
    "UmeAiRT_PipelineUltimateUpscale": UmeAiRT_PipelineUltimateUpscale,
    "UmeAiRT_PipelineSeedVR2Upscale": UmeAiRT_PipelineSeedVR2Upscale,
    "UmeAiRT_PipelineSubjectDetailer": UmeAiRT_PipelineSubjectDetailer,
    "UmeAiRT_Detailer_Daemon": UmeAiRT_Detailer_Daemon,
    "UmeAiRT_DetailRefiner": UmeAiRT_DetailRefiner,

    # Image
    "UmeAiRT_PipelineImageSaver": UmeAiRT_PipelineImageSaver,

    # Video
    "UmeAiRT_VideoGenerator": UmeAiRT_VideoGenerator,
    "UmeAiRT_LTXVideoGenerator": UmeAiRT_LTXVideoGenerator,
    "UmeAiRT_LTXVideoExtender": UmeAiRT_LTXVideoExtender,
    "UmeAiRT_LTXVideoEnhancer": UmeAiRT_LTXVideoEnhancer,
    "UmeAiRT_LTXKeyframeGenerator": UmeAiRT_LTXKeyframeGenerator,
    "UmeAiRT_PromptSegment": UmeAiRT_PromptSegment,
    "UmeAiRT_LTXPromptDirector": UmeAiRT_LTXPromptDirector,
    "UmeAiRT_LTXAudioReplacer": UmeAiRT_LTXAudioReplacer,
    "UmeAiRT_VideoSlicer": UmeAiRT_VideoSlicer,
    "UmeAiRT_VideoConcat": UmeAiRT_VideoConcat,
    "UmeAiRT_VideoLightningAccelerator": UmeAiRT_VideoLightningAccelerator,
    "UmeAiRT_VideoOptimization": UmeAiRT_VideoOptimization,
    "UmeAiRT_VideoFrameInterpolation": UmeAiRT_VideoFrameInterpolation,
    "UmeAiRT_VideoSmartUpscale": UmeAiRT_VideoSmartUpscale,
    "UmeAiRT_VideoOutput": UmeAiRT_VideoOutput,
    "UmeAiRT_VideoVacePrep": UmeAiRT_VideoVacePrep,
    "UmeAiRT_VideoExtender": UmeAiRT_VideoExtender,
    "UmeAiRT_VideoLooper": UmeAiRT_VideoLooper,
    "UmeAiRT_VideoControlNetApply": UmeAiRT_VideoControlNetApply,

    # Pack/Unpack (Interoperability)
    "UmeAiRT_Pack_Bundle": UmeAiRT_Pack_Bundle,
    "UmeAiRT_Unpack_Pipeline": UmeAiRT_Unpack_Pipeline,
    "UmeAiRT_Pack_VideoPipeline": UmeAiRT_Pack_VideoPipeline,
    "UmeAiRT_Unpack_VideoPipeline": UmeAiRT_Unpack_VideoPipeline,
    "UmeAiRT_VideoToImagePipeline": UmeAiRT_VideoToImagePipeline,
    "UmeAiRT_Unpack_FilesBundle": UmeAiRT_Unpack_FilesBundle,
    "UmeAiRT_Unpack_ImageBundle": UmeAiRT_Unpack_ImageBundle,
    "UmeAiRT_Unpack_SettingsBundle": UmeAiRT_Unpack_SettingsBundle,

    # Utils
    "UmeAiRT_Signature": UmeAiRT_Signature,
    "UmeAiRT_Bundle_Downloader": UmeAiRT_Bundle_Downloader,

    # Tools
    "UmeAiRT_ImageToPrompt": UmeAiRT_ImageToPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Loaders
    "UmeAiRT_FilesSettings_Checkpoint": "⬡ Checkpoint Loader",
    "UmeAiRT_FilesSettings_FLUX": "⬡ FLUX Loader",
    "UmeAiRT_FilesSettings_ZIMG": "⬡ Z-IMG Loader",
    "UmeAiRT_FilesSettings_QWEN": "⬡ QWEN Loader",
    "UmeAiRT_FilesSettings_ANIMA": "⬡ ANIMA Loader",
    "UmeAiRT_FilesSettings_HiDream": "⬡ HiDream Loader",
    "UmeAiRT_BundleLoader": "⬡ 📦 Bundle Auto-Loader",
    "UmeAiRT_LightningAccelerator": "⬡ ⚡ Lightning Accelerator",
    "UmeAiRT_FilesSettings_WAN": "⬡ WAN Loader",
    "UmeAiRT_FilesSettings_LTX": "⬡ LTX Loader",

    # Settings & Image
    "UmeAiRT_GenerationSettings": "⬡ Generation Settings",
    "UmeAiRT_VideoSettings": "⬡ Video Settings",
    "UmeAiRT_LTXVideoSettings": "⬡ LTX Video Settings",
    "UmeAiRT_BlockImageLoader": "⬡ Image Loader",
    "UmeAiRT_BlockVideoLoader": "⬡ Video Loader",
    "UmeAiRT_BlockImageProcess": "⬡ Image Process",
    "UmeAiRT_ImageProcess_Img2Img": "⬡ Image Process (Img2Img)",
    "UmeAiRT_ImageProcess_Inpaint": "⬡ Image Process (Inpaint)",
    "UmeAiRT_ImageProcess_Outpaint": "⬡ Image Process (Outpaint)",
    "UmeAiRT_ImageProcess_Kontext": "⬡ Image Process (Kontext)",
    "UmeAiRT_ImageProcess_Edit": "⬡ Image Process (Edit)",
    "UmeAiRT_LoraBlock_1": "⬡ LoRA 1x",
    "UmeAiRT_LoraBlock_3": "⬡ LoRA 3x",
    "UmeAiRT_LoraBlock_5": "⬡ LoRA 5x",
    "UmeAiRT_LoraBlock_10": "⬡ LoRA 10x",
    "UmeAiRT_WanLoraBlock_1": "⬡ WAN LoRA 1x",
    "UmeAiRT_WanLoraBlock_3": "⬡ WAN LoRA 3x",
    "UmeAiRT_WanLoraBlock_5": "⬡ WAN LoRA 5x",
    "UmeAiRT_WanLoraBlock_10": "⬡ WAN LoRA 10x",
    "UmeAiRT_ControlNetImageApply": "⬡ ControlNet Apply",

    # Prompt Editors
    "UmeAiRT_Positive_Input": "⬡ Positive Prompt Input",
    "UmeAiRT_Negative_Input": "⬡ Negative Prompt Input",

    # Sampler & Post-Process
    "UmeAiRT_BlockSampler": "⬡ Image Generator",
    "UmeAiRT_PackPipeline": "⬡ Pack Pipeline",
    "UmeAiRT_PipelineUltimateUpscale": "⬡ UltimateSD Upscale",
    "UmeAiRT_PipelineSeedVR2Upscale": "⬡ SeedVR2 Upscale",
    "UmeAiRT_PipelineSubjectDetailer": "⬡ Subject Detailer",
    "UmeAiRT_Detailer_Daemon": "⬡ Detailer Daemon",
    "UmeAiRT_DetailRefiner": "⬡ Detail Refiner",

    # Image
    "UmeAiRT_PipelineImageSaver": "⬡ Image Saver",

    # Video
    "UmeAiRT_VideoGenerator": "⬡ Video Generator",
    "UmeAiRT_LTXVideoGenerator": "⬡ LTX Video Generator",
    "UmeAiRT_LTXVideoExtender": "⬡ LTX Video Extender",
    "UmeAiRT_LTXVideoEnhancer": "⬡ LTX Video Enhancer",
    "UmeAiRT_LTXKeyframeGenerator": "⬡ LTX Keyframe Generator",
    "UmeAiRT_PromptSegment": "⬡ Prompt Segment",
    "UmeAiRT_LTXPromptDirector": "⬡ LTX Prompt Director",
    "UmeAiRT_LTXAudioReplacer": "⬡ LTX Audio Replacer",
    "UmeAiRT_VideoSlicer": "⬡ Video Slicer",
    "UmeAiRT_VideoConcat": "⬡ Video Concatenate",
    "UmeAiRT_VideoLightningAccelerator": "⬡ ⚡ Video Lightning",
    "UmeAiRT_VideoOptimization": "⬡ Video Optimization",
    "UmeAiRT_VideoFrameInterpolation": "⬡ Video Interpolation",
    "UmeAiRT_VideoSmartUpscale": "⬡ Smart Video Upscale",
    "UmeAiRT_VideoOutput": "⬡ Video Output",
    "UmeAiRT_VideoVacePrep": "⬡ Video VACE Prep",
    "UmeAiRT_VideoExtender": "⬡ Video Extender",
    "UmeAiRT_VideoLooper": "⬡ Video Looper",
    "UmeAiRT_VideoControlNetApply": "⬡ Video ControlNet Apply",

    # Pack/Unpack
    "UmeAiRT_Pack_Bundle": "⬡ Pack Models Bundle",
    "UmeAiRT_Unpack_Pipeline": "⬡ Unpack Pipeline",
    "UmeAiRT_Pack_VideoPipeline": "⬡ Pack Video Pipeline",
    "UmeAiRT_Unpack_VideoPipeline": "⬡ Unpack Video Pipeline",
    "UmeAiRT_VideoToImagePipeline": "⬡ Video Pipe to Image Pipe",
    "UmeAiRT_Unpack_FilesBundle": "⬡ Unpack Models Bundle",
    "UmeAiRT_Unpack_ImageBundle": "⬡ Unpack Image Bundle",
    "UmeAiRT_Unpack_SettingsBundle": "⬡ Unpack Settings Bundle",

    # Utils
    "UmeAiRT_Signature": "⬡ UmeAiRT Signature",
    "UmeAiRT_Bundle_Downloader": "⬡ 💾 Bundle Model Downloader",

    # Tools
    "UmeAiRT_ImageToPrompt": "⬡ 🔍 Image to Prompt",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Startup Logging & Optimization Check
from .modules.common import log_node
from .modules.optimization_utils import check_optimizations

# Register API routes (only when running inside ComfyUI server)
try:
    import server
    import folder_paths
    from aiohttp import web

    @server.PromptServer.instance.routes.get("/umeairt/signature")
    async def get_signature(request):
        signature_path = os.path.join(os.path.dirname(__file__), "assets", "signature.png")
        if os.path.exists(signature_path):
            return web.FileResponse(signature_path)
        return web.Response(status=404, text="Signature not found")

    @server.PromptServer.instance.routes.get("/umeairt/lora-info")
    async def get_lora_info(request):
        """Returns extracted metadata for a LoRA safetensors file (trigger words, base model, etc.)."""
        import struct
        import json as _json

        filename = request.rel_url.query.get("filename", "")
        if not filename:
            return web.json_response({"error": "Missing 'filename' query parameter."}, status=400)

        full_path = folder_paths.get_full_path("loras", filename)
        if not full_path or not os.path.exists(full_path):
            return web.json_response({"error": f"LoRA file not found: {filename}"}, status=404)

        # Read only the safetensors header — no GPU memory, very fast
        meta = {}
        try:
            with open(full_path, "rb") as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                # Safety cap: headers should never be larger than 100MB
                if header_size > 100 * 1024 * 1024:
                    return web.json_response({"error": "Header too large, likely not a safetensors file."}, status=400)
                header = _json.loads(f.read(header_size))
            meta = header.get("__metadata__", {})
        except Exception as e:
            return web.json_response({"error": f"Failed to read safetensors header: {e}"}, status=500)

        # Extract trigger words from ss_tag_frequency
        trigger_words = []
        tag_freq_raw = meta.get("ss_tag_frequency", "")
        if tag_freq_raw:
            try:
                freq_dict = _json.loads(tag_freq_raw) if isinstance(tag_freq_raw, str) else tag_freq_raw
                for dataset_tags in freq_dict.values():
                    if isinstance(dataset_tags, dict):
                        trigger_words.extend(
                            sorted(dataset_tags.keys(), key=lambda k: dataset_tags[k], reverse=True)
                        )
            except Exception:
                pass

        file_size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 1)

        return web.json_response({
            "filename": filename,
            "base_model": meta.get("ss_base_model_version", "unknown"),
            "trigger_words": trigger_words[:20],
            "network_dim": meta.get("ss_network_dim", ""),
            "network_alpha": meta.get("ss_network_alpha", ""),
            "training_comment": meta.get("ss_training_comment", ""),
            "resolution": meta.get("ss_resolution", ""),
            "file_size_mb": file_size_mb,
        })

    # --- Hardware Monitor API routes & service ---
    try:
        from .modules.monitor_hardware import get_monitor_service, get_gpu_info

        _monitor_service = get_monitor_service(rate=1.0)

        @server.PromptServer.instance.routes.get("/umeairt/monitor/gpu-info")
        async def monitor_gpu_info(request):
            """Return list of detected GPUs (name, index, type)."""
            return web.json_response(get_gpu_info())

        @server.PromptServer.instance.routes.patch("/umeairt/monitor/settings")
        async def monitor_settings(request):
            """Update monitor settings (rate, enabled)."""
            try:
                settings = await request.json()
                if "rate" in settings:
                    _monitor_service.set_rate(float(settings["rate"]))
                if "enabled" in settings:
                    if settings["enabled"]:
                        _monitor_service.start()
                    else:
                        _monitor_service.stop()
                return web.Response(status=200)
            except Exception as e:
                return web.Response(status=400, text=str(e))

        # Start the monitor service
        _monitor_service.start()

    except Exception as e:
        log_node(f"⚠️ Hardware monitor init: {e}", color="YELLOW")

except Exception:
    pass  # Not running inside ComfyUI server (tests, linting, etc.)

# 1. Print Node List
n_nodes = len(NODE_CLASS_MAPPINGS)
log_node(f"🧩 Loaded {n_nodes} nodes.", color="RESET")

# 2. Run Optimization Checks
try:
    check_optimizations()
except Exception as e:
    log_node(f"Optimization check failed: {e}", color="RED")

# 3. Final Summary
log_node("✅ Initialization Complete.", color="GREEN")
