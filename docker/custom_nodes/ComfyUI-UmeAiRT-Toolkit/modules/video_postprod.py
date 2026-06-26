"""
UmeAiRT Toolkit - Video Post-Production
---------------------------------------
Nodes for processing and refining generated videos, such as frame interpolation.
"""

import folder_paths
from .common import VideoGenerationContext, log_node, KNOWN_DIT_MODELS
from .manifest import load_manifest, download_bundle_files

def _get_upscale_models():
    """Combine local upscale models with auto-downloadable ones from manifest."""
    local_models = folder_paths.get_filename_list("upscale_models")
    manifest_models = []
    
    try:
        data = load_manifest()
        upscale_data = data.get("_UPSCALE_MODELS", {})
        for model_name in upscale_data.keys():
            if model_name not in local_models:
                manifest_models.append(model_name)
    except Exception as e:
        log_node(f"Video Smart Upscale: Could not load manifest for remote models: {e}", color="YELLOW")
        
    return sorted(list(set(local_models + manifest_models)))

def _get_rife_models():
    """Combine local RIFE models with auto-downloadable ones from manifest."""
    local_models = folder_paths.get_filename_list("frame_interpolation")
    manifest_models = []
    
    try:
        data = load_manifest()
        rife_data = data.get("_FRAME_INTERPOLATION_MODELS", {})
        for model_name in rife_data.keys():
            if model_name not in local_models:
                manifest_models.append(model_name)
    except Exception as e:
        log_node(f"Video Interpolation: Could not load manifest for remote models: {e}", color="YELLOW")
        
    return sorted(list(set(local_models + manifest_models)))

class UmeAiRT_VideoFrameInterpolation:
    """Frame interpolation node using RIFE.
    
    Doubles the FPS of a generated video by synthesizing intermediate frames.
    Requires RIFE models installed in ComfyUI/models/frame_interpolation/
    """

    @classmethod
    def INPUT_TYPES(s):
        fi_models = _get_rife_models()
        if not fi_models:
            fi_models = ["None"]

        return {
            "required": {
                "video_pipe": ("UME_VIDEO_PIPELINE", {"tooltip": "Video pipeline to interpolate."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough", "tooltip": "Turn interpolation on or off. When off, the video passes through unchanged."}),
                "model": (fi_models, {
                    "tooltip": "RIFE frame interpolation model. Doubles the FPS for smoother playback."
                }),
                "multiplier": ("INT", {
                    "default": 2, "min": 2, "max": 4, "step": 1,
                    "display": "slider",
                    "tooltip": "Multiplier for frame count (e.g. 2x doubles the FPS)."
                }),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Post-Process"
    DESCRIPTION = "Increases video framerate using frame interpolation techniques (e.g. RIFE)."

    def process(self, video_pipe: VideoGenerationContext, enabled=True, model="None", multiplier=2):
        if not enabled or model == "None" or not model:
            return (video_pipe,)

        frames = video_pipe.frames
        if frames is None or frames.shape[0] == 0:
            log_node("Frame Interpolation: No frames to interpolate.", color="YELLOW")
            return (video_pipe,)

        actual_model = model
        local_models = folder_paths.get_filename_list("frame_interpolation")
        
        # Auto-download remote manifest models if not present locally
        if actual_model not in local_models:
            log_node(f"Video Interpolation: '{actual_model}' not found locally. Auto-downloading...", color="CYAN")
            try:
                resolved_files, meta, dn, sk, err = download_bundle_files("_FRAME_INTERPOLATION_MODELS", actual_model)
                if err:
                     raise RuntimeError(f"Video Interpolation: Failed to auto-download {actual_model}: {', '.join(err)}")
            except Exception as e:
                log_node(f"Video Interpolation: Remote manifest resolution error for '{actual_model}': {e}", color="RED")
                raise RuntimeError(f"Video Interpolation: failed to retrieve '{actual_model}': {e}")

        try:
            from comfy_extras.nodes_frame_interpolation import FrameInterpolationModelLoader, FrameInterpolate
            
            # Load the interpolation model
            fi_model_res = FrameInterpolationModelLoader.execute(actual_model)
            # Handle tuple or comfy_api NodeOutput
            if isinstance(fi_model_res, tuple):
                fi_model = fi_model_res[0]
            elif hasattr(fi_model_res, 'args') and fi_model_res.args:
                fi_model = fi_model_res.args[0]
            else:
                fi_model = getattr(fi_model_res, 'value', getattr(fi_model_res, 'outputs', [fi_model_res])[0])
            
            if fi_model:
                # Force VRAM cleanup for lower-end GPUs (e.g., 6GB/8GB)
                import comfy.model_management as mm
                mm.soft_empty_cache()
                
                # Apply interpolation
                frames_res = FrameInterpolate.execute(fi_model, frames, multiplier)
                if isinstance(frames_res, tuple):
                    frames = frames_res[0]
                elif hasattr(frames_res, 'args') and frames_res.args:
                    frames = frames_res.args[0]
                else:
                    frames = getattr(frames_res, 'value', getattr(frames_res, 'outputs', [frames_res])[0])
                
                # Update pipeline context
                video_pipe.frames = frames
                video_pipe.fps = video_pipe.fps * multiplier
                video_pipe.frame_count = frames.shape[0]
                
                log_node(f"✨ Frame Interpolation: RIFE {actual_model} → {frames.shape[0]} frames @ {video_pipe.fps}fps", color="GREEN")
            else:
                log_node(f"Frame Interpolation: Failed to load model '{actual_model}'.", color="YELLOW")
        except Exception as e:
            log_node(f"Frame Interpolation: Failed ({e}), skipping.", color="YELLOW")

        return (video_pipe,)

class UmeAiRT_VideoSmartUpscale:
    """Intelligent video upscaler that selects between Classic and SeedVR2 based on VRAM."""

    @classmethod
    def INPUT_TYPES(s):
        default_dit = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
        try:
             from ..seedvr2_core.seedvr2_adapter import _ensure_seedvr2_path
             _ensure_seedvr2_path()
             from seedvr2_videoupscaler.src.utils.constants import get_all_model_files
             on_disk = list(get_all_model_files().keys())
             extra = [f for f in on_disk if f not in KNOWN_DIT_MODELS and f != "ema_vae_fp16.safetensors"]
             dit_models = KNOWN_DIT_MODELS + sorted(extra)
        except Exception as e:
             log_node(f"Video Smart Upscale: Could not map extended DiT models: {e}", color="YELLOW")
             dit_models = KNOWN_DIT_MODELS

        upscale_models = _get_upscale_models()
        if not upscale_models:
            upscale_models = ["None"]

        return {
            "required": {
                "video_pipe": ("UME_VIDEO_PIPELINE", {"tooltip": "Video pipeline to upscale."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough", "tooltip": "Turn upscale on or off. When off, the video passes through unchanged."}),
                "engine": (["Auto", "Classic (ESRGAN/NMKD)", "SeedVR2 (DiT)"], {"default": "Auto", "tooltip": "Auto: Chooses SeedVR2 if VRAM >= 24GB, else Classic. Classic: Fast spatial upscale. SeedVR2: High-end temporal upscale."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "display": "slider", "tooltip": "Target resolution multiplier."}),
                "classic_model": (upscale_models, {"tooltip": "Model used for Classic mode."}),
                "seedvr2_model": (dit_models, {"default": default_dit, "tooltip": "Model used for SeedVR2 mode."}),
            },
            "optional": {
                "seedvr2_batch_size": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1, "advanced": True, "tooltip": "SeedVR2: Number of frames to process at once."}),
                "seedvr2_temporal_overlap": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1, "advanced": True, "tooltip": "SeedVR2: Overlap between batches to ensure temporal consistency."}),
            }
        }

    RETURN_TYPES = ("UME_VIDEO_PIPELINE",)
    RETURN_NAMES = ("video_pipe",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Post-Process"
    DESCRIPTION = "Upscales video frames using spatial and temporal enhancement techniques."

    def process(self, video_pipe: VideoGenerationContext, enabled=True, engine="Auto", upscale_by=2.0, classic_model="None", seedvr2_model="None", seedvr2_batch_size=8, seedvr2_temporal_overlap=2):
        if not enabled:
            return (video_pipe,)

        frames = video_pipe.frames
        if frames is None or frames.shape[0] == 0:
            log_node("Video Smart Upscale: No frames to upscale.", color="YELLOW")
            return (video_pipe,)

        if upscale_by <= 1.0:
            return (video_pipe,)

        import comfy.model_management as mm
        device = mm.get_torch_device()
        total_vram_gb = mm.get_total_memory(device) / (1024**3)

        use_seedvr2 = False
        if engine == "SeedVR2 (DiT)":
            use_seedvr2 = True
        elif engine == "Auto":
            if total_vram_gb >= 22.0:
                log_node(f"Video Smart Upscale (Auto): VRAM {total_vram_gb:.1f}GB >= 24.0GB -> Routing to SeedVR2.", color="GREEN")
                use_seedvr2 = True
            else:
                log_node(f"Video Smart Upscale (Auto): VRAM {total_vram_gb:.1f}GB < 24.0GB -> Routing to Classic (safely avoiding OOM).", color="ORANGE")
                use_seedvr2 = False

        if use_seedvr2:
            try:
                from ..seedvr2_core.seedvr2_adapter import execute_seedvr2
                
                log_node(f"✨ Video Upscale (SeedVR2): {frames.shape[0]} frames | {seedvr2_model} | Batch {seedvr2_batch_size}", color="GREEN")
                
                # Free VRAM explicitly by unloading other models
                import gc
                required_vram = 12 * 1024**3  # Ask ComfyUI to free 12GB of models if possible
                mm.free_memory(required_vram, device)
                gc.collect()
                mm.soft_empty_cache()

                import torch
                is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                offload = "none" if is_mps else "cpu"

                dit_config = {
                    "model": seedvr2_model, "device": str(device), "offload_device": offload,
                    "cache_model": False, "blocks_to_swap": 0, "swap_io_components": False,
                    "attention_mode": "sdpa", "torch_compile_args": None, "node_id": None,
                }
                vae_config = {
                    "model": "ema_vae_fp16.safetensors", "device": str(device), "offload_device": offload,
                    "cache_model": False, "encode_tiled": True, "encode_tile_size": 512,
                    "encode_tile_overlap": 64, "decode_tiled": True, "decode_tile_size": 512,
                    "decode_tile_overlap": 64, "tile_debug": "false", "torch_compile_args": None, "node_id": None,
                }
                
                # Get resolution of shortest edge
                h, w = frames.shape[1], frames.shape[2]
                current_res = min(h, w)
                target_res = int(current_res * upscale_by)

                # execute_seedvr2 takes the whole video tensor [N, H, W, 3]!
                upscaled_frames = execute_seedvr2(
                    images=frames,
                    dit_config=dit_config,
                    vae_config=vae_config,
                    seed=int(video_pipe.seed or 100) % (2**32),
                    resolution=target_res,
                    batch_size=seedvr2_batch_size,
                    temporal_overlap=seedvr2_temporal_overlap,
                    color_correction="lab"
                )
                
                mm.soft_empty_cache()
                
                video_pipe.frames = upscaled_frames
                video_pipe.height = upscaled_frames.shape[1]
                video_pipe.width = upscaled_frames.shape[2]
            except Exception as e:
                log_node(f"Video Smart Upscale (SeedVR2): Failed ({e}), falling back to Classic...", color="RED")
                import traceback
                traceback.print_exc()
                use_seedvr2 = False # Fall back to classic!

        if not use_seedvr2:
            # Classic Upscale
            if classic_model == "None" or not classic_model:
                log_node("Video Smart Upscale (Classic): No model selected.", color="YELLOW")
                return (video_pipe,)

            actual_model = classic_model
            local_models = folder_paths.get_filename_list("upscale_models")
            
            # Auto-download remote manifest models if not present locally
            if actual_model not in local_models:
                log_node(f"Video Smart Upscale: '{actual_model}' not found locally. Auto-downloading...", color="CYAN")
                try:
                    resolved_files, meta, dn, sk, err = download_bundle_files("_UPSCALE_MODELS", actual_model)
                    if err:
                         raise RuntimeError(f"Video Smart Upscale: Failed to auto-download {actual_model}: {', '.join(err)}")
                except Exception as e:
                    log_node(f"Video Smart Upscale: Remote manifest resolution error for '{actual_model}': {e}", color="RED")
                    raise RuntimeError(f"Video Smart Upscale: failed to retrieve '{actual_model}': {e}")

            try:
                from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
                
                upscale_model_res = UpscaleModelLoader.execute(actual_model)
                if isinstance(upscale_model_res, tuple):
                    u_model = upscale_model_res[0]
                elif hasattr(upscale_model_res, 'args') and upscale_model_res.args:
                    u_model = upscale_model_res.args[0]
                else:
                    u_model = getattr(upscale_model_res, 'value', getattr(upscale_model_res, 'outputs', [upscale_model_res])[0])

                if u_model:
                    log_node(f"✨ Video Upscale (Classic): {frames.shape[0]} frames | {actual_model}", color="GREEN")
                    
                    # Force VRAM cleanup for lower-end GPUs before executing
                    import comfy.model_management as mm
                    mm.soft_empty_cache()
                    
                    frames_res = ImageUpscaleWithModel.execute(u_model, frames)
                    
                    if isinstance(frames_res, tuple):
                        upscaled_frames = frames_res[0]
                    elif hasattr(frames_res, 'args') and frames_res.args:
                        upscaled_frames = frames_res.args[0]
                    else:
                        upscaled_frames = getattr(frames_res, 'value', getattr(frames_res, 'outputs', [frames_res])[0])
                    
                    # Match requested upscale_by ratio if the model's native ratio differs (e.g., 4x model but user wants 2x)
                    orig_h, orig_w = frames.shape[1], frames.shape[2]
                    target_h = int(orig_h * upscale_by)
                    target_w = int(orig_w * upscale_by)
                    
                    up_h, up_w = upscaled_frames.shape[1], upscaled_frames.shape[2]
                    if up_h != target_h or up_w != target_w:
                        log_node(f"Video Smart Upscale (Classic): Resizing from {up_w}x{up_h} to target {target_w}x{target_h} (x{upscale_by})", color="CYAN")
                        import nodes
                        scaler = nodes.ImageScale()
                        scaled_res = scaler.upscale(upscaled_frames, "bicubic", target_w, target_h, "disabled")
                        upscaled_frames = scaled_res[0] if isinstance(scaled_res, tuple) else scaled_res

                    video_pipe.frames = upscaled_frames
                    video_pipe.height = upscaled_frames.shape[1]
                    video_pipe.width = upscaled_frames.shape[2]
            except Exception as e:
                log_node(f"Video Smart Upscale (Classic): Failed ({e}), skipping.", color="YELLOW")
                import traceback
                traceback.print_exc()

        return (video_pipe,)
