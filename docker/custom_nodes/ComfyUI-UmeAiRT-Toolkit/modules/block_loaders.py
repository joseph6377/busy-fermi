"""ComfyUI node definitions for model loading (Files Settings and Bundle Loader).

This module defines the UI-facing ComfyUI nodes. Download logic is in
download_utils.py and manifest handling is in manifest.py.
"""
import torch
import os
import folder_paths
import nodes as comfy_nodes
import comfy.sd
import comfy.utils
import comfy.clip_vision
from .common import GenerationContext, UmeBundle, log_node

# Re-export from refactored modules for backward compatibility
from .download_utils import get_hf_token, download_file, verify_file_hash
from .manifest import (
    PATH_TYPE_TO_FOLDERS, find_file_in_folders, get_download_dest,
    load_manifest, get_bundle_dropdowns, download_bundle_files,
)



# --- Shared Loader Helper ---

_is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

def _load_diffusion_model(filename, folder="diffusion_models", model_path=None):
    """Load a diffusion model (safetensors or GGUF) with dtype auto-detection.

    Centralizes the GGUF/dtype logic shared by all loaders (FLUX, ZIMG, QWEN,
    HiDream, WAN, LTX, and Bundle Auto-Loader).

    Args:
        filename (str): The model filename (may end in .gguf or .safetensors).
        folder (str): The ComfyUI folder category (default: "diffusion_models").
            Ignored when ``model_path`` is provided.
        model_path (str | None): Pre-resolved absolute path. When provided,
            skips folder_paths resolution. Used by BundleLoader which resolves
            paths via ``find_file_in_folders()``.

    Returns:
        The loaded diffusion model object.
    """
    if filename.endswith(".gguf"):
        from ..vendor.comfyui_gguf.gguf_nodes import UnetLoaderGGUF
        return UnetLoaderGGUF().load_unet(filename)[0]
    
    if model_path is None:
        model_path = folder_paths.get_full_path(folder, filename)
        if not model_path and folder == "diffusion_models":
            model_path = folder_paths.get_full_path("unet", filename)
        
    model_options = {}
    ln = filename.lower()
    if "e4m3fn" in ln:
        if _is_mps:
            log_node("\u26a0\ufe0f FP8 (e4m3fn) is not supported on MPS \u2014 model may fail to load. Use FP16 instead.", color="YELLOW")
        model_options["dtype"] = torch.float8_e4m3fn
    elif "e5m2" in ln:
        if _is_mps:
            log_node("\u26a0\ufe0f FP8 (e5m2) is not supported on MPS \u2014 model may fail to load. Use FP16 instead.", color="YELLOW")
        model_options["dtype"] = torch.float8_e5m2
    return comfy.sd.load_diffusion_model(model_path, model_options=model_options)

def _get_combined_dropdowns():
    """Helper to merge modern and legacy folder paths and manually inject .gguf files."""
    def _get_files_with_gguf(folder_name):
        # 1. Get the cached list from ComfyUI (which might be missing .gguf)
        base_list = folder_paths.get_filename_list(folder_name)
        if base_list is None: base_list = []
        else: base_list = list(base_list)

        # 2. Manually scan the folders for .gguf to bypass ComfyUI's cache/extension filters
        if folder_name in folder_paths.folder_names_and_paths:
            paths = folder_paths.folder_names_and_paths[folder_name][0]
            for p in paths:
                if not os.path.exists(p): continue
                for root, _, files in os.walk(p, followlinks=True):
                    for f in files:
                        if f.endswith(".gguf"):
                            # Compute relative path like ComfyUI does
                            relpath = os.path.relpath(os.path.join(root, f), p)
                            relpath = relpath.replace("\\", "/")
                            if relpath not in base_list:
                                base_list.append(relpath)
        return base_list

    diff_models = _get_files_with_gguf("diffusion_models")
    clips = _get_files_with_gguf("clip")
    clips.extend([x for x in _get_files_with_gguf("text_encoders") if x not in clips])
    vaes = _get_files_with_gguf("vae")

    diff_models.sort()
    clips.sort()
    vaes.sort()
    return diff_models, clips, vaes

def _resolve_clip_path(filename):
    """Find text encoder in clip/ or text_encoders/ folders."""
    if not filename: return None
    p = folder_paths.get_full_path("clip", filename)
    if not p:
        p = folder_paths.get_full_path("text_encoders", filename)
    return p


# --- Files / Model Loaders (Block) ---

class UmeAiRT_FilesSettings_Checkpoint:
    """Simplified loader for standard SD1.5 / SDXL checkpoints."""

    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        vaes = ["Baked VAE"] + folder_paths.get_filename_list("vae")
        return {
            "required": {
                "ckpt_name": (checkpoints, {"tooltip": "Select a checkpoint file."}),
            },
            "optional": {
                "vae_name": (vaes, {"default": "Baked VAE", "advanced": True, "tooltip": "Select an external VAE or use the one baked into the checkpoint."}),
                "clip_skip": ("INT", {"default": 1, "min": 1, "max": 24, "step": 1, "display": "slider", "advanced": True, "tooltip": "CLIP skip (1 = no skip, 2 = skip one layer, common for anime)."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = 'Loads a standard SD1.5 or SDXL checkpoint with VAE.'

    def load_checkpoint(self, ckpt_name, vae_name="Baked VAE", clip_skip=1):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)
        model, clip, vae_ckpt = out[:3]

        real_clip_skip = -clip_skip
        if real_clip_skip < -1:
            clip = clip.clone()
            clip.clip_layer(real_clip_skip)

        if vae_name != "Baked VAE":
            vae_path = folder_paths.get_full_path("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        else:
            vae = vae_ckpt
            
        bundle_type = "image_inpaint" if "inpaint" in ckpt_name.lower() else ""
        return (UmeBundle(model=model, clip=clip, vae=vae, model_name=ckpt_name, bundle_type=bundle_type),)







class UmeAiRT_FilesSettings_FLUX:
    """FLUX model loader (diffusion model + dual CLIP + VAE).

    Loads a FLUX architecture model with its required dual text encoder
    (CLIP-L + T5-XXL) and VAE. Supports both standard safetensors and
    quantized GGUF model formats.
    """

    @classmethod
    def INPUT_TYPES(s):
        diff_models, clips, vaes = _get_combined_dropdowns()
        return {
            "required": {
                "diff_model": (diff_models, {"tooltip": "The FLUX diffusion model file (e.g. flux1-dev-fp8.safetensors or a GGUF variant)."}),
                "clip_1": (clips, {"tooltip": "First text encoder — typically CLIP-L (clip_l.safetensors)."}),
                "clip_2": (clips, {"tooltip": "Second text encoder — typically T5-XXL (t5xxl_fp16.safetensors or GGUF)."}),
                "vae": (vaes, {"tooltip": "VAE model (e.g. ae.safetensors for FLUX)."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_flux"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = 'Auto-loads a FLUX model bundle with its required text encoders (CLIP-L + T5) and VAE.'

    def load_flux(self, diff_model, clip_1, clip_2, vae):
        model_name = diff_model
        # Model
        model = _load_diffusion_model(diff_model)
        # Dual CLIP
        if clip_1.endswith(".gguf") or clip_2.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import DualCLIPLoaderGGUF
            clip = DualCLIPLoaderGGUF().load_clip(clip_1, clip_2, type="flux")[0]
        else:
            clip_paths = [
                _resolve_clip_path(clip_1),
                _resolve_clip_path(clip_2),
            ]
            ct = getattr(comfy.sd.CLIPType, "FLUX", comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip = comfy.sd.load_clip(
                ckpt_paths=clip_paths, 
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=ct
            )
        # VAE
        vae_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path("vae", vae)))
        # Infer bundle_type for inpainting models
        bundle_type = "image_inpaint" if "fill" in model_name.lower() or "inpaint" in model_name.lower() else ""
        return (UmeBundle(model=model, clip=clip, vae=vae_obj, model_name=model_name, loader_type="flux", bundle_type=bundle_type),)


class UmeAiRT_FilesSettings_ZIMG:
    """Z-IMAGE model loader (Lumina2 architecture).

    Loads a Z-IMAGE architecture model with its Qwen text encoder
    and VAE. Supports safetensors and GGUF quantized formats.
    """

    @classmethod
    def INPUT_TYPES(s):
        diff_models, clips, vaes = _get_combined_dropdowns()
        return {
            "required": {
                "diff_model": (diff_models, {"tooltip": "The Z-IMAGE diffusion model file (e.g. z-image-turbo-bf16.safetensors or GGUF)."}),
                "clip": (clips, {"tooltip": "Qwen text encoder (e.g. qwen3-4b.safetensors or GGUF)."}),
                "vae": (vaes, {"tooltip": "VAE model (e.g. ae.safetensors)."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_zimg"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = 'Auto-loads a Z-IMAGE (Lumina2) model bundle with its Qwen text encoder and VAE.'

    def load_zimg(self, diff_model, clip, vae):
        model_name = diff_model
        # Model
        model = _load_diffusion_model(diff_model)
        # CLIP (Lumina2 / Qwen)
        if clip.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
            clip_obj = CLIPLoaderGGUF().load_clip(clip, type="lumina2")[0]
        else:
            clip_path = _resolve_clip_path(clip)
            ct = getattr(comfy.sd.CLIPType, "LUMINA2", comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip_obj = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=ct
            )
        # VAE
        vae_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path("vae", vae)))
        # Infer bundle_type for inpainting models
        bundle_type = "image_inpaint" if "inpaint" in model_name.lower() else ""
        return (UmeBundle(model=model, clip=clip_obj, vae=vae_obj, model_name=model_name, loader_type="zimg", bundle_type=bundle_type),)


class UmeAiRT_BundleLoader:
    """Bundle Auto-Loader: select a model + version, auto-download missing files, and load them.

    Combines downloading and loading into one node.
    Reads from the remote model manifest to populate dropdowns and determine loading strategy.
    """

    @classmethod
    def INPUT_TYPES(s):
        categories, versions_list = get_bundle_dropdowns()
        return {
            "required": {
                "category": (categories, {"tooltip": "Select model family (e.g. FLUX/Dev, Z-IMAGE/Turbo)."}),
                "version": (versions_list, {"tooltip": "Select quantization/precision version (e.g. fp16, GGUF_Q4)."}),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_bundle"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = 'Automatically downloads and loads any UmeAiRT model bundle from the remote manifest.'

    def load_bundle(self, category, version):
        """Download missing files and load the selected model bundle."""
        resolved_files, meta, downloaded, skipped, errors = download_bundle_files(category, version)
        if errors:
            raise RuntimeError(f"Bundle Loader: {len(errors)} file(s) failed to download: {', '.join(errors)}")
        loader_type = meta.get("loader_type", "zimg")
        clip_type_str = meta.get("clip_type", "lumina2")

        model, clip, vae = None, None, None
        model_low_noise = None
        model_name = ""
        model_pt = None
        for pt_key in ["zimg_diff", "flux_diff", "wan_diff", "anima_diff", "anima_diff_gguf", "hidream_diff", "qwen_diff",
                        "ltxv_diff", "ltx2_diff", "zimg_unet", "flux_unet"]:
            if pt_key in resolved_files:
                model_pt = pt_key
                break
        if model_pt:
            diff_files = resolved_files[model_pt]
            # --- MoE detection: WAN 2.2 14B uses two diff models (high-noise + low-noise) ---
            _HN_PATTERNS = ("highnoise", "high_noise", "high-noise")
            _LN_PATTERNS = ("lownoise", "low_noise", "low-noise")

            if len(diff_files) >= 2:
                high_file = None
                low_file = None
                for f in diff_files:
                    fl = f.lower()
                    if any(p in fl for p in _HN_PATTERNS):
                        high_file = f
                    elif any(p in fl for p in _LN_PATTERNS):
                        low_file = f
                if high_file and low_file:
                    # MoE path: load both models
                    model_name = high_file
                    model = self._load_single_model(high_file, model_pt)
                    model_low_noise = self._load_single_model(low_file, model_pt)
                    log_node(f"Bundle Loader: MoE loaded (high={high_file}, low={low_file})", color="CYAN")
                else:
                    # Fallback: multiple diff files but no MoE pattern — use first
                    model_filename = diff_files[0]
                    model_name = model_filename
                    model = self._load_single_model(model_filename, model_pt)
            else:
                model_filename = diff_files[0]
                model_name = model_filename
                model = self._load_single_model(model_filename, model_pt)

        # Collect all text encoder files from various path types
        clip_files = []
        for te_key in ["clip", "text_encoders", "text_encoders_t5", "text_encoders_qwen",
                        "text_encoders_gemma", "text_encoders_llama", "text_encoders_ltx"]:
            clip_files.extend(resolved_files.get(te_key, []))
            
        # Do not pass mmproj directly; ComfyUI-GGUF handles it implicitly
        clip_files = [f for f in clip_files if "mmproj" not in f.lower()]

        if clip_files:
            has_gguf = any(f.endswith(".gguf") for f in clip_files)
            clip_paths = [find_file_in_folders(cf, ["clip", "text_encoders"]) for cf in clip_files]
            missing = [cf for cf, cp in zip(clip_files, clip_paths) if not cp]
            if missing:
                raise ValueError(f"Bundle Loader: CLIP file(s) not found: {', '.join(missing)}")

            clip_type_str = meta.get("clip_type", "stable_diffusion")
            if clip_type_str.lower() == "umt5":
                clip_type_str = "wan"
            if clip_type_str.lower() == "gemma3":
                clip_type_str = "ltxv"
            ct = getattr(comfy.sd.CLIPType, clip_type_str.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

            if has_gguf:
                from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
                gguf_loader = CLIPLoaderGGUF()
                clip_data = gguf_loader.load_data(clip_paths)
                clip = gguf_loader.load_patcher(clip_paths, ct, clip_data)
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=clip_paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=ct
                )

        vae_files = resolved_files.get("vae", [])
        if vae_files:
            vp = find_file_in_folders(vae_files[0], ["vae"])
            if not vp: raise ValueError(f"Bundle Loader: VAE '{vae_files[0]}' not found.")
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vp))

        # Audio VAE (LTX-2.3: second VAE file in the manifest)
        audio_vae = None
        if len(vae_files) >= 2:
            avp = find_file_in_folders(vae_files[1], ["vae"])
            if avp:
                metadata = None
                if avp.lower().endswith(".safetensors"):
                    try:
                        import safetensors
                        with safetensors.safe_open(avp, framework="pt", device="cpu") as f:
                            metadata = f.metadata()
                    except Exception:
                        pass
                audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(avp), metadata=metadata)
                log_node(f"Bundle Loader: Audio VAE loaded ({vae_files[1]})", color="CYAN")

        # Latent upscale model (LTX-2.3: spatial upscaler 2x)
        latent_upscale_model = None
        upscale_files = resolved_files.get("latent_upscale", [])
        if upscale_files:
            up_path = find_file_in_folders(upscale_files[0], PATH_TYPE_TO_FOLDERS.get("latent_upscale", ["upscale_models"]))
            if up_path:
                from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
                latent_upscale_model = LatentUpscaleModelLoader.execute(upscale_files[0]).args[0]
                log_node(f"Bundle Loader: Latent upscaler loaded ({upscale_files[0]})", color="CYAN")

        # CLIP Vision (for I2V video bundles)
        clip_vision_obj = None
        cv_files = resolved_files.get("clip_vision", [])
        if cv_files:
            cv_path = find_file_in_folders(cv_files[0], PATH_TYPE_TO_FOLDERS.get("clip_vision", ["clip_vision"]))
            if cv_path:
                clip_vision_obj = comfy.clip_vision.load(cv_path)
                log_node(f"Bundle Loader: CLIP Vision loaded ({cv_files[0]})", color="CYAN")

        bundle_type = meta.get("bundle_type", "")
        shift = meta.get("shift", 0.0)
        log_node(f"Bundle Loader: ✅ {category}/{version} ready.", color="GREEN")
        return (UmeBundle(model=model, model_low_noise=model_low_noise, clip=clip, vae=vae,
                          model_name=model_name, bundle_type=bundle_type, loader_type=loader_type,
                          shift=shift, clip_vision=clip_vision_obj,
                          audio_vae=audio_vae, latent_upscale_model=latent_upscale_model),)

    def _load_single_model(self, model_filename, model_pt):
        """Load a single diffusion model, delegating to the shared loader.

        Resolves the path via ``find_file_in_folders`` (manifest-aware) and
        delegates GGUF/dtype handling to ``_load_diffusion_model``.
        """
        resolved = None
        if not model_filename.endswith(".gguf"):
            resolved = find_file_in_folders(model_filename, PATH_TYPE_TO_FOLDERS.get(model_pt, ["diffusion_models"]))
            if not resolved:
                raise ValueError(f"Bundle Loader: Model '{model_filename}' not found.")
        return _load_diffusion_model(model_filename, model_path=resolved)


class UmeAiRT_FilesSettings_QWEN:
    """QWEN Image model loader (qwen2.5-vl architecture).

    Loads a QWEN Image architecture model with its dedicated
    Qwen2.5-VL text encoder and QWEN-specific VAE.
    Supports safetensors and GGUF quantized formats.
    """

    @classmethod
    def INPUT_TYPES(s):
        diff_models, clips, vaes = _get_combined_dropdowns()
        return {
            "required": {
                "diff_model": (diff_models, {"tooltip": "The QWEN diffusion model file."}),
                "clip": (clips, {"tooltip": "Qwen2.5-VL text encoder (e.g. qwen2.5-vl-7b-instruct.safetensors)."}),
                "vae": (vaes, {"tooltip": "QWEN-specific VAE (e.g. qwen_image_vae.safetensors)."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_qwen"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = 'Auto-loads a QWEN Image model bundle with its Qwen2.5-VL text encoder and VAE.'

    def load_qwen(self, diff_model, clip, vae):
        model_name = diff_model
        # Model
        model = _load_diffusion_model(diff_model)
        # CLIP (QWEN_IMAGE)
        if clip.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
            clip_obj = CLIPLoaderGGUF().load_clip(clip, type="qwen_image")[0]
        else:
            clip_path = _resolve_clip_path(clip)
            ct = getattr(comfy.sd.CLIPType, "QWEN_IMAGE", comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip_obj = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=ct
            )
        # VAE
        vae_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path("vae", vae)))
        # Infer bundle_type for inpainting models
        bundle_type = "image_inpaint" if "inpaint" in model_name.lower() else ""
        return (UmeBundle(model=model, clip=clip_obj, vae=vae_obj,
                          model_name=model_name, loader_type="qwen", bundle_type=bundle_type),)


class UmeAiRT_FilesSettings_ANIMA:
    """Anima model loader (Cosmos-Predict2 architecture).

    Loads an Anima model with its Qwen 3B text encoder
    and QWEN VAE. Supports safetensors and GGUF quantized formats.
    """

    @classmethod
    def INPUT_TYPES(s):
        diff_models, clips, vaes = _get_combined_dropdowns()
        return {
            "required": {
                "diff_model": (diff_models, {"tooltip": "The Anima diffusion model file (e.g. anima-base-v1.0.safetensors or GGUF)."}),
                "clip": (clips, {"tooltip": "Text encoder (e.g. qwen_3_06b_base.safetensors)."}),
                "vae": (vaes, {"tooltip": "VAE model (e.g. qwen_image_vae.safetensors)."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_anima"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = 'Auto-loads an Anima model bundle with its text encoder and VAE.'

    def load_anima(self, diff_model, clip, vae):
        model_name = diff_model
        # Model
        model = _load_diffusion_model(diff_model)
        # CLIP (Stable Diffusion fallback for Anima)
        if clip.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
            clip_obj = CLIPLoaderGGUF().load_clip(clip, type="stable_diffusion")[0]
        else:
            clip_path = _resolve_clip_path(clip)
            ct = getattr(comfy.sd.CLIPType, "STABLE_DIFFUSION", comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip_obj = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=ct
            )
        # VAE
        vae_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path("vae", vae)))
        # Infer bundle_type for inpainting models
        bundle_type = "image_inpaint" if "inpaint" in model_name.lower() else ""
        return (UmeBundle(model=model, clip=clip_obj, vae=vae_obj, model_name=model_name, loader_type="anima", bundle_type=bundle_type),)


class UmeAiRT_FilesSettings_HiDream:
    """HiDream model loader (QuadrupleCLIP architecture).

    Loads a HiDream model with its 4 text encoders (CLIP-L, CLIP-G,
    T5-XXL, LLaMA 3.1 8B) and VAE. The shift parameter controls
    ModelSamplingSD3 applied by the Image Generator (default 2.0).
    """

    @classmethod
    def INPUT_TYPES(s):
        diff_models, clips, vaes = _get_combined_dropdowns()
        return {
            "required": {
                "diff_model": (diff_models, {"tooltip": "The HiDream diffusion model file."}),
                "clip_l": (clips, {"tooltip": "CLIP-L text encoder (e.g. clip_l_hidream.safetensors)."}),
                "clip_g": (clips, {"tooltip": "CLIP-G text encoder (e.g. clip_g_hidream.safetensors)."}),
                "t5xxl": (clips, {"tooltip": "T5-XXL text encoder (e.g. t5xxl_fp8_e4m3fn_scaled.safetensors)."}),
                "llama": (clips, {"tooltip": "LLaMA 3.1 8B text encoder (e.g. llama_3.1_8b_instruct_fp8_scaled.safetensors)."}),
                "vae": (vaes, {"tooltip": "VAE model (e.g. ae.safetensors)."}),
            },
            "optional": {
                "shift": ("FLOAT", {
                    "tooltip": "ModelSamplingSD3 shift value. Default 2.0 is recommended for HiDream. Set to 0 to disable.",
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "display": "slider", "advanced": True,
                }),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_hidream"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = 'Auto-loads a HiDream model bundle with its 4 text encoders and VAE.'

    def load_hidream(self, diff_model, clip_l, clip_g, t5xxl, llama, vae, shift=2.0):
        model_name = diff_model
        # Model
        model = _load_diffusion_model(diff_model)

        # QuadrupleCLIP — 4 text encoder files
        te_files = [clip_l, clip_g, t5xxl, llama]
        has_gguf = any(f.endswith(".gguf") for f in te_files)

        ct = getattr(comfy.sd.CLIPType, "HIDREAM", comfy.sd.CLIPType.STABLE_DIFFUSION)

        if has_gguf:
            from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
            gguf_loader = CLIPLoaderGGUF()
            # Do not pass mmproj directly; ComfyUI-GGUF handles it implicitly
            clip_paths = [_resolve_clip_path(f) for f in te_files if "mmproj" not in f.lower()]
            clip_data = gguf_loader.load_data(clip_paths)
            clip_obj = gguf_loader.load_patcher(clip_paths, ct, clip_data)
        else:
            clip_paths = [_resolve_clip_path(f) for f in te_files if "mmproj" not in f.lower()]
            clip_obj = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=ct
            )

        # VAE
        vae_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path("vae", vae)))
        return (UmeBundle(model=model, clip=clip_obj, vae=vae_obj,
                          model_name=model_name, loader_type="hidream", shift=shift),)


class UmeAiRT_FilesSettings_WAN:
    """WAN video model loader (WAN 2.1 / 2.2).

    Loads a WAN video model with its umt5-xxl text encoder, VAE,
    and optional CLIP Vision encoder for I2V conditioning.
    For WAN 2.2 MoE (14B), connect the low-noise expert via the optional input.
    Supports safetensors and GGUF quantized formats.
    """

    @classmethod
    def INPUT_TYPES(s):
        diff_models, clips, vaes = _get_combined_dropdowns()
        clip_visions = folder_paths.get_filename_list("clip_vision")
        return {
            "required": {
                "diff_model": (diff_models, {"tooltip": "The WAN diffusion model (high-noise expert for WAN 2.2 MoE, or single model for WAN 2.1)."}),
                "clip": (clips, {"tooltip": "Text encoder — umt5-xxl (e.g. umt5-xxl-encoder-fp8-e4m3fn-scaled.safetensors)."}),
                "vae": (vaes, {"tooltip": "WAN VAE (e.g. wan_2.1_vae.safetensors for 14B, wan2.2_vae.safetensors for 5B)."}),
            },
            "optional": {
                "diff_model_low_noise": (["None"] + diff_models, {
                    "default": "None",
                    "tooltip": "WAN 2.2 MoE: Low-Noise expert diffusion model. Leave 'None' for WAN 2.1 or single-model pipelines."
                }),
                "clip_vision": (["None"] + clip_visions, {
                    "default": "None",
                    "tooltip": "CLIP Vision encoder for Image-to-Video (e.g. clip_vision_h.safetensors). Required for I2V mode."
                }),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_wan"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = 'Auto-loads a WAN Video model bundle. Supports WAN 2.1 (single model) and WAN 2.2 MoE (dual high/low noise).'

    def load_wan(self, diff_model, clip, vae, diff_model_low_noise="None", clip_vision="None"):
        model_name = diff_model
        # Model (high-noise / primary)
        model = _load_diffusion_model(diff_model)
        # Model (low-noise / MoE — optional)
        model_low_noise = None
        if diff_model_low_noise and diff_model_low_noise != "None":
            model_low_noise = _load_diffusion_model(diff_model_low_noise)
            log_node(f"WAN Loader: MoE low-noise model loaded ({diff_model_low_noise})", color="CYAN")
        # CLIP (WAN / umt5-xxl)
        if clip.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
            clip_obj = CLIPLoaderGGUF().load_clip(clip, type="wan")[0]
        else:
            clip_path = folder_paths.get_full_path("clip", clip) or folder_paths.get_full_path("text_encoders", clip)
            ct = getattr(comfy.sd.CLIPType, "WAN", comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip_obj = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=ct
            )
        # VAE
        vae_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path("vae", vae)))
        # CLIP Vision (optional — for I2V conditioning)
        clip_vision_obj = None
        if clip_vision and clip_vision != "None":
            cv_path = folder_paths.get_full_path("clip_vision", clip_vision)
            if cv_path:
                clip_vision_obj = comfy.clip_vision.load(cv_path)
                log_node(f"WAN Loader: CLIP Vision loaded ({clip_vision})", color="CYAN")
            else:
                log_node(f"WAN Loader: CLIP Vision '{clip_vision}' not found, skipping.", color="YELLOW")

        return (UmeBundle(model=model, model_low_noise=model_low_noise, clip=clip_obj, vae=vae_obj,
                          model_name=model_name, loader_type="wan", clip_vision=clip_vision_obj),)


class UmeAiRT_FilesSettings_LTX:
    """LTX-2.3 video+audio model loader.

    Loads the LTX-2.3 model with Gemma 3 + LTX text projection dual CLIP,
    video VAE (bf16), audio VAE (bf16), and optional spatial upscaler.
    Supports GGUF quantized formats for both diffusion model and text encoders.
    """

    @classmethod
    def INPUT_TYPES(s):
        diff_models, clips, vaes = _get_combined_dropdowns()
        return {
            "required": {
                "diff_model": (diff_models, {"tooltip": "LTX-2.3 diffusion model (GGUF or safetensors)."}),
                "clip_gemma": (clips, {"tooltip": "Gemma 3 text encoder (e.g. gemma-3-12b-it-IQ4_XS.gguf)."}),
                "clip_ltx": (clips, {"tooltip": "LTX text projection / embeddings connector (e.g. ltx-2-*-embeddings_connector_dev_bf16.safetensors)."}),
                "video_vae": (vaes, {"tooltip": "LTX Video VAE (e.g. LTX2_video_vae_bf16.safetensors)."}),
                "audio_vae": (vaes, {"tooltip": "LTX Audio VAE (e.g. LTX2_audio_vae_bf16.safetensors)."}),
            },
            "optional": {
                "latent_upscale_model": (["None"] + (folder_paths.get_filename_list("latent_upscale_models") or []), {
                    "default": "None",
                    "tooltip": "Spatial upscaler 2x for the dual-pass pipeline (e.g. ltx-2-spatial-upscaler-x2-1.0.safetensors)."
                }),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_ltx"
    CATEGORY = "UmeAiRT/Loaders"
    DESCRIPTION = 'Loads LTX-2.3 video+audio model with Gemma 3 dual CLIP, video/audio VAEs, and optional spatial upscaler.'

    def load_ltx(self, diff_model, clip_gemma, clip_ltx, video_vae, audio_vae,
                 latent_upscale_model="None"):
        model_name = diff_model

        # --- Diffusion model ---
        model = _load_diffusion_model(diff_model)

        # --- Dual CLIP (Gemma 3 + LTX text projection) ---
        clip_paths = []
        for clip_file in [clip_gemma, clip_ltx]:
            if clip_file.endswith(".gguf"):
                # For GGUF CLIP, use the GGUF loader path
                cp = folder_paths.get_full_path("clip", clip_file) or folder_paths.get_full_path("text_encoders", clip_file)
            else:
                cp = folder_paths.get_full_path("clip", clip_file) or folder_paths.get_full_path("text_encoders", clip_file) or folder_paths.get_full_path("checkpoints", clip_file)
            if not cp:
                raise ValueError(f"LTX Loader: CLIP file '{clip_file}' not found.")
            clip_paths.append(cp)

        has_gguf = any(f.endswith(".gguf") for f in [clip_gemma, clip_ltx])
        ct = getattr(comfy.sd.CLIPType, "LTXV", comfy.sd.CLIPType.STABLE_DIFFUSION)

        if has_gguf:
            from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
            gguf_loader = CLIPLoaderGGUF()
            clip_data = gguf_loader.load_data(clip_paths)
            clip_obj = gguf_loader.load_patcher(clip_paths, ct, clip_data)
        else:
            clip_obj = comfy.sd.load_clip(
                ckpt_paths=clip_paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=ct
            )

        # --- Video VAE ---
        vae_path = folder_paths.get_full_path("vae", video_vae)
        if not vae_path:
            raise ValueError(f"LTX Loader: Video VAE '{video_vae}' not found.")
        vae_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        # --- Audio VAE ---
        audio_vae_path = folder_paths.get_full_path("vae", audio_vae)
        if not audio_vae_path:
            raise ValueError(f"LTX Loader: Audio VAE '{audio_vae}' not found.")
        # Audio VAE uses a specific prefix mapping
        audio_sd = comfy.utils.load_torch_file(audio_vae_path)
        audio_sd = comfy.utils.state_dict_prefix_replace(
            audio_sd, {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."},
            filter_keys=True
        )
        metadata = None
        if audio_vae_path.lower().endswith(".safetensors"):
            try:
                import safetensors
                with safetensors.safe_open(audio_vae_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata()
            except Exception:
                pass
        audio_vae_obj = comfy.sd.VAE(sd=audio_sd, metadata=metadata)

        # --- Latent upscale model (optional) ---
        latent_upscale_obj = None
        if latent_upscale_model and latent_upscale_model != "None":
            try:
                from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
                latent_upscale_obj = LatentUpscaleModelLoader.execute(latent_upscale_model).args[0]
                log_node(f"LTX Loader: Latent upscaler loaded ({latent_upscale_model})", color="CYAN")
            except Exception as e:
                log_node(f"LTX Loader: Failed to load latent upscaler: {e}", color="YELLOW")

        log_node(f"LTX Loader: ✅ {model_name} ready.", color="GREEN")
        return (UmeBundle(
            model=model, clip=clip_obj, vae=vae_obj,
            model_name=model_name, loader_type="ltx2",
            audio_vae=audio_vae_obj, latent_upscale_model=latent_upscale_obj,
        ),)
