"""
UmeAiRT Toolkit — Image to Prompt (VLM)
--------------------------------------------------
Native Vision-Language Model captioning node.
Supports Qwen2-VL, Llama-3.2-Vision, and other modern VLMs.

Modes:
  - Tags:   Booru-style comma-separated tags
  - Prompt: Natural-language description
  - Mixed:  Hybrid tags + description
"""

import gc
import os
import re
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import folder_paths
import comfy.model_management as mm

from .common import log_node
from .manifest import get_llm_dropdowns, download_bundle_files

# ---------------------------------------------------------------------------
# LLM folder setup
# ---------------------------------------------------------------------------

_llm_dir = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(_llm_dir, exist_ok=True)
try:
    folder_paths.add_model_folder_path("LLM", _llm_dir)
except Exception:  # Non-critical: folder may already be registered
    pass

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_cached_model = {
    "path": None,
    "model": None,
    "processor": None,
    "dtype": None,
}

def _load_vlm(model_path, dtype, model_name):
    global _cached_model

    if _cached_model["path"] == model_path and _cached_model["dtype"] == dtype:
        return _cached_model["model"], _cached_model["processor"]

    # Unload previous model (accelerate dispatch hooks prevent .to('cpu'),
    # so we delete refs and force garbage collection instead)
    if _cached_model["model"] is not None:
        del _cached_model["model"]
        del _cached_model["processor"]
        _cached_model["model"] = None
        _cached_model["processor"] = None
        gc.collect()
        mm.soft_empty_cache()

    log_node(f"Image to Prompt: Loading {model_name} ({dtype})...", color="CYAN")

    from transformers import AutoProcessor
    import os
    
    # 🔒 SECURE OFFLINE FP8 KERNELS
    # Instead of downloading dynamic CUDA code from HF Hub (which fails trust verification and requires internet),
    # we point the 'kernels' module to the pre-packaged offline kernels shipped inside the Toolkit assets.
    toolkit_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_dir = os.path.join(toolkit_dir, "assets", "kernels", "finegrained-fp8")
    
    if os.path.exists(kernel_dir):
        # The kernels package has a bug on Windows where it splits LOCAL_KERNELS by ":" and breaks drive letters (E:\...)
        # We bypass this by monkey-patching the override function directly in memory.
        try:
            import kernels.utils
            from pathlib import Path
            kernels.utils._get_local_kernel_overrides = lambda: {"kernels-community/finegrained-fp8": Path(kernel_dir)}
        except ImportError:
            pass
    else:
        # Fallback to downloading if assets are missing
        os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
    
    try:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Use device_map="auto" so HuggingFace/accelerate handles GPU/CPU offloading
    # natively. This properly supports models larger than VRAM (e.g. 8B on a 16GB card).
    # ComfyUI's ModelPatcher is designed for diffusion UNets, not autoregressive VLMs.
    model = model_cls.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    _cached_model.update(
        path=model_path, model=model, processor=processor, dtype=dtype
    )

    log_node(f"Image to Prompt: ✅ {model_name} ready", color="GREEN")
    return model, processor


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class UmeAiRT_ImageToPrompt:
    """Vision-Language Model image captioning node."""

    @classmethod
    def INPUT_TYPES(s):
        models = get_llm_dropdowns()
        
        # Merge local models transparently (UmeAiRT standard pattern)
        local_models = []
        if os.path.isdir(_llm_dir):
            for item in os.listdir(_llm_dir):
                item_path = os.path.join(_llm_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                    local_models.append(item)
                    
        # Combine and deduplicate
        all_models = list(models) + [m for m in local_models if m not in models]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (all_models, {"tooltip": "Select VLM architecture from UmeAiRT Assets."}),
                "mode": (
                    ["Tags", "Prompt", "Mixed", "Custom"],
                    {
                        "default": "Prompt",
                        "tooltip": "Tags: booru-style. Prompt: natural description. Mixed: both. Custom: your own prompt."
                    },
                ),
                "keep_loaded": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Keep the VLM in VRAM after inference. Faster re-runs but uses memory."}),
            },
            "optional": {
                "custom_prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail.", "advanced": True, "tooltip": "Used only when mode is 'Custom'."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "advanced": True, "tooltip": "Controls randomness. Change the seed to get a different description of the same image."}),
                "max_image_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64, "display": "slider", "advanced": True, "tooltip": "Scales the image down before processing to save VRAM."}),
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "display": "slider", "advanced": True, "tooltip": "Maximum output length. 0 = auto (Tags: 64, Prompt: 256, Mixed: 384, Custom: 512)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "analyze"
    CATEGORY = "UmeAiRT/Tools"
    OUTPUT_NODE = True
    DESCRIPTION = "Generates text descriptions or booru-style tags from images using Qwen3-VL models. Supports Tags, Prompt, Mixed, and Custom modes with automatic token limits. Seed control for varied results on the same image."

    def analyze(self, image, model, mode, keep_loaded=True, custom_prompt="", seed=0, max_image_size=1024, max_tokens=0):
        if model == "No LLM Models Found":
            raise ValueError("No LLM models found in the manifest.")

        # Default to bfloat16 for modern VLMs (Qwen/Llama) to prevent NaN overflows causing '!!!!' garbage output.
        dtype = torch.bfloat16 if any(name in model.lower() for name in ["qwen", "llama"]) else torch.float16

        # 1. Resolve Model Path
        models_manifest = get_llm_dropdowns()
        
        # Check if the selected model is managed by the manifest
        if model in models_manifest:
            # Managed model: download via bundle loader
            manifest_key = f"LLM/{model}"
            
            # Since version dropdown is removed, we auto-select the first available version in the manifest
            from .manifest import load_manifest
            full_manifest = load_manifest()
            variant_data = full_manifest.get("LLM", {}).get(model, {})
            available_versions = [v for v in variant_data.keys() if not v.startswith("_")]
            version = available_versions[0] if available_versions else "v1"
            
            resolved, meta, dl, sk, errs = download_bundle_files(manifest_key, version)
            if errs:
                raise RuntimeError(f"Failed to download VLM files: {errs}")
                
            llm_files = resolved.get("llm", [])
            if not llm_files:
                raise RuntimeError(f"Manifest entry for {manifest_key}/{version} has no 'llm' files.")
                
            config_file = next((f for f in llm_files if f.endswith("config.json")), llm_files[0])
            model_dir = os.path.dirname(os.path.join(_llm_dir, config_file))
            model_id = f"{manifest_key}/{version}"
        else:
            # Unmanaged local folder
            model_dir = os.path.join(_llm_dir, model)
            if not os.path.isdir(model_dir):
                raise RuntimeError(f"Local model directory not found: {model_dir}")
            model_id = f"Local/{model}"

        # 2. Setup Prompts & auto max_tokens
        # Recommended token limits per mode (when max_tokens=0):
        #   Tags:   64  (20 tags × ~3 tokens each)
        #   Prompt: 256 (a few detailed sentences)
        #   Mixed:  384 (description + tags)
        #   Custom: 512 (user-defined, generous default)
        _default_tokens = {"Tags": 64, "Prompt": 256, "Mixed": 384, "Custom": 512}
        if max_tokens <= 0:
            max_tokens = _default_tokens.get(mode, 256)

        if mode == "Tags":
            prompt_text = "List exactly 20 descriptive booru tags for this image, separated by commas. Stop writing after the 20th tag."
        elif mode == "Mixed":
            prompt_text = "Write a 2-sentence description of this image, followed by a new line with exactly 15 comma-separated booru tags."
        elif mode == "Custom":
            prompt_text = custom_prompt if custom_prompt.strip() else "Describe this image."
        else:
            prompt_text = "Write a brief, one-paragraph description of this image. Focus on the main subject, style, and lighting. Do not write more than 3 sentences."

        # 3. Load Model (device_map="auto" handles GPU/CPU placement natively)
        hf_model, processor = _load_vlm(model_dir, dtype, model_id)

        # 4. Inference
        results = []
        image_batch = image.permute(0, 3, 1, 2)  # BHWC → BCHW

        for img_tensor in image_batch:
            pil_image = TF.to_pil_image(img_tensor)
            
            # Prevent OOM by scaling down huge images (Qwen/Llama token generation scales with resolution)
            if max(pil_image.size) > max_image_size:
                ratio = max_image_size / max(pil_image.size)
                new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                log_node(f"Image to Prompt: Image resized to {new_size} to save VRAM.", color="CYAN")
            
            # Most modern VLMs use a chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
            except Exception as e:
                # Fallback for models without chat templates or specific processor signatures
                log_node(f"Image to Prompt: Chat template failed, using fallback formatting. Error: {e}", color="YELLOW")
                try:
                    # Strip the image object from the template if it causes an error, and use the old API
                    fallback_messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
                    text = processor.apply_chat_template(fallback_messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[pil_image], padding=True, return_tensors="pt")
                except Exception as e:
                    log_node(f"Image to Prompt: Fallback formatting also failed ({e}), using raw processor.", color="YELLOW")
                    inputs = processor(text=prompt_text, images=pil_image, return_tensors="pt")
                
            inputs = inputs.to(hf_model.device)

            # Only override max_new_tokens and anti-repetition. All other generation
            # params (do_sample, eos_token_id, temperature, top_k, top_p) come from
            # the model's built-in generation_config.json — do NOT override them.
            generate_kwargs = {"max_new_tokens": max_tokens}

            # For tag-style outputs, prevent n-gram repetition at the token level.
            # This physically blocks the model from emitting the same word sequence twice.
            if mode in ("Tags", "Mixed"):
                generate_kwargs["no_repeat_ngram_size"] = 3

            # Seed controls sampling randomness — changing it gives a different "opinion"
            # on the same image. Also forces ComfyUI to re-execute the node.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            generated_ids = hf_model.generate(**inputs, **generate_kwargs)
            
            # Trim the prompt from the output
            if "input_ids" in inputs:
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
            else:
                generated_ids_trimmed = generated_ids
                
            decoded = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            # Post-process tag modes: deduplicate and clean up
            if mode in ("Tags", "Mixed"):
                tags = [t.strip() for t in decoded.split(",") if t.strip()]
                seen = set()
                unique_tags = []
                for tag in tags:
                    tag_lower = tag.lower()
                    if tag_lower not in seen:
                        seen.add(tag_lower)
                        unique_tags.append(tag)
                decoded = ", ".join(unique_tags)

            results.append(decoded)

        output_text = results[0] if len(results) == 1 else "\n---\n".join(results)
        log_node(f"Image to Prompt: ✅ [{mode}] {len(output_text)} chars", color="GREEN")

        # 5. Cleanup
        if not keep_loaded:
            global _cached_model
            del _cached_model["model"]
            del _cached_model["processor"]
            _cached_model.update(path=None, model=None, processor=None, dtype=None)
            gc.collect()
            mm.soft_empty_cache()

        return (output_text,)
