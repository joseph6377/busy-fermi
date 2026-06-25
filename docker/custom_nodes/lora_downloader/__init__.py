import os
import urllib.request
import hashlib
import folder_paths
import comfy.utils
import comfy.sd

class LoadLoraFromURL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "url": ("STRING", {"default": ""}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora_url"
    CATEGORY = "loaders"

    def load_lora_url(self, model, clip, url, strength_model, strength_clip):
        if not url or url.strip() == "":
            return (model, clip)
            
        lora_paths = folder_paths.get_folder_paths("loras")
        loras_dir = lora_paths[0] if lora_paths else os.path.join(folder_paths.models_dir, "loras")
        
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        lora_name = f"downloaded_{url_hash}.safetensors"
        lora_path = os.path.join(loras_dir, lora_name)
        
        if not os.path.exists(lora_path):
            print(f"Downloading LoRA from {url} to {lora_path}...")
            os.makedirs(loras_dir, exist_ok=True)
            try:
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                with urllib.request.urlopen(req) as response:
                    with open(lora_path, 'wb') as f:
                        f.write(response.read())
                print("LoRA download complete.")
            except Exception as e:
                print(f"Error downloading LoRA: {e}")
                return (model, clip)
                
        try:
            lora = comfy.utils.load_torch_file(lora_path)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            return (model, clip)

NODE_CLASS_MAPPINGS = {
    "LoadLoraFromURL": LoadLoraFromURL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraFromURL": "Load LoRA From URL"
}
