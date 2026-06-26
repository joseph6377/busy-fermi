import json
import os

manifest_path = r'y:\UmeAiRT-Studio-Repos\ComfyUI-UmeAiRT-Toolkit\data\model_manifest.json'

with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

# The mmproj file to add
mmproj_file = {
    "path": "text_encoders/QWEN/qwen2.5-vl-7b-instruct-mmproj-f16.gguf",
    "path_type": "text_encoders_qwen",
    "sha256": "",
    "size_mb": 1184
}

def update_qwen_bundle(bundle_name):
    if bundle_name not in manifest["QWEN"]:
        return
    bundle = manifest["QWEN"][bundle_name]
    for variant, data in bundle.items():
        if variant.startswith("GGUF_"):
            q_level = variant.replace("GGUF_", "")
            if q_level == "Q8":
                q_level = "Q8_0"
            elif q_level == "Q3":
                q_level = "Q3_K_S"
            elif q_level == "Q4":
                q_level = "Q4_K_S"
            elif q_level == "Q5":
                q_level = "Q5_K_S"
            elif q_level == "Q6":
                q_level = "Q6_K"
            
            # Find the text encoder file
            for i, file_obj in enumerate(data.get("files", [])):
                if file_obj["path_type"] == "text_encoders_qwen" and "UD" in file_obj["path"]:
                    # Replace UD with the non-UD version matching the quantization
                    new_path = f"text_encoders/QWEN/Qwen2.5-VL-7B-Instruct-{q_level}.gguf"
                    # Sizes (approximate based on dir listing)
                    sizes = {
                        "Q3_K_S": 3492,
                        "Q4_K_S": 4457,
                        "Q5_K_S": 5315,
                        "Q6_K": 6254,
                        "Q8_0": 8098
                    }
                    file_obj["path"] = new_path
                    file_obj["size_mb"] = sizes.get(q_level, 4500)
            
            # Check if mmproj is already there
            has_mmproj = any("mmproj" in f["path"].lower() for f in data.get("files", []))
            if not has_mmproj:
                data["files"].append(mmproj_file)

update_qwen_bundle("Image_Edit_2509")
update_qwen_bundle("Image_Edit_2511")
update_qwen_bundle("Image_2512")
update_qwen_bundle("Image_Distill")

with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

print("Manifest updated successfully!")
