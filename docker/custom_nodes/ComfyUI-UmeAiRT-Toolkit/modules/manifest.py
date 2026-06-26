"""Model manifest loading, caching, and bundle resolution.

Fetches model_manifest.json from the UmeAiRT Assets repo (HuggingFace),
caches locally, and provides bundle dropdown population and file resolution
for the Bundle Auto-Loader.
"""
import os
import json
import urllib.request
import folder_paths
from .common import log_node
from .download_utils import get_hf_token, download_file


# Maps path_type values from model_manifest.json to ComfyUI folder names
PATH_TYPE_TO_FOLDERS = {
    # Diffusion models
    "flux_diff":       ["diffusion_models"],
    "flux_unet":       ["unet"],
    "zimg_diff":       ["diffusion_models"],
    "zimg_unet":       ["unet"],
    "wan_diff":        ["diffusion_models"],
    "anima_diff":      ["diffusion_models"],
    "anima_diff_gguf": ["diffusion_models"],
    "hidream_diff":    ["diffusion_models"],
    "qwen_diff":       ["diffusion_models"],
    "ltxv_diff":       ["diffusion_models"],
    "ltxv_ckpt":       ["checkpoints"],
    "ltx2_diff":       ["diffusion_models"],
    # Text encoders
    "clip":                ["clip", "text_encoders"],
    "text_encoders_t5":    ["text_encoders", "clip"],
    "text_encoders_qwen":  ["text_encoders", "clip"],
    "text_encoders_gemma": ["text_encoders", "clip"],
    "text_encoders_llama": ["text_encoders", "clip"],
    "text_encoders_ltx":   ["text_encoders", "clip"],
    # Vision / VAE / Other
    "clip_vision":     ["clip_vision"],
    "vae":             ["vae"],
    "latent_upscale":  ["latent_upscale_models", "upscale_models"],
    "frame_interpolation": ["frame_interpolation"],
    "melband":         ["custom"],
    # LoRAs (acceleration, style, etc.)
    "loras":           ["loras"],
    # LLM / VLM (Florence2, etc.)
    "llm":             ["LLM"],
}


def find_file_in_folders(filename, folder_types):
    """Search for a file across multiple ComfyUI folder types by filename only.

    Most users dump files at the root of the category folder, so we search
    by filename regardless of subdirectory structure.

    If a .aria2 or .download control file exists alongside the file, the
    previous download was interrupted — the file is considered incomplete.

    Args:
        filename (str): The filename to search for.
        folder_types (list[str]): ComfyUI folder type names to search in.

    Returns:
        str or None: The full path if found and complete, otherwise None.
    """
    for folder_type in folder_types:
        try:
            path = folder_paths.get_full_path(folder_type, filename)
            if not path:
                # Fallback: check models_dir directly
                fallback_path = os.path.join(folder_paths.models_dir, folder_type, filename)
                if os.path.exists(fallback_path):
                    path = fallback_path
            if path and os.path.exists(path):
                # Check for interrupted download markers
                if os.path.exists(path + ".aria2") or os.path.exists(path + ".download"):
                    log_node(f"  ⚠️ '{filename}' has incomplete download — will resume.", color="YELLOW")
                    return None
                return path
        except Exception as e:
            log_node(f"Bundle Loader: Error searching in '{folder_type}': {e}", color="YELLOW")
        # Also try GGUF-specific folders
        if folder_type == "unet":
            try:
                path = folder_paths.get_full_path("unet_gguf", filename)
                if path and os.path.exists(path):
                    if os.path.exists(path + ".aria2") or os.path.exists(path + ".download"):
                        log_node(f"  ⚠️ '{filename}' has incomplete download — will resume.", color="YELLOW")
                        return None
                    return path
            except Exception as e:
                log_node(f"Bundle Loader: unet_gguf lookup failed for '{filename}': {e}", color="YELLOW")
        if folder_type == "clip":
            try:
                path = folder_paths.get_full_path("clip_gguf", filename)
                if path and os.path.exists(path):
                    if os.path.exists(path + ".aria2") or os.path.exists(path + ".download"):
                        log_node(f"  ⚠️ '{filename}' has incomplete download — will resume.", color="YELLOW")
                        return None
                    return path
            except Exception as e:
                log_node(f"Bundle Loader: clip_gguf lookup failed for '{filename}': {e}", color="YELLOW")
    return None


def get_download_dest(filename, folder_type):
    """Get the download destination path (root of the first registered folder).

    Args:
        filename (str): The target filename.
        folder_type (str): The primary ComfyUI folder type name.

    Returns:
        str: The absolute path where the file should be downloaded.
    """
    try:
        paths = folder_paths.get_folder_paths(folder_type)
        if paths:
            dest_dir = paths[0]
            os.makedirs(dest_dir, exist_ok=True)
            return os.path.join(dest_dir, filename)
    except Exception as e:
        log_node(f"Bundle Loader: Could not resolve folder path for '{folder_type}': {e}", color="YELLOW")
    # Fallback: models/<folder_type>/
    fallback = os.path.join(folder_paths.models_dir, folder_type)
    os.makedirs(fallback, exist_ok=True)
    return os.path.join(fallback, filename)


# --- Manifest Loading & Caching ---

_MANIFEST_CACHE = None

_MANIFEST_URLS = [
    {"name": "HuggingFace", "url": "https://huggingface.co/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/main/model_manifest.json"},
    {"name": "ModelScope", "url": "https://www.modelscope.ai/datasets/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/master/model_manifest.json"}
]

# The manifest lives inside the toolkit: data/model_manifest.json
_BUNDLED_MANIFEST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "model_manifest.json")


def load_manifest():
    """Load the model manifest, fetching from remote if stale.

    The manifest is stored at data/model_manifest.json inside the toolkit.
    Remote fetches update this file in-place so users always have the
    freshest model catalog available, even after going offline.

    Priority:
    1. In-memory cache (fastest)
    2. Remote fetch from HuggingFace/ModelScope (if file is missing or >24h old)
    3. Bundled file (data/model_manifest.json — always present, ships with releases)

    Returns:
        dict: The parsed manifest data.
    """
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE

    import time
    manifest_path = _BUNDLED_MANIFEST_PATH
    cache_max_age = 24 * 60 * 60  # 24 hours

    # Check if local file is fresh enough
    need_fetch = True
    if os.path.exists(manifest_path):
        age = time.time() - os.path.getmtime(manifest_path)
        if age < cache_max_age:
            need_fetch = False

    # Try remote fetch
    if need_fetch:
        hf_token = get_hf_token()
        headers = {"User-Agent": "ComfyUI-UmeAiRT-Toolkit"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            
        sources = list(_MANIFEST_URLS)
        if os.environ.get("UMEAIRT_PREFER_MODELSCOPE", "0") in ("1", "true", "True"):
            sources.reverse()

        for source in sources:
            try:
                req = urllib.request.Request(source["url"], headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read()
                    data = json.loads(raw)
                    # Update the bundled file in-place
                    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
                    with open(manifest_path, 'wb') as f:
                        f.write(raw)
                    log_node(f"Bundle Loader: 📡 Model manifest updated from {source['name']}.", color="GREEN")
                    _MANIFEST_CACHE = data
                    return _MANIFEST_CACHE
            except Exception as e:
                log_node(f"Bundle Loader: Remote fetch from {source['name']} failed: {e}", color="YELLOW")

    # Read from local file
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8-sig') as f:
                _MANIFEST_CACHE = json.load(f)
                if not need_fetch:
                    log_node("Bundle Loader: 📦 Using local model manifest (cache is fresh).", color="CYAN")
                else:
                    log_node("Bundle Loader: ⚠️ Using local model manifest (offline fallback).", color="YELLOW")
                return _MANIFEST_CACHE
        except Exception as e:
            log_node(f"Bundle Loader: Failed to read manifest: {e}", color="RED")

    log_node("Bundle Loader: ❌ No model manifest found. Bundle Auto-Loader will be empty.", color="RED")
    return {}



# Standardized version list — ordered from highest quality to most compressed.
# This is the canonical dropdown; exotic/internal version keys are excluded.
STANDARD_VERSIONS = ["bf16", "fp16", "fp8", "GGUF_Q8", "GGUF_Q6", "GGUF_Q5", "GGUF_Q4", "GGUF_Q3"]

# Bundle types that should NOT appear in the Auto-Loader dropdown.
_EXCLUDED_BUNDLE_TYPES = {"system"}
_EXCLUDED_LOADER_TYPES = {"preprocessor"}


def _is_loadable_variant(variant_data):
    """Return True if a variant is a user-loadable model (not a preprocessor/system bundle)."""
    meta = variant_data.get("_meta", {})
    if meta.get("bundle_type", "") in _EXCLUDED_BUNDLE_TYPES:
        return False
    if meta.get("loader_type", "") in _EXCLUDED_LOADER_TYPES:
        return False
    return True


def get_bundle_dropdowns():
    """Return (categories, versions) lists from model manifest for dropdown population.

    Categories are FAMILY/VARIANT pairs (e.g. 'FLUX/Dev', 'Z-IMAGE/Turbo').
    System/preprocessor bundles are automatically excluded.
    Versions are a fixed, ordered list of standard precision tiers.
    """
    data = load_manifest()
    categories = []

    for family_key, family_data in data.items():
        if family_key.startswith("_") or family_key == "LLM":
            continue
        if not isinstance(family_data, dict):
            continue

        # Detect manifest v3 (has _family_meta) vs legacy (has _meta directly)
        if "_family_meta" in family_data:
            # Manifest v3: FAMILY → VARIANT → version
            for variant_key, variant_data in family_data.items():
                if variant_key.startswith("_") or not isinstance(variant_data, dict):
                    continue
                if not _is_loadable_variant(variant_data):
                    continue
                cat_label = f"{family_key}/{variant_key}"
                categories.append(cat_label)
        else:
            # Legacy flat structure: CATEGORY → version (no _family_meta)
            if not _is_loadable_variant(family_data):
                continue
            categories.append(family_key)

    if not categories:
        categories = ["No Bundles Found"]
    return categories, list(STANDARD_VERSIONS)


def get_llm_dropdowns():
    """Return models list for LLM/VLM models."""
    data = load_manifest()
    models = []
    
    llm_data = data.get("LLM", {})
    if isinstance(llm_data, dict):
        for variant_key, variant_data in llm_data.items():
            if variant_key.startswith("_") or not isinstance(variant_data, dict):
                continue
            models.append(variant_key)
            
    if not models:
        models = ["No LLM Models Found"]
    
    return models


def download_bundle_files(category, version):
    """Download all files for a bundle, skipping already-present ones.

    Supports both manifest v3 ('FAMILY/VARIANT' categories) and legacy flat categories.

    Returns:
        tuple: (resolved_files dict, meta dict, downloaded count, skipped count, errors list)
    """
    hf_token = get_hf_token()
    if not hf_token:
        log_node(
            "💡 No HF token found. To speed up downloads, create a token at "
            "https://huggingface.co/settings/tokens and set HF_TOKEN in your environment variables.",
            color="YELLOW"
        )

    data = load_manifest()

    # Resolve category to the right level in the manifest
    if "/" in category:
        # Manifest v3: FAMILY/VARIANT
        family_key, variant_key = category.split("/", 1)
        if family_key not in data:
            raise ValueError(f"Family '{family_key}' not found in manifest.")
        family_data = data[family_key]
        if variant_key not in family_data:
            raise ValueError(f"Variant '{variant_key}' not found for {family_key}.")
        variant_data = family_data[variant_key]
        meta = variant_data.get("_meta", {})
        sources = data.get("_sources", {})
        base_urls = [
            sources.get("huggingface", "https://huggingface.co/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/main/models"),
            sources.get("modelscope", "https://www.modelscope.ai/datasets/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/master/models")
        ]
    else:
        # Legacy flat structure
        if category not in data:
            raise ValueError(f"Category '{category}' not found in manifest.")
        variant_data = data[category]
        meta = variant_data.get("_meta", {})
        base_urls = [meta.get("base_url", "https://huggingface.co/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/main/models")]

    if os.environ.get("UMEAIRT_PREFER_MODELSCOPE", "0") in ("1", "true", "True") and len(base_urls) > 1:
        base_urls.reverse()

    if version not in variant_data:
        available = [v for v in variant_data.keys() if v != "_meta"]
        raise ValueError(
            f"Version '{version}' is not available for {category}.\n"
            f"  Available versions: {', '.join(available)}"
        )
    bundle_def = variant_data[version]
    files = bundle_def.get("files", [])
    min_vram = bundle_def.get("min_vram", 0)
    log_node(f"📥 {category} / {version} ({len(files)} files, min VRAM: {min_vram}GB)")

    resolved_files = {}
    downloaded = 0
    skipped = 0
    errors = []

    for file_entry in files:
        pt = file_entry["path_type"]
        # Manifest v3 uses "path", legacy uses "filename" + "url"
        rel_path = file_entry.get("path", file_entry.get("url", ""))
        rel_path = rel_path.lstrip("/")
        
        if pt == "llm":
            # For LLMs, we must preserve the directory structure (e.g., model_name/config.json)
            # otherwise config.json files will collide.
            filename = rel_path if rel_path else file_entry.get("filename", "")
            # Prevent nested LLM/LLM/ folders if the manifest path starts with LLM/
            if filename.startswith("LLM/"):
                filename = filename[4:]
        else:
            filename = os.path.basename(rel_path) if rel_path else file_entry.get("filename", "")
            
        expected_sha256 = file_entry.get("sha256", "")
        folder_types = PATH_TYPE_TO_FOLDERS.get(pt, [pt])
        
        # If it's an LLM, find_file_in_folders might match a different config.json.
        # Check explicit relative path first if pt == "llm".
        local_path = None
        if pt == "llm":
            explicit_path = os.path.join(folder_paths.models_dir, "LLM", filename)
            if os.path.exists(explicit_path):
                local_path = explicit_path
        
        if not local_path:
            local_path = find_file_in_folders(filename, folder_types)
            
        if local_path:
            log_node(f"  ✅ '{filename}' already present — skipping.", color="GREEN")
            skipped += 1
        else:
            downloaded_this_file = False
            for base_url in base_urls:
                try:
                    full_url = f"{base_url}/{rel_path}" if not rel_path.startswith("http") else rel_path
                    dest = get_download_dest(filename, folder_types[0])
                    # Ensure parent dirs exist for LLMs
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    download_file(full_url, dest, hf_token=hf_token, expected_sha256=expected_sha256)
                    downloaded += 1
                    downloaded_this_file = True
                    break
                except Exception as e:
                    log_node(f"  ⚠️ Failed from {base_url}: {e}", color="YELLOW")
                    
            if not downloaded_this_file:
                log_node(f"  ❌ Failed to download '{filename}' from all sources.", color="RED")
                errors.append((filename, folder_types[0], rel_path))
        if pt not in resolved_files:
            resolved_files[pt] = []
        resolved_files[pt].append(filename)

    # If any files failed, log a clear manual-install guide
    if errors:
        log_node(
            f"\n{'='*60}\n"
            f"  ❌ {len(errors)} file(s) could not be downloaded for {category}/{version}.\n"
            f"  If you are offline, you can install them manually:\n"
            f"{'='*60}",
            color="RED"
        )
        for filename, folder_type, rel_path in errors:
            dest_dir = get_download_dest(filename, folder_type)
            dest_folder = os.path.dirname(dest_dir)
            url = f"{base_urls[0]}/{rel_path}"
            log_node(
                f"  📄 {filename}\n"
                f"     → Place in: {dest_folder}\n"
                f"     → Download: {url}",
                color="YELLOW"
            )
        log_node(f"{'='*60}\n", color="RED")

    return resolved_files, meta, downloaded, skipped, [e[0] for e in errors]

