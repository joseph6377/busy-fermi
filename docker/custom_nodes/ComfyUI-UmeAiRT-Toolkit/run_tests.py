import sys
from unittest.mock import MagicMock

# Force UTF-8 encoding for headless environments
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import importlib.machinery

def create_mock_module(name, base_mock=None):
    mock = base_mock if base_mock is not None else MagicMock()
    mock.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = mock
    return mock

# Mock ComfyUI modules completely
create_mock_module('comfy')
comfy_utils_mock = MagicMock()
comfy_utils_mock.model_trange = lambda *args, **kwargs: range(*args)
create_mock_module('comfy.utils', comfy_utils_mock)
create_mock_module('comfy.sd')
create_mock_module('comfy.sd1_clip')
create_mock_module('comfy.samplers')
create_mock_module('comfy.sample')
create_mock_module('comfy.model_management')
create_mock_module('nodes')
create_mock_module('node_helpers')
fp_mock = MagicMock()
fp_mock.supported_pt_extensions = {".pt", ".bin", ".safetensors", ".ckpt"}
fp_mock.get_full_path = lambda f, n: f"/{f}/{n}"
fp_mock.get_filename_list = lambda f: ["model.safetensors", "model.pt"]
fp_mock.folder_names_and_paths = {"checkpoints": (["/mock/models"], set()), "embeddings": (["/mock/embeds"], set())}
fp_mock.output_directory = "/fake/out"
fp_mock.models_dir = "/fake/models"
create_mock_module('folder_paths', fp_mock)
create_mock_module('psutil')
tqdm_mock = MagicMock()
tqdm_mock.tqdm = lambda *a, **kw: MagicMock()
create_mock_module('tqdm', tqdm_mock)
create_mock_module('tqdm.auto', tqdm_mock)
create_mock_module('tqdm._tqdm_pandas')

# Ensure numpy is available (required by detail_daemon_nodes schedule helpers)
try:
    import numpy
except ImportError:
    create_mock_module('numpy')

# Mock PIL/Pillow if not available (required by image_saver_core)
try:
    import PIL
    import PIL.Image
    import PIL.PngImagePlugin
except ImportError:
    pil_mock = MagicMock()
    create_mock_module('PIL', pil_mock)
    create_mock_module('PIL.Image', pil_mock)
    create_mock_module('PIL.PngImagePlugin', pil_mock)

# Use real torch/torchvision if available (required for sampler modules that use
# torch.Tensor, torch.zeros, etc.). Only mock if not installed.
try:
    import torch
    import torchvision
    import torchvision.transforms
    import torchvision.transforms.functional
except (ImportError, ValueError):
    class DummyTorch:
        def __init__(self):
            self._cache = {}
        def no_grad(self):
            def decorator(func):
                return func
            return decorator
        def __getattr__(self, name):
            if name not in self._cache:
                self._cache[name] = MagicMock()
            return self._cache[name]

    create_mock_module('torch', DummyTorch())
    create_mock_module('torchvision')
    create_mock_module('torchvision.transforms')
    create_mock_module('torchvision.transforms.functional')

# Lazy loads
create_mock_module('comfy.k_diffusion')
create_mock_module('comfy.k_diffusion.sampling')
create_mock_module('comfy.k_diffusion.utils')
create_mock_module('comfy.model_patcher')
create_mock_module('comfy.model_sampling')
create_mock_module('comfy.nested_tensor')
create_mock_module('comfy_extras')
create_mock_module('comfy_extras.nodes_upscale_model')
create_mock_module('comfy_extras.nodes_custom_sampler')
create_mock_module('comfy_extras.nodes_post_processing')
create_mock_module('comfy_extras.nodes_lt')
create_mock_module('comfy_extras.nodes_lt_audio')
create_mock_module('comfy_extras.nodes_lt_upsampler')
create_mock_module('comfy_extras.nodes_hunyuan')
create_mock_module('comfy.patcher_extension')
create_mock_module('comfy.clip_vision')
create_mock_module('comfy_extras.nodes_model_advanced')
create_mock_module('comfy_extras.nodes_cfg')
create_mock_module('comfy_extras.nodes_easycache')
create_mock_module('comfy_extras.nodes_nag')
create_mock_module('latent_preview')

import unittest

if __name__ == '__main__':
    # Run unittest discovery
    unittest.main(module=None, argv=['run_tests.py', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'])
