"""
UmeAiRT Toolkit - Unit Tests for LTX Video Extender
------------------------------------------------------
Tests for node definition, input validation, and pipeline structure.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Force UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Mock ComfyUI + LTX dependencies
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.samplers'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['comfy.clip_vision'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['comfy.model_sampling'] = MagicMock()
sys.modules['comfy.nested_tensor'] = MagicMock()
sys.modules['nodes'] = MagicMock()
sys.modules['node_helpers'] = MagicMock()
sys.modules['comfy_extras'] = MagicMock()
sys.modules['comfy_extras.nodes_lt'] = MagicMock()
sys.modules['comfy_extras.nodes_lt_audio'] = MagicMock()
sys.modules['comfy_extras.nodes_lt_upsampler'] = MagicMock()
sys.modules['comfy_extras.nodes_hunyuan'] = MagicMock()
sys.modules['comfy_extras.nodes_custom_sampler'] = MagicMock()
sys.modules['comfy_extras.nodes_post_processing'] = MagicMock()
sys.modules['av'] = MagicMock()

# folder_paths mock
mock_fp = MagicMock()
mock_fp.get_filename_list.return_value = []
mock_fp.get_folder_paths.return_value = []
mock_fp.get_full_path.return_value = None
sys.modules['folder_paths'] = mock_fp

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use real torch if available, else mock
try:
    import torch
except ImportError:
    sys.modules['torch'] = MagicMock()
    sys.modules['torchvision'] = MagicMock()
    sys.modules['torchvision.transforms'] = MagicMock()
    sys.modules['torchvision.transforms.functional'] = MagicMock()
    import torch

from modules.common import UmeBundle, UmeVideoSettings, VideoGenerationContext


class TestLTXVideoExtenderDefinition(unittest.TestCase):
    """Tests for UmeAiRT_LTXVideoExtender node definition."""

    def setUp(self):
        from modules.block_nodes import UmeAiRT_LTXVideoExtender
        self.cls = UmeAiRT_LTXVideoExtender

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("model_bundle", required)
        self.assertIn("video_settings", required)
        self.assertIn("video_pipe", required)
        self.assertIn("positive", required)

    def test_optional_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        optional = inputs["optional"]
        self.assertIn("negative", optional)
        self.assertIn("loras", optional)

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_VIDEO_PIPELINE",))
        self.assertEqual(self.cls.RETURN_NAMES, ("video_pipe",))
        self.assertEqual(self.cls.FUNCTION, "extend")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Video")

    def test_has_description(self):
        self.assertTrue(hasattr(self.cls, 'DESCRIPTION'))
        self.assertIn("extend", self.cls.DESCRIPTION.lower())

    def test_instantiation(self):
        node = self.cls()
        self.assertIsNotNone(node)

    def test_extend_requires_frames(self):
        """Should raise ValueError when video_pipe has no frames."""
        node = self.cls()
        bundle = UmeBundle(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
        settings = UmeVideoSettings(width=768, height=512, frame_rate=25, seed=42)
        ctx = VideoGenerationContext()
        ctx.frames = None  # No frames

        with self.assertRaises(ValueError):
            node.extend(model_bundle=bundle, video_settings=settings,
                        video_pipe=ctx, positive="test prompt")

    def test_extend_requires_bundle_attrs(self):
        """Should raise ValueError when bundle is missing model/clip/vae."""
        node = self.cls()
        bundle = UmeBundle(model=None, clip=None, vae=None)  # Missing
        settings = UmeVideoSettings()
        ctx = VideoGenerationContext()
        ctx.frames = MagicMock()  # Has frames but bundle is bad
        ctx.frames.shape = (49, 480, 768, 3)

        with self.assertRaises(ValueError):
            node.extend(model_bundle=bundle, video_settings=settings,
                        video_pipe=ctx, positive="test")


class TestLTXVideoExtenderTooltips(unittest.TestCase):
    """Every input should have a tooltip."""

    def test_all_required_inputs_have_tooltips(self):
        from modules.block_nodes import UmeAiRT_LTXVideoExtender
        inputs = UmeAiRT_LTXVideoExtender.INPUT_TYPES()
        for key, spec in inputs["required"].items():
            if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                self.assertIn("tooltip", spec[1],
                              f"Missing tooltip for '{key}' in LTXVideoExtender.required")

    def test_all_optional_inputs_have_tooltips(self):
        from modules.block_nodes import UmeAiRT_LTXVideoExtender
        inputs = UmeAiRT_LTXVideoExtender.INPUT_TYPES()
        for key, spec in inputs["optional"].items():
            if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                self.assertIn("tooltip", spec[1],
                              f"Missing tooltip for '{key}' in LTXVideoExtender.optional")


if __name__ == "__main__":
    unittest.main()
