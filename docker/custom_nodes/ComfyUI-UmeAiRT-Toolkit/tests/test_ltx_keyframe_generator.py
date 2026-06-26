"""
UmeAiRT Toolkit - Unit Tests for LTX Keyframe Generator
----------------------------------------------------------
Tests for node definition, keyframe index calculation, and validation.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.samplers'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
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

mock_fp = MagicMock()
mock_fp.get_filename_list.return_value = []
mock_fp.get_folder_paths.return_value = []
mock_fp.get_full_path.return_value = None
sys.modules['folder_paths'] = mock_fp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
except ImportError:
    sys.modules['torch'] = MagicMock()
    sys.modules['torchvision'] = MagicMock()
    sys.modules['torchvision.transforms'] = MagicMock()
    sys.modules['torchvision.transforms.functional'] = MagicMock()
    import torch

from modules.common import UmeBundle, UmeVideoSettings


class TestLTXKeyframeGeneratorDefinition(unittest.TestCase):
    """Tests for UmeAiRT_LTXKeyframeGenerator node definition."""

    def setUp(self):
        from modules.ltx_keyframe_generator import UmeAiRT_LTXKeyframeGenerator
        self.cls = UmeAiRT_LTXKeyframeGenerator

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("model_bundle", required)
        self.assertIn("positive", required)
        self.assertIn("video_settings", required)
        self.assertIn("first_frame", required)
        self.assertIn("last_frame", required)

    def test_optional_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        optional = inputs["optional"]
        self.assertIn("middle_frame", optional)
        self.assertIn("negative", optional)
        self.assertIn("loras", optional)
        self.assertIn("cond_image_strength", optional)

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_VIDEO_PIPELINE",))
        self.assertEqual(self.cls.RETURN_NAMES, ("video_pipe",))
        self.assertEqual(self.cls.FUNCTION, "generate")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Video")

    def test_has_description(self):
        self.assertTrue(hasattr(self.cls, 'DESCRIPTION'))
        self.assertIn("keyframe", self.cls.DESCRIPTION.lower())

    def test_no_keyframe_mode_dropdown(self):
        """No mode dropdown — auto-detect from middle_frame connection."""
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertNotIn("keyframe_mode", required)

    def test_delegates_to_base_sampler(self):
        """Node should NOT create manual latents — delegates to BaseSampler."""
        import inspect
        from modules.ltx_keyframe_generator import UmeAiRT_LTXKeyframeGenerator
        source = inspect.getsource(UmeAiRT_LTXKeyframeGenerator.generate)
        # Should use BaseSampler, not create manual latents
        self.assertIn("LTXVBaseSampler", source)
        # Should NOT have NestedTensor (AV combining is handled by BaseSampler)
        self.assertNotIn("NestedTensor", source)


class TestLTXKeyframeGeneratorValidation(unittest.TestCase):

    def setUp(self):
        from modules.ltx_keyframe_generator import UmeAiRT_LTXKeyframeGenerator
        self.node = UmeAiRT_LTXKeyframeGenerator()

    def test_missing_first_frame_raises(self):
        """Should raise ValueError when first_frame is None."""
        bundle = UmeBundle(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
        settings = UmeVideoSettings(width=768, height=512, frame_rate=25, seed=42)
        with self.assertRaises(ValueError):
            self.node.generate(model_bundle=bundle, positive="test",
                               video_settings=settings, first_frame=None, last_frame=MagicMock())

    def test_missing_last_frame_raises(self):
        """Should raise ValueError when last_frame is None."""
        bundle = UmeBundle(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
        settings = UmeVideoSettings(width=768, height=512, frame_rate=25, seed=42)
        with self.assertRaises(ValueError):
            self.node.generate(model_bundle=bundle, positive="test",
                               video_settings=settings, first_frame=MagicMock(), last_frame=None)

    def test_missing_bundle_attrs_raises(self):
        """Should raise ValueError when bundle is missing model/clip/vae."""
        bundle = UmeBundle(model=None, clip=None, vae=None)
        settings = UmeVideoSettings()
        with self.assertRaises(ValueError):
            self.node.generate(model_bundle=bundle, positive="test",
                               video_settings=settings,
                               first_frame=MagicMock(), last_frame=MagicMock())


class TestLTXKeyframeGeneratorTooltips(unittest.TestCase):
    def test_all_required_inputs_have_tooltips(self):
        from modules.ltx_keyframe_generator import UmeAiRT_LTXKeyframeGenerator
        inputs = UmeAiRT_LTXKeyframeGenerator.INPUT_TYPES()
        for key, spec in inputs["required"].items():
            if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                self.assertIn("tooltip", spec[1],
                              f"Missing tooltip for '{key}' in LTXKeyframeGenerator.required")


if __name__ == "__main__":
    unittest.main()
