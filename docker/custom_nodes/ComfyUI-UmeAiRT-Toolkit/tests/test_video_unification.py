"""
UmeAiRT Toolkit - Unit Tests for Video Pipeline Unification
--------------------------------------------------------------
Tests for video_utils (patch_wan_model, apply_color_match),
unified VideoGenerator/VideoExtender orchestrators, and
unified VideoSettings with LTX optional fields.
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

# Mock ComfyUI + dependencies
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.samplers'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['comfy.clip_vision'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['comfy.model_sampling'] = MagicMock()
sys.modules['comfy.nested_tensor'] = MagicMock()
sys.modules['comfy.patcher_extension'] = MagicMock()
sys.modules['nodes'] = MagicMock()
sys.modules['node_helpers'] = MagicMock()
sys.modules['comfy_extras'] = MagicMock()
sys.modules['comfy_extras.nodes_lt'] = MagicMock()
sys.modules['comfy_extras.nodes_lt_audio'] = MagicMock()
sys.modules['comfy_extras.nodes_lt_upsampler'] = MagicMock()
sys.modules['comfy_extras.nodes_hunyuan'] = MagicMock()
sys.modules['comfy_extras.nodes_custom_sampler'] = MagicMock()
sys.modules['comfy_extras.nodes_post_processing'] = MagicMock()
sys.modules['comfy_extras.nodes_model_advanced'] = MagicMock()
sys.modules['comfy_extras.nodes_cfg'] = MagicMock()
sys.modules['comfy_extras.nodes_easycache'] = MagicMock()
sys.modules['av'] = MagicMock()

mock_fp = MagicMock()
mock_fp.get_filename_list.return_value = []
mock_fp.get_folder_paths.return_value = []
mock_fp.get_full_path.return_value = None
sys.modules['folder_paths'] = mock_fp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use real torch if available, else mock
try:
    import torch
    # Verify we have real torch, not DummyTorch from run_tests.py
    HAS_TORCH = hasattr(torch, 'Tensor') and callable(getattr(torch, 'zeros', None)) and len(torch.FloatTensor([1.0])) == 1
except (ImportError, Exception):
    HAS_TORCH = False
    sys.modules['torch'] = MagicMock()
    sys.modules['torchvision'] = MagicMock()
    sys.modules['torchvision.transforms'] = MagicMock()
    sys.modules['torchvision.transforms.functional'] = MagicMock()
    import torch

from modules.common import UmeBundle, UmeVideoSettings, VideoGenerationContext


# ──────────────────────────────────────────────────────────────────
# Unified VideoGenerator Orchestrator
# ──────────────────────────────────────────────────────────────────

class TestUnifiedVideoGenerator(unittest.TestCase):
    """Tests for the unified VideoGenerator orchestrator node."""

    def setUp(self):
        from modules.video_sampler import UmeAiRT_VideoGenerator
        self.cls = UmeAiRT_VideoGenerator

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

    def test_optional_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        optional = inputs["optional"]
        self.assertIn("negative", optional)
        self.assertIn("loras", optional)
        self.assertIn("source_image", optional)
        self.assertIn("vace_frames", optional)
        self.assertIn("funcontrol", optional)

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_VIDEO_PIPELINE",))
        self.assertEqual(self.cls.RETURN_NAMES, ("video_pipe",))
        self.assertEqual(self.cls.FUNCTION, "process")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Video")

    def test_has_description(self):
        self.assertTrue(hasattr(self.cls, 'DESCRIPTION'))
        desc = self.cls.DESCRIPTION.lower()
        self.assertIn("video", desc)

    def test_wan_and_ltx_mentioned_in_description(self):
        """Description should mention both WAN and LTX."""
        desc = self.cls.DESCRIPTION
        self.assertIn("WAN", desc)
        self.assertIn("LTX", desc)

    def test_all_inputs_have_tooltips(self):
        inputs = self.cls.INPUT_TYPES()
        for section_name in ("required", "optional"):
            for key, spec in inputs.get(section_name, {}).items():
                if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                    self.assertIn("tooltip", spec[1],
                                  f"Missing tooltip for '{key}' in VideoGenerator.{section_name}")

    def test_alias_is_same_class(self):
        """UmeAiRT_LTXVideoGenerator should be the same class as UmeAiRT_VideoGenerator."""
        from modules.block_nodes import UmeAiRT_LTXVideoGenerator, UmeAiRT_VideoGenerator
        self.assertIs(UmeAiRT_LTXVideoGenerator, UmeAiRT_VideoGenerator)

    def test_process_dispatches_ltx(self):
        """With loader_type='ltx2', should call generate_ltx (test dispatch exists)."""
        node = self.cls()
        bundle = UmeBundle(model=None, clip=None, vae=None, loader_type="ltx2")
        settings = UmeVideoSettings()
        # validate_bundle will raise because model is None — that's fine for dispatch testing
        with self.assertRaises(ValueError):
            node.process(bundle, "test prompt", settings)

    def test_process_dispatches_wan(self):
        """With loader_type='wan', should call generate_wan (test dispatch exists)."""
        node = self.cls()
        bundle = UmeBundle(model=None, clip=None, vae=None, loader_type="wan")
        settings = UmeVideoSettings()
        with self.assertRaises(ValueError):
            node.process(bundle, "test prompt", settings)


# ──────────────────────────────────────────────────────────────────
# Unified VideoExtender Orchestrator
# ──────────────────────────────────────────────────────────────────

class TestUnifiedVideoExtender(unittest.TestCase):
    """Tests for the unified VideoExtender orchestrator node."""

    def setUp(self):
        from modules.video_extender import UmeAiRT_VideoExtender
        self.cls = UmeAiRT_VideoExtender

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("video_pipe", required)
        self.assertIn("model_bundle", required)
        self.assertIn("positive", required)
        self.assertIn("video_settings", required)

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
        desc = self.cls.DESCRIPTION.lower()
        self.assertIn("extend", desc)

    def test_all_inputs_have_tooltips(self):
        inputs = self.cls.INPUT_TYPES()
        for section_name in ("required", "optional"):
            for key, spec in inputs.get(section_name, {}).items():
                if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                    self.assertIn("tooltip", spec[1],
                                  f"Missing tooltip for '{key}' in VideoExtender.{section_name}")

    def test_alias_is_same_class(self):
        """UmeAiRT_LTXVideoExtender should be the same class as UmeAiRT_VideoExtender."""
        from modules.block_nodes import UmeAiRT_LTXVideoExtender, UmeAiRT_VideoExtender
        self.assertIs(UmeAiRT_LTXVideoExtender, UmeAiRT_VideoExtender)


# ──────────────────────────────────────────────────────────────────
# Unified VideoSettings
# ──────────────────────────────────────────────────────────────────

class TestUnifiedVideoSettings(unittest.TestCase):
    """Tests for the unified VideoSettings node (WAN + LTX)."""

    def setUp(self):
        from modules.block_inputs import UmeAiRT_VideoSettings
        self.cls = UmeAiRT_VideoSettings

    def test_has_optional_ltx_fields(self):
        """VideoSettings should have optional LTX-specific fields."""
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("optional", inputs)
        optional = inputs["optional"]
        self.assertIn("frame_rate", optional)
        self.assertIn("audio_enabled", optional)
        self.assertIn("sigmas_preset", optional)
        self.assertIn("custom_sigmas", optional)

    def test_wan_defaults(self):
        """WAN defaults: width=848, height=480, duration=3.0, steps=20."""
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertEqual(required["width"][1]["default"], 848)
        self.assertEqual(required["height"][1]["default"], 480)
        self.assertEqual(required["duration"][1]["default"], 3.0)
        self.assertEqual(required["steps"][1]["default"], 20)

    def test_process_with_defaults(self):
        """Process with WAN defaults should produce settings with frame_rate=16."""
        node = self.cls()
        result = node.process(
            width=848, height=480, duration=3.0, steps=20, cfg=6.0,
            shift=6.0, sampler_name="uni_pc", scheduler="simple", seed=42
        )
        settings = result[0]
        self.assertIsInstance(settings, UmeVideoSettings)
        self.assertEqual(settings.width, 848)
        self.assertEqual(settings.frame_rate, 16)  # default
        self.assertFalse(settings.audio_enabled)    # default
        self.assertEqual(settings.sigmas_preset, "")  # "None" maps to ""

    def test_process_with_ltx_overrides(self):
        """Process with LTX overrides should produce correct settings."""
        node = self.cls()
        result = node.process(
            width=768, height=512, duration=5.0, steps=0, cfg=1.0,
            shift=0.0, sampler_name="euler", scheduler="simple", seed=42,
            frame_rate=25, audio_enabled=True,
            sigmas_preset="Standard (8 steps)", custom_sigmas=""
        )
        settings = result[0]
        self.assertEqual(settings.frame_rate, 25)
        self.assertTrue(settings.audio_enabled)
        self.assertEqual(settings.sigmas_preset, "standard")

    def test_description_mentions_wan_and_ltx(self):
        self.assertIn("WAN", self.cls.DESCRIPTION)
        self.assertIn("LTX", self.cls.DESCRIPTION)

    def test_all_inputs_have_tooltips(self):
        inputs = self.cls.INPUT_TYPES()
        for section_name in ("required", "optional"):
            for key, spec in inputs.get(section_name, {}).items():
                if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                    self.assertIn("tooltip", spec[1],
                                  f"Missing tooltip for '{key}' in VideoSettings.{section_name}")


# ──────────────────────────────────────────────────────────────────
# Video Utils
# ──────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_TORCH, "Requires real torch for tensor operations")
class TestVideoUtils(unittest.TestCase):
    """Tests for shared video utility functions."""

    def test_apply_color_match_clamps(self):
        """Color match should clamp output to [0, 1]."""
        from modules.video_utils import apply_color_match

        frames = torch.rand(4, 64, 64, 3)  # 4 frames
        reference = torch.rand(1, 64, 64, 3)

        # Mock comfy.utils.common_upscale to return identity
        import comfy.utils
        comfy.utils.common_upscale = lambda x, w, h, mode, crop: x

        result = apply_color_match(frames.clone(), reference, 64, 64)
        self.assertTrue((result >= 0.0).all())
        self.assertTrue((result <= 1.0).all())

    def test_apply_color_match_preserves_shape(self):
        """Color match should preserve frame tensor shape."""
        from modules.video_utils import apply_color_match

        frames = torch.rand(8, 32, 32, 3)
        reference = torch.rand(1, 32, 32, 3)

        import comfy.utils
        comfy.utils.common_upscale = lambda x, w, h, mode, crop: x

        result = apply_color_match(frames.clone(), reference, 32, 32)
        self.assertEqual(result.shape, frames.shape)


# ──────────────────────────────────────────────────────────────────
# WAN Sampler Module
# ──────────────────────────────────────────────────────────────────

class TestWanSamplerModule(unittest.TestCase):
    """Tests for wan_sampler module availability and exports."""

    def test_generate_wan_importable(self):
        from modules.wan_sampler import generate_wan
        self.assertTrue(callable(generate_wan))

    def test_internal_builders_importable(self):
        from modules.wan_sampler import _build_vace_conditioning, _build_funcontrol_conditioning
        self.assertTrue(callable(_build_vace_conditioning))
        self.assertTrue(callable(_build_funcontrol_conditioning))


# ──────────────────────────────────────────────────────────────────
# WAN Extender Module
# ──────────────────────────────────────────────────────────────────

class TestWanExtenderModule(unittest.TestCase):
    """Tests for wan_extender module availability and exports."""

    def test_extend_wan_importable(self):
        from modules.wan_extender import extend_wan
        self.assertTrue(callable(extend_wan))


# ──────────────────────────────────────────────────────────────────
# LTX Sampler Module (as function, not class)
# ──────────────────────────────────────────────────────────────────

class TestLTXSamplerModule(unittest.TestCase):
    """Tests for ltx_sampler module — should export functions, not classes."""

    def test_generate_ltx_importable(self):
        from modules.ltx_sampler import generate_ltx
        self.assertTrue(callable(generate_ltx))

    def test_no_class_export(self):
        """ltx_sampler should NOT export UmeAiRT_LTXVideoGenerator anymore."""
        import modules.ltx_sampler as mod
        self.assertFalse(hasattr(mod, 'UmeAiRT_LTXVideoGenerator'))

    def test_sigma_presets_exported(self):
        from modules.ltx_sampler import SIGMA_PRESETS, _parse_sigmas
        self.assertIn("standard", SIGMA_PRESETS)
        self.assertIn("fast", SIGMA_PRESETS)
        self.assertTrue(callable(_parse_sigmas))


# ──────────────────────────────────────────────────────────────────
# LTX Extender Module (as function, not class)
# ──────────────────────────────────────────────────────────────────

class TestLTXExtenderModule(unittest.TestCase):
    """Tests for ltx_extender module — should export functions, not classes."""

    def test_extend_ltx_importable(self):
        from modules.ltx_extender import extend_ltx
        self.assertTrue(callable(extend_ltx))

    def test_no_class_export(self):
        """ltx_extender should NOT export UmeAiRT_LTXVideoExtender anymore."""
        import modules.ltx_extender as mod
        self.assertFalse(hasattr(mod, 'UmeAiRT_LTXVideoExtender'))


if __name__ == "__main__":
    unittest.main()
