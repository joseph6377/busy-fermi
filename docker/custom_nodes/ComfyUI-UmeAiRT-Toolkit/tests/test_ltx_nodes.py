"""
UmeAiRT Toolkit - Unit Tests for LTX-2.3 Nodes
-------------------------------------------------
Tests for LTXVideoSettings, LTXVideoGenerator, LTX Loader,
ltx_utils tiled decode, and VideoGenerationContext audio fields.
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
sys.modules['av'] = MagicMock()

# folder_paths mock — ComfyUI runtime module
mock_fp = MagicMock()
mock_fp.get_filename_list.return_value = []
mock_fp.get_folder_paths.return_value = []
mock_fp.get_full_path.return_value = None
sys.modules['folder_paths'] = mock_fp

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    # Verify we have real torch, not DummyTorch from run_tests.py
    HAS_TORCH = hasattr(torch, 'Tensor') and callable(getattr(torch, 'zeros', None)) and len(torch.FloatTensor([1.0])) == 1
except Exception:
    HAS_TORCH = False
    import torch  # Will use DummyTorch from run_tests.py
from modules.common import UmeBundle, UmeVideoSettings, VideoGenerationContext


# ──────────────────────────────────────────────────────────────────
# LTX Video Settings
# ──────────────────────────────────────────────────────────────────

class TestLTXVideoSettings(unittest.TestCase):
    """Tests for UmeAiRT_LTXVideoSettings node definition."""

    def setUp(self):
        from modules.block_inputs import UmeAiRT_LTXVideoSettings
        self.cls = UmeAiRT_LTXVideoSettings

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("width", required)
        self.assertIn("height", required)
        self.assertIn("duration", required)
        self.assertIn("frame_rate", required)
        self.assertIn("seed", required)

    def test_optional_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        optional = inputs["optional"]
        self.assertIn("audio_enabled", optional)
        self.assertIn("sigmas_preset", optional)
        self.assertIn("custom_sigmas", optional)

    def test_ltx_defaults(self):
        """LTX defaults: width=768, height=512, frame_rate=25, resolution step=32."""
        inputs = self.cls.INPUT_TYPES()
        self.assertEqual(inputs["required"]["width"][1]["default"], 768)
        self.assertEqual(inputs["required"]["height"][1]["default"], 512)
        self.assertEqual(inputs["required"]["frame_rate"][1]["default"], 25)
        self.assertEqual(inputs["required"]["width"][1]["step"], 32)

    def test_process_returns_settings(self):
        node = self.cls()
        result = node.process(width=768, height=512, duration=5.0, frame_rate=25, seed=42)
        settings = result[0]
        self.assertIsInstance(settings, UmeVideoSettings)
        self.assertEqual(settings.width, 768)
        self.assertEqual(settings.height, 512)
        self.assertEqual(settings.duration, 5.0)
        self.assertEqual(settings.frame_rate, 25)
        self.assertEqual(settings.seed, 42)
        self.assertTrue(settings.audio_enabled)  # default
        self.assertEqual(settings.sigmas_preset, "standard")  # default

    def test_process_custom_preset(self):
        node = self.cls()
        result = node.process(width=768, height=512, duration=3.0, frame_rate=25, seed=0,
                              sigmas_preset="Fast (4 steps)", audio_enabled=False)
        settings = result[0]
        self.assertEqual(settings.sigmas_preset, "fast")
        self.assertFalse(settings.audio_enabled)

    def test_sigmas_preset_custom(self):
        node = self.cls()
        result = node.process(width=512, height=512, duration=2.0, frame_rate=25, seed=0,
                              sigmas_preset="Custom", custom_sigmas="1.0, 0.5, 0.0")
        settings = result[0]
        self.assertEqual(settings.sigmas_preset, "custom")
        self.assertEqual(settings.custom_sigmas, "1.0, 0.5, 0.0")

    def test_return_types(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_VIDEO_SETTINGS",))
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Video")

    def test_all_inputs_have_tooltips(self):
        """Every input parameter should have a tooltip."""
        inputs = self.cls.INPUT_TYPES()
        for section_name in ("required", "optional"):
            for key, spec in inputs.get(section_name, {}).items():
                if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                    self.assertIn("tooltip", spec[1],
                                  f"Missing tooltip for '{key}' in LTXVideoSettings.{section_name}")


# ──────────────────────────────────────────────────────────────────
# LTX Loader
# ──────────────────────────────────────────────────────────────────

class TestLTXLoader(unittest.TestCase):
    """Tests for UmeAiRT_FilesSettings_LTX node definition."""

    def setUp(self):
        from modules.block_loaders import UmeAiRT_FilesSettings_LTX
        self.cls = UmeAiRT_FilesSettings_LTX

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("diff_model", required)
        self.assertIn("clip_gemma", required)
        self.assertIn("clip_ltx", required)
        self.assertIn("video_vae", required)
        self.assertIn("audio_vae", required)

    def test_optional_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        optional = inputs["optional"]
        self.assertIn("latent_upscale_model", optional)

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_BUNDLE",))
        self.assertEqual(self.cls.FUNCTION, "load_ltx")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Loaders")

    def test_has_description(self):
        self.assertTrue(hasattr(self.cls, 'DESCRIPTION'))
        self.assertIn("LTX", self.cls.DESCRIPTION)

    def test_all_inputs_have_tooltips(self):
        inputs = self.cls.INPUT_TYPES()
        for section_name in ("required", "optional"):
            for key, spec in inputs.get(section_name, {}).items():
                if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                    self.assertIn("tooltip", spec[1],
                                  f"Missing tooltip for '{key}' in LTXLoader.{section_name}")


# ──────────────────────────────────────────────────────────────────
# LTX Video Generator
# ──────────────────────────────────────────────────────────────────

class TestLTXVideoGenerator(unittest.TestCase):
    """Tests for UmeAiRT_LTXVideoGenerator (alias → unified VideoGenerator)."""

    def setUp(self):
        from modules.block_nodes import UmeAiRT_LTXVideoGenerator
        self.cls = UmeAiRT_LTXVideoGenerator

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

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_VIDEO_PIPELINE",))
        self.assertEqual(self.cls.FUNCTION, "process")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Video")

    def test_has_description(self):
        self.assertTrue(hasattr(self.cls, 'DESCRIPTION'))
        # Unified node description mentions 'video' (covers WAN + LTX)
        self.assertIn("video", self.cls.DESCRIPTION.lower())

    def test_instantiation(self):
        node = self.cls()
        self.assertIsNotNone(node)


# ──────────────────────────────────────────────────────────────────
# Sigma Presets
# ──────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_TORCH, "Requires real torch for tensor operations")
class TestSigmaPresets(unittest.TestCase):
    """Tests for ManualSigmas preset parsing."""

    def setUp(self):
        from modules.ltx_sampler import _parse_sigmas, SIGMA_PRESETS
        self._parse_sigmas = _parse_sigmas
        self.SIGMA_PRESETS = SIGMA_PRESETS

    def test_standard_pass1_has_9_values(self):
        sigmas = self._parse_sigmas("standard", "pass1")
        self.assertEqual(len(sigmas), 9)
        self.assertAlmostEqual(sigmas[0].item(), 1.0)
        self.assertAlmostEqual(sigmas[-1].item(), 0.0)

    def test_standard_pass2_has_4_values(self):
        sigmas = self._parse_sigmas("standard", "pass2")
        self.assertEqual(len(sigmas), 4)

    def test_fast_preset(self):
        sigmas = self._parse_sigmas("fast", "pass1")
        self.assertEqual(len(sigmas), 4)

    def test_custom_sigmas_parsing(self):
        sigmas = self._parse_sigmas("custom", "pass1", "1.0, 0.5, 0.0")
        self.assertEqual(len(sigmas), 3)
        self.assertAlmostEqual(sigmas[0].item(), 1.0)
        self.assertAlmostEqual(sigmas[1].item(), 0.5)
        self.assertAlmostEqual(sigmas[2].item(), 0.0)

    def test_custom_sigmas_invalid_falls_back(self):
        sigmas = self._parse_sigmas("custom", "pass1", "not,valid,numbers")
        self.assertEqual(len(sigmas), 9)  # Falls back to standard pass1

    def test_custom_sigmas_empty_falls_back(self):
        sigmas = self._parse_sigmas("custom", "pass1", "")
        self.assertEqual(len(sigmas), 9)  # Falls back to standard

    def test_unknown_preset_falls_back(self):
        sigmas = self._parse_sigmas("nonexistent", "pass1")
        self.assertEqual(len(sigmas), 9)  # Falls back to standard


# ──────────────────────────────────────────────────────────────────
# UmeBundle LTX Fields
# ──────────────────────────────────────────────────────────────────

class TestUmeBundleLTXFields(unittest.TestCase):
    """Tests for new LTX-2.3 fields on UmeBundle."""

    def test_audio_vae_default_none(self):
        bundle = UmeBundle(model=None, clip=None, vae=None)
        self.assertIsNone(bundle.audio_vae)

    def test_latent_upscale_default_none(self):
        bundle = UmeBundle(model=None, clip=None, vae=None)
        self.assertIsNone(bundle.latent_upscale_model)

    def test_audio_vae_assignment(self):
        mock_vae = MagicMock()
        bundle = UmeBundle(model=None, clip=None, vae=None, audio_vae=mock_vae)
        self.assertIs(bundle.audio_vae, mock_vae)

    def test_latent_upscale_assignment(self):
        mock_up = MagicMock()
        bundle = UmeBundle(model=None, clip=None, vae=None, latent_upscale_model=mock_up)
        self.assertIs(bundle.latent_upscale_model, mock_up)


# ──────────────────────────────────────────────────────────────────
# UmeVideoSettings LTX Fields
# ──────────────────────────────────────────────────────────────────

class TestUmeVideoSettingsLTXFields(unittest.TestCase):
    """Tests for new LTX-2.3 fields on UmeVideoSettings."""

    def test_defaults(self):
        vs = UmeVideoSettings()
        self.assertEqual(vs.frame_rate, 16)  # Default (WAN)
        self.assertFalse(vs.audio_enabled)   # Default
        self.assertEqual(vs.sigmas_preset, "")
        self.assertEqual(vs.custom_sigmas, "")

    def test_ltx_values(self):
        vs = UmeVideoSettings(frame_rate=25, audio_enabled=True, sigmas_preset="standard")
        self.assertEqual(vs.frame_rate, 25)
        self.assertTrue(vs.audio_enabled)
        self.assertEqual(vs.sigmas_preset, "standard")


# ──────────────────────────────────────────────────────────────────
# VideoGenerationContext Audio Fields
# ──────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_TORCH, "Requires real torch for tensor operations")
class TestVideoContextAudioFields(unittest.TestCase):
    """Tests for audio-related fields on VideoGenerationContext."""

    def test_audio_default_none(self):
        ctx = VideoGenerationContext()
        self.assertIsNone(ctx.audio)
        self.assertIsNone(ctx.audio_vae)

    def test_audio_assignment(self):
        ctx = VideoGenerationContext()
        ctx.audio = {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}
        self.assertIsNotNone(ctx.audio)
        self.assertEqual(ctx.audio["sample_rate"], 16000)

    def test_clone_preserves_audio(self):
        ctx = VideoGenerationContext()
        ctx.audio = {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}
        ctx.audio_vae = MagicMock()
        cloned = ctx.clone()
        self.assertIs(cloned.audio, ctx.audio)  # shallow copy shares ref
        self.assertIs(cloned.audio_vae, ctx.audio_vae)


# ──────────────────────────────────────────────────────────────────
# LTX Utils — Tiled Decode Helpers
# ──────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_TORCH, "Requires real torch for tensor operations")
class TestLTXUtilsHelpers(unittest.TestCase):
    """Tests for internal helpers in ltx_utils."""

    def setUp(self):
        from modules.ltx_utils import _compute_chunk_boundaries, _calculate_temporal_output_boundaries
        self._compute_chunk_boundaries = _compute_chunk_boundaries
        self._calculate_temporal_output_boundaries = _calculate_temporal_output_boundaries

    def test_first_chunk_starts_at_zero(self):
        start, end = self._compute_chunk_boundaries(0, 16, 1, 32)
        self.assertEqual(start, 0)
        self.assertEqual(end, 16)

    def test_first_chunk_clamps_to_total(self):
        start, end = self._compute_chunk_boundaries(0, 16, 1, 10)
        self.assertEqual(start, 0)
        self.assertEqual(end, 10)

    def test_second_chunk_includes_overlap(self):
        start, end = self._compute_chunk_boundaries(16, 16, 2, 40)
        self.assertLess(start, 16)  # overlap_start < chunk_start
        self.assertGreater(end, start)

    def test_temporal_output_boundaries(self):
        t_start, t_end = self._calculate_temporal_output_boundaries(4, 8, 10)
        self.assertEqual(t_start, 33)  # 1 + 4*8
        self.assertEqual(t_end, 43)    # 33 + 10

    def test_tiled_decode_validation_error(self):
        """temporal_tile_length must be > temporal_overlap + 1."""
        from modules.ltx_utils import ltx_spatio_temporal_tiled_decode
        with self.assertRaises(ValueError):
            ltx_spatio_temporal_tiled_decode(
                vae=MagicMock(), latent_samples=torch.zeros(1, 128, 8, 4, 4),
                temporal_tile_length=1, temporal_overlap=1
            )


if __name__ == "__main__":
    unittest.main()
