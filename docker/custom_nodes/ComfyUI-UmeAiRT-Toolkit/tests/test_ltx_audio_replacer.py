"""
UmeAiRT Toolkit - Unit Tests for LTX Audio Replacer
------------------------------------------------------
Tests for node definition, mode validation, and waveform processing.
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
    # Verify we have real torch, not DummyTorch from run_tests.py
    HAS_TORCH = hasattr(torch, 'Tensor') and callable(getattr(torch, 'zeros', None)) and len(torch.FloatTensor([1.0])) == 1
except Exception:
    HAS_TORCH = False
    sys.modules['torch'] = MagicMock()
    sys.modules['torchvision'] = MagicMock()
    sys.modules['torchvision.transforms'] = MagicMock()
    sys.modules['torchvision.transforms.functional'] = MagicMock()
    import torch

from modules.common import UmeBundle, VideoGenerationContext


class TestLTXAudioReplacerDefinition(unittest.TestCase):
    """Tests for UmeAiRT_LTXAudioReplacer node definition."""

    def setUp(self):
        from modules.ltx_audio_replacer import UmeAiRT_LTXAudioReplacer
        self.cls = UmeAiRT_LTXAudioReplacer

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("video_pipe", required)
        self.assertIn("model_bundle", required)
        self.assertIn("mode", required)

    def test_optional_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        optional = inputs["optional"]
        self.assertIn("audio", optional)
        self.assertIn("positive", optional)
        self.assertIn("seed", optional)

    def test_mode_options(self):
        inputs = self.cls.INPUT_TYPES()
        mode_spec = inputs["required"]["mode"]
        self.assertIn("Replace from File", mode_spec[0])
        self.assertIn("Regenerate from Video", mode_spec[0])

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_VIDEO_PIPELINE",))
        self.assertEqual(self.cls.RETURN_NAMES, ("video_pipe",))
        self.assertEqual(self.cls.FUNCTION, "replace_audio")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Video")

    def test_has_description(self):
        self.assertTrue(hasattr(self.cls, 'DESCRIPTION'))
        self.assertIn("audio", self.cls.DESCRIPTION.lower())


@unittest.skipUnless(HAS_TORCH, "Requires real torch for tensor operations")
class TestLTXAudioReplacerValidation(unittest.TestCase):
    """Tests for input validation."""

    def setUp(self):
        from modules.ltx_audio_replacer import UmeAiRT_LTXAudioReplacer
        self.node = UmeAiRT_LTXAudioReplacer()

    def test_no_frames_raises(self):
        """Should raise ValueError when video_pipe has no frames."""
        ctx = VideoGenerationContext()
        ctx.frames = None
        bundle = UmeBundle(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
        with self.assertRaises(ValueError):
            self.node.replace_audio(video_pipe=ctx, model_bundle=bundle, mode="Replace from File")

    def test_replace_requires_audio_input(self):
        """Replace mode without audio should raise ValueError."""
        ctx = VideoGenerationContext()
        ctx.frames = torch.zeros(49, 480, 640, 3)
        ctx.fps = 25
        bundle = UmeBundle(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
        with self.assertRaises(ValueError):
            self.node.replace_audio(video_pipe=ctx, model_bundle=bundle,
                                    mode="Replace from File", audio=None)

    def test_replace_from_file_trims_audio(self):
        """Audio longer than video should be trimmed."""
        ctx = VideoGenerationContext()
        ctx.frames = torch.zeros(50, 480, 640, 3)  # 50 frames
        ctx.fps = 25  # 2s of video

        audio_input = {
            "waveform": torch.zeros(1, 1, 44100 * 5),  # 5s of audio
            "sample_rate": 44100,
        }
        bundle = UmeBundle(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
        result = self.node.replace_audio(
            video_pipe=ctx, model_bundle=bundle,
            mode="Replace from File", audio=audio_input,
        )
        # 2s * 44100 = 88200 samples
        self.assertEqual(result[0].audio["waveform"].shape[-1], 88200)

    def test_replace_from_file_pads_audio(self):
        """Audio shorter than video should be padded."""
        ctx = VideoGenerationContext()
        ctx.frames = torch.zeros(50, 480, 640, 3)  # 50 frames
        ctx.fps = 25  # 2s of video

        audio_input = {
            "waveform": torch.zeros(1, 1, 44100),  # 1s of audio
            "sample_rate": 44100,
        }
        bundle = UmeBundle(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
        result = self.node.replace_audio(
            video_pipe=ctx, model_bundle=bundle,
            mode="Replace from File", audio=audio_input,
        )
        # 2s * 44100 = 88200 samples (padded)
        self.assertEqual(result[0].audio["waveform"].shape[-1], 88200)


class TestLTXAudioReplacerTooltips(unittest.TestCase):
    def test_all_required_inputs_have_tooltips(self):
        from modules.ltx_audio_replacer import UmeAiRT_LTXAudioReplacer
        inputs = UmeAiRT_LTXAudioReplacer.INPUT_TYPES()
        for key, spec in inputs["required"].items():
            if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                self.assertIn("tooltip", spec[1],
                              f"Missing tooltip for '{key}' in LTXAudioReplacer.required")


if __name__ == "__main__":
    unittest.main()
