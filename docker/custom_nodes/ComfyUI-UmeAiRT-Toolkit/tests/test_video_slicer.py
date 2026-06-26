"""
UmeAiRT Toolkit - Unit Tests for Video Slicer
-------------------------------------------------
Tests for node definition, slicing logic, and audio trimming.
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

from modules.common import VideoGenerationContext


class TestVideoSlicerDefinition(unittest.TestCase):
    """Tests for UmeAiRT_VideoSlicer node definition."""

    def setUp(self):
        from modules.video_slicer import UmeAiRT_VideoSlicer
        self.cls = UmeAiRT_VideoSlicer

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("video_pipe", required)
        self.assertIn("start_time", required)
        self.assertIn("end_time", required)

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_VIDEO_PIPELINE",))
        self.assertEqual(self.cls.RETURN_NAMES, ("video_pipe",))
        self.assertEqual(self.cls.FUNCTION, "slice")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Post-Process")

    def test_has_description(self):
        self.assertTrue(hasattr(self.cls, 'DESCRIPTION'))
        self.assertIn("trim", self.cls.DESCRIPTION.lower())


@unittest.skipUnless(HAS_TORCH, "Requires real torch for tensor operations")
class TestVideoSlicerLogic(unittest.TestCase):
    """Tests for slicing logic."""

    def setUp(self):
        from modules.video_slicer import UmeAiRT_VideoSlicer
        self.slicer = UmeAiRT_VideoSlicer()

    def _make_ctx(self, num_frames=100, fps=25):
        ctx = VideoGenerationContext()
        ctx.frames = torch.zeros(num_frames, 480, 640, 3)
        ctx.fps = fps
        ctx.frame_count = num_frames
        ctx.duration = num_frames / fps
        return ctx

    def test_full_video_passthrough(self):
        """start=0, end=-1 should return all frames."""
        ctx = self._make_ctx(100, 25)
        result = self.slicer.slice(ctx, start_time=0.0, end_time=-1.0)
        self.assertEqual(result[0].frames.shape[0], 100)

    def test_trim_first_second(self):
        """Trim first 1s from 25fps video = remove 25 frames."""
        ctx = self._make_ctx(100, 25)
        result = self.slicer.slice(ctx, start_time=1.0, end_time=-1.0)
        self.assertEqual(result[0].frames.shape[0], 75)

    def test_trim_last_second(self):
        """Keep only first 3s from 4s video."""
        ctx = self._make_ctx(100, 25)
        result = self.slicer.slice(ctx, start_time=0.0, end_time=3.0)
        self.assertEqual(result[0].frames.shape[0], 75)

    def test_extract_middle(self):
        """Extract 1s-3s from 4s video."""
        ctx = self._make_ctx(100, 25)
        result = self.slicer.slice(ctx, start_time=1.0, end_time=3.0)
        self.assertEqual(result[0].frames.shape[0], 50)

    def test_invalid_range_raises(self):
        """start >= end should raise ValueError."""
        ctx = self._make_ctx(100, 25)
        with self.assertRaises(ValueError):
            self.slicer.slice(ctx, start_time=3.0, end_time=1.0)

    def test_start_beyond_duration_raises(self):
        """start beyond video duration should raise ValueError."""
        ctx = self._make_ctx(100, 25)
        with self.assertRaises(ValueError):
            self.slicer.slice(ctx, start_time=10.0, end_time=-1.0)

    def test_no_frames_raises(self):
        """No frames should raise ValueError."""
        ctx = VideoGenerationContext()
        ctx.frames = None
        with self.assertRaises(ValueError):
            self.slicer.slice(ctx, start_time=0.0, end_time=-1.0)

    def test_audio_trimmed(self):
        """Audio should be trimmed proportionally."""
        ctx = self._make_ctx(100, 25)
        ctx.audio = {
            "waveform": torch.zeros(1, 1, 44100 * 4),  # 4s @ 44100Hz
            "sample_rate": 44100,
        }
        result = self.slicer.slice(ctx, start_time=1.0, end_time=3.0)
        # 2s of audio = 2 * 44100 = 88200 samples
        self.assertEqual(result[0].audio["waveform"].shape[-1], 88200)

    def test_duration_updated(self):
        """ctx.duration and frame_count should be updated."""
        ctx = self._make_ctx(100, 25)
        result = self.slicer.slice(ctx, start_time=1.0, end_time=3.0)
        self.assertEqual(result[0].frame_count, 50)
        self.assertAlmostEqual(result[0].duration, 2.0, places=1)


class TestVideoSlicerTooltips(unittest.TestCase):
    def test_all_required_inputs_have_tooltips(self):
        from modules.video_slicer import UmeAiRT_VideoSlicer
        inputs = UmeAiRT_VideoSlicer.INPUT_TYPES()
        for key, spec in inputs["required"].items():
            if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                self.assertIn("tooltip", spec[1],
                              f"Missing tooltip for '{key}' in VideoSlicer.required")


if __name__ == "__main__":
    unittest.main()
