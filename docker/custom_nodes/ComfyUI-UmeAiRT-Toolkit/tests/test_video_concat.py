"""
UmeAiRT Toolkit - Unit Tests for Video Concatenate
---------------------------------------------------
Tests for node definition, merging logic, auto-resizing, and audio concatenation/padding.
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
    HAS_TORCH = hasattr(torch, 'Tensor') and callable(getattr(torch, 'zeros', None)) and len(torch.FloatTensor([1.0])) == 1
except Exception:
    HAS_TORCH = False
    sys.modules['torch'] = MagicMock()
    sys.modules['torchvision'] = MagicMock()
    sys.modules['torchvision.transforms'] = MagicMock()
    sys.modules['torchvision.transforms.functional'] = MagicMock()
    import torch

from modules.common import VideoGenerationContext


class TestVideoConcatDefinition(unittest.TestCase):
    """Tests for UmeAiRT_VideoConcat node definition."""

    def setUp(self):
        from modules.video_slicer import UmeAiRT_VideoConcat
        self.cls = UmeAiRT_VideoConcat

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("video_pipe_1", required)
        self.assertIn("video_pipe_2", required)
        self.assertIn("fps_source", required)

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_VIDEO_PIPELINE",))
        self.assertEqual(self.cls.RETURN_NAMES, ("video_pipe",))
        self.assertEqual(self.cls.FUNCTION, "concat")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Post-Process")

    def test_has_description(self):
        self.assertTrue(hasattr(self.cls, 'DESCRIPTION'))
        self.assertIn("merge", self.cls.DESCRIPTION.lower())


@unittest.skipUnless(HAS_TORCH, "Requires real torch for tensor operations")
class TestVideoConcatLogic(unittest.TestCase):
    """Tests for concatenation logic."""

    def setUp(self):
        from modules.video_slicer import UmeAiRT_VideoConcat
        self.concat_node = UmeAiRT_VideoConcat()

    def _make_ctx(self, num_frames=50, fps=25, width=640, height=480):
        ctx = VideoGenerationContext()
        ctx.frames = torch.zeros(num_frames, height, width, 3)
        ctx.fps = fps
        ctx.frame_count = num_frames
        ctx.duration = num_frames / fps
        ctx.width = width
        ctx.height = height
        ctx.audio = None
        return ctx

    def test_basic_concat(self):
        """Standard concatenation of matching video settings, no audio."""
        ctx1 = self._make_ctx(50, 25)
        ctx2 = self._make_ctx(30, 25)
        
        result = self.concat_node.concat(ctx1, ctx2, fps_source="first_video")
        res_ctx = result[0]
        
        self.assertEqual(res_ctx.frames.shape[0], 80)
        self.assertEqual(res_ctx.frame_count, 80)
        self.assertEqual(res_ctx.fps, 25)
        self.assertAlmostEqual(res_ctx.duration, 3.2, places=2)
        self.assertIsNone(res_ctx.audio)

    def test_concat_different_resolutions(self):
        """Second video is resized to match the first video's resolution."""
        ctx1 = self._make_ctx(50, 25, width=640, height=480)
        ctx2 = self._make_ctx(30, 25, width=320, height=240)
        
        result = self.concat_node.concat(ctx1, ctx2, fps_source="first_video")
        res_ctx = result[0]
        
        self.assertEqual(res_ctx.frames.shape[0], 80)
        self.assertEqual(res_ctx.width, 640)
        self.assertEqual(res_ctx.height, 480)
        self.assertEqual(res_ctx.frames.shape[1], 480)
        self.assertEqual(res_ctx.frames.shape[2], 640)

    def test_concat_different_fps_first_source(self):
        """Test mismatched FPS using first video as source."""
        ctx1 = self._make_ctx(50, 25)  # 2s duration
        ctx2 = self._make_ctx(30, 30)  # 1s duration
        
        result = self.concat_node.concat(ctx1, ctx2, fps_source="first_video")
        res_ctx = result[0]
        
        self.assertEqual(res_ctx.fps, 25)
        self.assertEqual(res_ctx.frame_count, 80)
        self.assertAlmostEqual(res_ctx.duration, 3.2, places=2)

    def test_concat_different_fps_second_source(self):
        """Test mismatched FPS using second video as source."""
        ctx1 = self._make_ctx(50, 25)
        ctx2 = self._make_ctx(30, 30)
        
        result = self.concat_node.concat(ctx1, ctx2, fps_source="second_video")
        res_ctx = result[0]
        
        self.assertEqual(res_ctx.fps, 30)
        self.assertEqual(res_ctx.frame_count, 80)
        self.assertAlmostEqual(res_ctx.duration, 80/30, places=2)

    def test_concat_both_audio_mismatched_sample_rates(self):
        """Audio resampling is applied when sample rates mismatch."""
        ctx1 = self._make_ctx(50, 25)
        ctx1.audio = {
            "waveform": torch.zeros(1, 2, 44100 * 2),  # 2s stereo @ 44100Hz
            "sample_rate": 44100,
        }
        
        ctx2 = self._make_ctx(30, 30)  # 1s duration
        ctx2.audio = {
            "waveform": torch.zeros(1, 2, 48000 * 1),  # 1s stereo @ 48000Hz
            "sample_rate": 48000,
        }
        
        result = self.concat_node.concat(ctx1, ctx2, fps_source="first_video")
        res_ctx = result[0]
        
        self.assertIsNotNone(res_ctx.audio)
        self.assertEqual(res_ctx.audio["sample_rate"], 44100)
        # Expected total samples = 44100 * 2 (video 1) + 44100 * 1 (video 2 resampled) = 132300
        # Tolerating small rounding differences in size calculation
        self.assertAlmostEqual(res_ctx.audio["waveform"].shape[-1], 132300, delta=100)

    def test_concat_both_audio_mismatched_channels(self):
        """Mono audio is converted to stereo when matched with a stereo video."""
        ctx1 = self._make_ctx(50, 25)
        ctx1.audio = {
            "waveform": torch.zeros(1, 1, 44100 * 2),  # 2s mono
            "sample_rate": 44100,
        }
        
        ctx2 = self._make_ctx(25, 25)
        ctx2.audio = {
            "waveform": torch.zeros(1, 2, 44100 * 1),  # 1s stereo
            "sample_rate": 44100,
        }
        
        result = self.concat_node.concat(ctx1, ctx2, fps_source="first_video")
        res_ctx = result[0]
        
        self.assertIsNotNone(res_ctx.audio)
        self.assertEqual(res_ctx.audio["waveform"].shape[1], 2)  # Stereo output

    def test_concat_only_first_audio_silent_padding(self):
        """If only first video has audio, second video part gets padded with silence."""
        ctx1 = self._make_ctx(50, 25)
        ctx1.audio = {
            "waveform": torch.zeros(1, 2, 44100 * 2),  # 2s @ 44100Hz
            "sample_rate": 44100,
        }
        
        ctx2 = self._make_ctx(25, 25)  # 1s duration, no audio
        
        result = self.concat_node.concat(ctx1, ctx2, fps_source="first_video")
        res_ctx = result[0]
        
        self.assertIsNotNone(res_ctx.audio)
        self.assertEqual(res_ctx.audio["sample_rate"], 44100)
        # Expected total samples = 2s of audio + 1s of silence = 3s @ 44100Hz = 132300
        self.assertEqual(res_ctx.audio["waveform"].shape[-1], 132300)

    def test_concat_only_second_audio_silent_padding(self):
        """If only second video has audio, first video part gets padded with silence at start."""
        ctx1 = self._make_ctx(50, 25)  # 2s duration, no audio
        
        ctx2 = self._make_ctx(25, 25)  # 1s duration
        ctx2.audio = {
            "waveform": torch.zeros(1, 1, 16000 * 1),  # 1s @ 16000Hz mono
            "sample_rate": 16000,
        }
        
        result = self.concat_node.concat(ctx1, ctx2, fps_source="first_video")
        res_ctx = result[0]
        
        self.assertIsNotNone(res_ctx.audio)
        self.assertEqual(res_ctx.audio["sample_rate"], 16000)
        # Expected total samples = 2s silence + 1s audio = 3s @ 16000Hz = 48000
        self.assertEqual(res_ctx.audio["waveform"].shape[-1], 48000)


class TestVideoConcatTooltips(unittest.TestCase):
    def test_all_required_inputs_have_tooltips(self):
        from modules.video_slicer import UmeAiRT_VideoConcat
        inputs = UmeAiRT_VideoConcat.INPUT_TYPES()
        for key, spec in inputs["required"].items():
            if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                self.assertIn("tooltip", spec[1],
                              f"Missing tooltip for '{key}' in VideoConcat.required")


if __name__ == "__main__":
    unittest.main()
