"""
UmeAiRT Toolkit - Unit Tests for LTX Prompt Director
-------------------------------------------------------
Tests for Prompt Segment chaining and Prompt Director node definition.
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


class TestPromptSegmentDefinition(unittest.TestCase):
    """Tests for UmeAiRT_PromptSegment node definition."""

    def setUp(self):
        from modules.ltx_prompt_director import UmeAiRT_PromptSegment
        self.cls = UmeAiRT_PromptSegment

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("start_time", required)
        self.assertIn("prompt", required)

    def test_optional_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        optional = inputs["optional"]
        self.assertIn("previous", optional)

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_PROMPT_SCHEDULE",))
        self.assertEqual(self.cls.RETURN_NAMES, ("schedule",))
        self.assertEqual(self.cls.FUNCTION, "build")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Video")


class TestPromptSegmentChaining(unittest.TestCase):
    """Tests for chaining Prompt Segments."""

    def setUp(self):
        from modules.ltx_prompt_director import UmeAiRT_PromptSegment
        self.segment = UmeAiRT_PromptSegment()

    def test_single_segment(self):
        """Single segment should produce list with one entry."""
        result = self.segment.build(start_time=0.0, prompt="A forest scene")
        schedule = result[0]
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]["start_time"], 0.0)
        self.assertEqual(schedule[0]["prompt"], "A forest scene")

    def test_chain_two_segments(self):
        """Two chained segments should produce sorted list."""
        result1 = self.segment.build(start_time=0.0, prompt="A forest scene")
        result2 = self.segment.build(start_time=3.0, prompt="Walking through trees", previous=result1[0])
        schedule = result2[0]
        self.assertEqual(len(schedule), 2)
        self.assertEqual(schedule[0]["prompt"], "A forest scene")
        self.assertEqual(schedule[1]["prompt"], "Walking through trees")

    def test_chain_three_segments(self):
        """Three chained segments."""
        r1 = self.segment.build(start_time=0.0, prompt="Scene 1")
        r2 = self.segment.build(start_time=3.0, prompt="Scene 2", previous=r1[0])
        r3 = self.segment.build(start_time=6.0, prompt="Scene 3", previous=r2[0])
        schedule = r3[0]
        self.assertEqual(len(schedule), 3)

    def test_immutability(self):
        """Chaining should not modify previous schedule."""
        r1 = self.segment.build(start_time=0.0, prompt="Original")
        original_schedule = r1[0]
        original_len = len(original_schedule)
        self.segment.build(start_time=3.0, prompt="Added", previous=original_schedule)
        self.assertEqual(len(original_schedule), original_len)  # Should not be modified

    def test_empty_prompt_trimmed(self):
        """Empty prompt should be trimmed."""
        result = self.segment.build(start_time=0.0, prompt="  \n  ")
        schedule = result[0]
        self.assertEqual(schedule[0]["prompt"], "")


class TestLTXPromptDirectorDefinition(unittest.TestCase):
    """Tests for UmeAiRT_LTXPromptDirector node definition."""

    def setUp(self):
        from modules.ltx_prompt_director import UmeAiRT_LTXPromptDirector
        self.cls = UmeAiRT_LTXPromptDirector

    def test_input_types_structure(self):
        inputs = self.cls.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)

    def test_required_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        required = inputs["required"]
        self.assertIn("model_bundle", required)
        self.assertIn("video_settings", required)
        self.assertIn("schedule", required)

    def test_optional_inputs(self):
        inputs = self.cls.INPUT_TYPES()
        optional = inputs["optional"]
        self.assertIn("negative", optional)
        self.assertIn("loras", optional)
        self.assertIn("source_image", optional)

    def test_class_attributes(self):
        self.assertEqual(self.cls.RETURN_TYPES, ("UME_VIDEO_PIPELINE",))
        self.assertEqual(self.cls.RETURN_NAMES, ("video_pipe",))
        self.assertEqual(self.cls.FUNCTION, "generate")
        self.assertEqual(self.cls.CATEGORY, "UmeAiRT/Video")

    def test_has_description(self):
        self.assertTrue(hasattr(self.cls, 'DESCRIPTION'))
        self.assertIn("prompt", self.cls.DESCRIPTION.lower())


class TestPromptDirectorValidation(unittest.TestCase):

    def setUp(self):
        from modules.ltx_prompt_director import UmeAiRT_LTXPromptDirector
        self.director = UmeAiRT_LTXPromptDirector()

    def test_sort_schedule(self):
        """Schedule should be sorted by start_time."""
        schedule = [
            {"start_time": 6.0, "prompt": "C"},
            {"start_time": 0.0, "prompt": "A"},
            {"start_time": 3.0, "prompt": "B"},
        ]
        sorted_sched = self.director._sort_schedule(schedule)
        self.assertEqual(sorted_sched[0]["prompt"], "A")
        self.assertEqual(sorted_sched[1]["prompt"], "B")
        self.assertEqual(sorted_sched[2]["prompt"], "C")

    def test_empty_prompts_filtered(self):
        """Empty prompts should be filtered out."""
        schedule = [
            {"start_time": 0.0, "prompt": "Valid"},
            {"start_time": 3.0, "prompt": ""},
            {"start_time": 6.0, "prompt": "Also valid"},
        ]
        sorted_sched = self.director._sort_schedule(schedule)
        self.assertEqual(len(sorted_sched), 2)

    def test_all_empty_raises(self):
        """All empty prompts should raise ValueError."""
        schedule = [
            {"start_time": 0.0, "prompt": ""},
            {"start_time": 3.0, "prompt": ""},
        ]
        with self.assertRaises(ValueError):
            self.director._sort_schedule(schedule)

    def test_missing_bundle_raises(self):
        """Missing model/clip/vae should raise ValueError."""
        bundle = UmeBundle(model=None, clip=None, vae=None)
        settings = UmeVideoSettings()
        schedule = [{"start_time": 0.0, "prompt": "Test"}]
        with self.assertRaises(ValueError):
            self.director.generate(model_bundle=bundle, video_settings=settings, schedule=schedule)


class TestPromptDirectorTooltips(unittest.TestCase):
    def test_segment_inputs_have_tooltips(self):
        from modules.ltx_prompt_director import UmeAiRT_PromptSegment
        inputs = UmeAiRT_PromptSegment.INPUT_TYPES()
        for key, spec in inputs["required"].items():
            if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                self.assertIn("tooltip", spec[1],
                              f"Missing tooltip for '{key}' in PromptSegment.required")

    def test_director_inputs_have_tooltips(self):
        from modules.ltx_prompt_director import UmeAiRT_LTXPromptDirector
        inputs = UmeAiRT_LTXPromptDirector.INPUT_TYPES()
        for key, spec in inputs["required"].items():
            if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                self.assertIn("tooltip", spec[1],
                              f"Missing tooltip for '{key}' in LTXPromptDirector.required")


if __name__ == "__main__":
    unittest.main()
