"""Tests for modules/block_loaders.py — FilesSettings* and BundleLoader nodes."""
import unittest
from unittest.mock import patch, MagicMock
import sys
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.clip_vision'] = MagicMock()

from modules.block_loaders import (
    UmeAiRT_FilesSettings_Checkpoint,
    UmeAiRT_FilesSettings_FLUX,
    UmeAiRT_FilesSettings_LTX,
    UmeAiRT_BundleLoader,
)
from modules.common import UmeBundle


class TestFilesSettingsCheckpoint(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_FilesSettings_Checkpoint.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)
        self.assertIn("ckpt_name", inputs["required"])
        self.assertIn("vae_name", inputs["optional"])
        self.assertIn("clip_skip", inputs["optional"])

    def test_instantiation(self):
        node = UmeAiRT_FilesSettings_Checkpoint()
        self.assertIsNotNone(node)


class TestFilesSettingsFLUX(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_FilesSettings_FLUX.INPUT_TYPES()
        self.assertIn("required", inputs)


class TestFilesSettingsLTX(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_FilesSettings_LTX.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)

    def test_required_inputs(self):
        inputs = UmeAiRT_FilesSettings_LTX.INPUT_TYPES()
        required = inputs["required"]
        for key in ("diff_model", "clip_gemma", "clip_ltx", "video_vae", "audio_vae"):
            self.assertIn(key, required, f"Missing required input: {key}")

    def test_return_type(self):
        self.assertEqual(UmeAiRT_FilesSettings_LTX.RETURN_TYPES, ("UME_BUNDLE",))
        self.assertEqual(UmeAiRT_FilesSettings_LTX.FUNCTION, "load_ltx")


class TestBundleLoader(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_BundleLoader.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_instantiation(self):
        node = UmeAiRT_BundleLoader()
        self.assertIsNotNone(node)


if __name__ == "__main__":
    unittest.main()
