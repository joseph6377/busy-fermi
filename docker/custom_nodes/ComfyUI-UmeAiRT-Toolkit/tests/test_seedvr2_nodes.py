"""Tests for modules/seedvr2_nodes.py — SeedVR2 Upscale pipeline node."""
import unittest

from modules.seedvr2_nodes import UmeAiRT_PipelineSeedVR2Upscale


class TestPipelineSeedVR2Upscale(unittest.TestCase):
    def test_input_types_required(self):
        inputs = UmeAiRT_PipelineSeedVR2Upscale.INPUT_TYPES()
        self.assertIn("required", inputs)
        req = inputs["required"]
        self.assertIn("gen_pipe", req)
        self.assertIn("enabled", req)
        self.assertIn("model", req)
        self.assertIn("upscale_by", req)

    def test_input_types_has_tiling_params(self):
        inputs = UmeAiRT_PipelineSeedVR2Upscale.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("tile_width", req)
        self.assertIn("tile_height", req)
        self.assertIn("tiling_strategy", req)
        self.assertIn("blending_method", req)
        self.assertIn("color_correction", req)

    def test_return_types(self):
        self.assertEqual(UmeAiRT_PipelineSeedVR2Upscale.RETURN_TYPES, ("UME_PIPELINE",))

    def test_function_name(self):
        self.assertEqual(UmeAiRT_PipelineSeedVR2Upscale.FUNCTION, "upscale")

    def test_category(self):
        self.assertEqual(UmeAiRT_PipelineSeedVR2Upscale.CATEGORY, "UmeAiRT/Post-Process")

    def test_has_upscale_method(self):
        node = UmeAiRT_PipelineSeedVR2Upscale()
        self.assertTrue(callable(node.upscale))

    def test_has_build_configs(self):
        self.assertTrue(hasattr(UmeAiRT_PipelineSeedVR2Upscale, "_build_configs"))
        self.assertTrue(callable(UmeAiRT_PipelineSeedVR2Upscale._build_configs))

    def test_build_configs_device_not_hardcoded(self):
        """_build_configs should NOT hardcode 'cuda:0' — must use mm.get_torch_device()."""
        dit_config, vae_config = UmeAiRT_PipelineSeedVR2Upscale._build_configs("test_model.safetensors")
        # Regression: the old code hardcoded "cuda:0" or "cpu"
        self.assertNotEqual(dit_config["device"], "cuda:0")
        self.assertNotEqual(dit_config["device"], "cpu")
        self.assertEqual(dit_config["device"], vae_config["device"])

    def test_build_configs_offload_coherent(self):
        """offload_device should be 'none' on MPS, 'cpu' otherwise."""
        import torch
        dit_config, vae_config = UmeAiRT_PipelineSeedVR2Upscale._build_configs("test_model.safetensors")
        is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        expected_offload = "none" if is_mps else "cpu"
        self.assertEqual(dit_config["offload_device"], expected_offload)
        self.assertEqual(vae_config["offload_device"], expected_offload)


if __name__ == "__main__":
    unittest.main()
