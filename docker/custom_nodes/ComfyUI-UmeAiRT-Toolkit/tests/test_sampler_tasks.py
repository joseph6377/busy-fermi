"""Tests for sampler_tasks module — image preparation and inpaint compositing."""
import sys
import os
import unittest
from unittest.mock import MagicMock

# Check if REAL torch is available (not the DummyTorch mock from run_tests.py)
try:
    import torch
    import torchvision
    HAS_TORCH = type(torch).__name__ == 'module'  # DummyTorch has type 'DummyTorch'
except ImportError:
    HAS_TORCH = False

# Mock ComfyUI-only dependencies
for mod in ["comfy", "comfy.sd", "comfy.utils", "comfy.model_management",
            "comfy.samplers", "comfy.sample", "node_helpers", "nodes", "folder_paths"]:
    sys.modules.setdefault(mod, MagicMock())

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDetectFluxFill(unittest.TestCase):
    """Tests for _detect_flux_fill — no torch needed."""

    def setUp(self):
        from modules.sampler_tasks import _detect_flux_fill
        self.fn = _detect_flux_fill

    def test_non_flux(self):
        bundle = MagicMock()
        bundle.loader_type = "checkpoint"
        self.assertFalse(self.fn(bundle))

    def test_flux_non_fill(self):
        bundle = MagicMock()
        bundle.loader_type = "flux"
        bundle.bundle_type = "standard"
        bundle.category = "image"
        self.assertFalse(self.fn(bundle))

    def test_flux_fill_by_bundle_type(self):
        bundle = MagicMock()
        bundle.loader_type = "flux"
        bundle.bundle_type = "image_inpaint"
        bundle.category = ""
        self.assertTrue(self.fn(bundle))

    def test_flux_fill_by_category(self):
        bundle = MagicMock()
        bundle.loader_type = "flux"
        bundle.bundle_type = ""
        bundle.category = "fill"
        self.assertTrue(self.fn(bundle))


class TestComputePadding(unittest.TestCase):
    """Tests for _compute_padding — no torch needed."""

    def setUp(self):
        from modules.sampler_tasks import _compute_padding
        self.fn = _compute_padding

    def test_center(self):
        s, e = self.fn(100, "center", "left", "right")
        self.assertEqual(s + e, 100)
        self.assertEqual(s, 50)

    def test_left(self):
        s, e = self.fn(100, "left", "left", "right")
        self.assertEqual(s, 0)
        self.assertEqual(e, 100)

    def test_right(self):
        s, e = self.fn(100, "right", "left", "right")
        self.assertEqual(s, 100)
        self.assertEqual(e, 0)

    def test_negative_clamped(self):
        s, e = self.fn(-50, "center", "left", "right")
        self.assertEqual(s + e, 0)

    def test_odd_total(self):
        s, e = self.fn(101, "center", "left", "right")
        self.assertEqual(s + e, 101)
        self.assertEqual(s, 50)
        self.assertEqual(e, 51)


class TestImagePrepResult(unittest.TestCase):
    """Tests for ImagePrepResult dataclass defaults — no torch needed."""

    def setUp(self):
        from modules.sampler_tasks import ImagePrepResult
        self.cls = ImagePrepResult

    def test_defaults(self):
        r = self.cls()
        self.assertIsNone(r.raw_image)
        self.assertIsNone(r.source_mask)
        self.assertEqual(r.mode_str, "txt2img")
        self.assertFalse(r.is_outpaint)
        self.assertEqual(r.denoise, 1.0)
        self.assertIsNone(r.flux_fill_info)


@unittest.skipUnless(HAS_TORCH, "Requires real torch")
class TestPrepareTxt2Img(unittest.TestCase):
    """Tests for prepare_txt2img."""

    def setUp(self):
        from modules.sampler_tasks import prepare_txt2img
        self.fn = prepare_txt2img

    def _make_model(self, channels):
        model = MagicMock()
        model.model.latent_format.latent_channels = channels
        return model

    def test_returns_latent_dict_sd(self):
        result = self.fn(1024, 1024, self._make_model(4), 1.0)
        self.assertIn("samples", result)
        self.assertEqual(result["samples"].shape, (1, 4, 128, 128))

    def test_flux_16_channels(self):
        result = self.fn(1024, 1024, self._make_model(16), 1.0)
        self.assertEqual(result["samples"].shape, (1, 16, 128, 128))


@unittest.skipUnless(HAS_TORCH, "Requires real torch")
class TestApplyMaskBlur(unittest.TestCase):
    """Tests for _apply_mask_blur."""

    def setUp(self):
        from modules.sampler_tasks import _apply_mask_blur
        self.fn = _apply_mask_blur

    def test_2d_mask(self):
        mask = torch.ones(64, 64)
        result = self.fn(mask, 3)
        self.assertEqual(result.shape, (64, 64))

    def test_even_kernel_corrected(self):
        mask = torch.ones(1, 1, 64, 64)
        result = self.fn(mask, 4)
        self.assertEqual(result.shape, (1, 1, 64, 64))


@unittest.skipUnless(HAS_TORCH, "Requires real torch")
class TestCompositeInpaint(unittest.TestCase):
    """Tests for composite_inpaint."""

    def setUp(self):
        from modules.sampler_tasks import composite_inpaint
        self.fn = composite_inpaint

    def test_preserves_unmasked_region(self):
        image_out = torch.ones(1, 64, 64, 3)
        raw_image = torch.zeros(1, 64, 64, 3)
        mask = torch.zeros(64, 64)
        result = self.fn(image_out, raw_image, mask)
        self.assertTrue(torch.allclose(result, raw_image, atol=1e-5))

    def test_replaces_masked_region(self):
        image_out = torch.ones(1, 64, 64, 3)
        raw_image = torch.zeros(1, 64, 64, 3)
        mask = torch.ones(64, 64)
        result = self.fn(image_out, raw_image, mask)
        self.assertTrue(torch.allclose(result, image_out, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
