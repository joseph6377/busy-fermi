"""Tests for sampler_cache module — PromptCache, _check_controlnets_equal, build_zero_cond."""
import sys
import os
import unittest
from unittest.mock import MagicMock

# Check if REAL torch is available (not the DummyTorch mock from run_tests.py)
try:
    import torch
    HAS_TORCH = type(torch).__name__ == 'module'  # DummyTorch has type 'DummyTorch'
except ImportError:
    HAS_TORCH = False

# Mock ComfyUI-only dependencies
for mod in ["comfy", "comfy.sd", "comfy.utils", "comfy.model_management",
            "comfy.samplers", "comfy.sample", "node_helpers", "nodes", "folder_paths"]:
    sys.modules.setdefault(mod, MagicMock())

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCheckControlnetsEqual(unittest.TestCase):
    """Tests for _check_controlnets_equal — works with or without real torch."""

    def setUp(self):
        from modules.sampler_cache import _check_controlnets_equal
        self.fn = _check_controlnets_equal

    def test_both_none(self):
        self.assertTrue(self.fn(None, None))

    def test_both_empty(self):
        self.assertTrue(self.fn([], []))

    def test_none_vs_empty(self):
        self.assertTrue(self.fn(None, []))

    def test_different_lengths(self):
        obj = object()
        self.assertFalse(self.fn(
            [("a", obj, 1.0)],
            [("a", obj, 1.0), ("b", obj, 0.5)]
        ))

    def test_same_identity(self):
        obj = object()
        stack = [("cnet", obj, 1.0, 0.0, 1.0)]
        self.assertTrue(self.fn(stack, stack))

    def test_different_object_identity(self):
        obj1 = object()
        obj2 = object()
        self.assertFalse(self.fn(
            [("cnet", obj1, 1.0, 0.0, 1.0)],
            [("cnet", obj2, 1.0, 0.0, 1.0)]
        ))

    def test_different_names(self):
        obj = object()
        self.assertFalse(self.fn(
            [("cnet_a", obj, 1.0)],
            [("cnet_b", obj, 1.0)]
        ))

    def test_different_params(self):
        obj = object()
        self.assertFalse(self.fn(
            [("cnet", obj, 1.0, 0.0, 1.0)],
            [("cnet", obj, 0.5, 0.0, 1.0)]
        ))


@unittest.skipUnless(HAS_TORCH, "Requires real torch (not DummyTorch mock)")
class TestBuildZeroCond(unittest.TestCase):
    """Tests for build_zero_cond — requires real torch for isinstance(x, torch.Tensor)."""

    def setUp(self):
        from modules.sampler_cache import build_zero_cond
        self.fn = build_zero_cond

    def test_zeros_out_tensors(self):
        pos = [[[torch.ones(2, 3), {"pooled_output": torch.ones(4)}]]]
        zero = self.fn(pos)
        self.assertTrue(torch.all(zero[0][0][0] == 0))
        self.assertTrue(torch.all(zero[0][0][1]["pooled_output"] == 0))

    def test_does_not_modify_original(self):
        pos = [[[torch.ones(2, 3), {"pooled_output": torch.ones(4)}]]]
        self.fn(pos)
        self.assertTrue(torch.all(pos[0][0][0] == 1))

    def test_preserves_shape(self):
        pos = [[[torch.randn(5, 10), {}]]]
        zero = self.fn(pos)
        self.assertEqual(zero[0][0][0].shape, (5, 10))


@unittest.skipUnless(HAS_TORCH, "Requires real torch (not DummyTorch mock)")
class TestPromptCache(unittest.TestCase):
    """Tests for PromptCache."""

    def setUp(self):
        from modules.sampler_cache import PromptCache
        self.cache = PromptCache()

    def test_initial_cache_miss(self):
        clip = MagicMock()
        result = self.cache.try_get_cached("hello", "", clip, None, None)
        self.assertIsNone(result)

    def test_cache_hit_after_update(self):
        clip = MagicMock()
        pos_cond = [[[torch.randn(2, 3), {}]]]
        neg_cond = [[[torch.randn(2, 3), {}]]]
        self.cache.update("hello", "bad", clip, None, None, pos_cond, neg_cond)
        result = self.cache.try_get_cached("hello", "bad", clip, None, None)
        self.assertIsNotNone(result)

    def test_cache_miss_on_text_change(self):
        clip = MagicMock()
        pos_cond = [[[torch.randn(2, 3), {}]]]
        neg_cond = [[[torch.randn(2, 3), {}]]]
        self.cache.update("hello", "bad", clip, None, None, pos_cond, neg_cond)
        result = self.cache.try_get_cached("different", "bad", clip, None, None)
        self.assertIsNone(result)

    def test_cache_miss_on_clip_change(self):
        clip1 = MagicMock()
        clip2 = MagicMock()
        pos_cond = [[[torch.randn(2, 3), {}]]]
        neg_cond = [[[torch.randn(2, 3), {}]]]
        self.cache.update("hello", "bad", clip1, None, None, pos_cond, neg_cond)
        result = self.cache.try_get_cached("hello", "bad", clip2, None, None)
        self.assertIsNone(result)

    def test_cache_returns_deep_copy(self):
        clip = MagicMock()
        pos_cond = [[[torch.ones(2, 3), {}]]]
        neg_cond = [[[torch.ones(2, 3), {}]]]
        self.cache.update("hello", "", clip, None, None, pos_cond, neg_cond)
        result1 = self.cache.try_get_cached("hello", "", clip, None, None)
        result1[0][0][0][0] *= 0
        result2 = self.cache.try_get_cached("hello", "", clip, None, None)
        self.assertTrue(torch.all(result2[0][0][0][0] == 1))

    def test_cache_miss_on_lora_change(self):
        clip = MagicMock()
        pos_cond = [[[torch.randn(2, 3), {}]]]
        neg_cond = [[[torch.randn(2, 3), {}]]]
        loras1 = [("lora_a", 1.0, 1.0)]
        loras2 = [("lora_b", 1.0, 1.0)]
        self.cache.update("hello", "", clip, loras1, None, pos_cond, neg_cond)
        result = self.cache.try_get_cached("hello", "", clip, loras2, None)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
