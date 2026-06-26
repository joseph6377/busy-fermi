"""
UmeAiRT Toolkit — ComfyUI Runtime Integration Tests
----------------------------------------------------
These tests run ONLY inside the UmeAiRT Docker container where
real ComfyUI + PyTorch are available. They are automatically
skipped in local/mock environments.

Container: registry.gitlab.com/umeairt-studio/comfyui-auto_installer-python:latest

Tested scenarios:
  Phase 1 (original):
  - Real node instantiation with ComfyUI runtime
  - INPUT_TYPES validation against ComfyUI type system
  - Custom sampler registration in comfy.samplers
  - folder_paths integration (bbox, models dirs)
  - Torch tensor operations (the 6 tests skipped in mock mode)

  Phase 2 (coverage expansion):
  - resize_tensor with real GPU tensors (bypasses DummyTorch skip)
  - Asymmetric outpaint padding edge cases
  - GPU VRAM detection and optimization library checks
  - SamplerContext context manager with real libraries
  - All 6 custom samplers individually verified in KSampler.SAMPLERS
  - SA-Solver math helpers with real tensors
  - extract_pipeline_params / validate_bundle runtime
  - GenerationContext clone independence with real tensors
  - Detail Daemon schedule generation (numpy) and sigma interpolation
  - NODE_DISPLAY_NAME_MAPPINGS completeness
"""
import os
import sys
import unittest

# Skip entire module if not in ComfyUI container
COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", "/app/ComfyUI")
IN_CONTAINER = os.path.isfile(os.path.join(COMFYUI_ROOT, "main.py"))

TOOLKIT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def setUpModule():
    """Add ComfyUI and Toolkit to sys.path for real imports."""
    if not IN_CONTAINER:
        return
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    if TOOLKIT_ROOT not in sys.path:
        sys.path.insert(0, TOOLKIT_ROOT)


# ═══════════════════════════════════════════════════════════════
# Phase 1 — Original Tests
# ═══════════════════════════════════════════════════════════════

@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealImport(unittest.TestCase):
    """Validate the Toolkit loads cleanly with REAL ComfyUI — zero mocks."""

    def test_import_folder_paths(self):
        import folder_paths
        self.assertTrue(hasattr(folder_paths, "folder_names_and_paths"))
        self.assertTrue(hasattr(folder_paths, "get_filename_list"))

    def test_import_comfy_core(self):
        import comfy.samplers
        import comfy.model_management
        self.assertTrue(hasattr(comfy.samplers, "KSampler"))

    def test_import_init_real(self):
        """Full __init__.py import with real runtime — THE key test."""
        custom_nodes_dir = os.path.dirname(TOOLKIT_ROOT)
        if custom_nodes_dir not in sys.path:
            sys.path.insert(0, custom_nodes_dir)

        toolkit_name = os.path.basename(TOOLKIT_ROOT)
        mod = __import__(toolkit_name)
        self.assertTrue(hasattr(mod, "NODE_CLASS_MAPPINGS"))
        self.assertGreaterEqual(len(mod.NODE_CLASS_MAPPINGS), 25)
        print(f"✅ {len(mod.NODE_CLASS_MAPPINGS)} nodes loaded with real ComfyUI")


@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealNodeValidation(unittest.TestCase):
    """Validate INPUT_TYPES format against ComfyUI expectations."""

    @classmethod
    def setUpClass(cls):
        custom_nodes_dir = os.path.dirname(TOOLKIT_ROOT)
        if custom_nodes_dir not in sys.path:
            sys.path.insert(0, custom_nodes_dir)
        toolkit_name = os.path.basename(TOOLKIT_ROOT)
        mod = __import__(toolkit_name)
        cls.mappings = mod.NODE_CLASS_MAPPINGS

    def test_all_input_types_have_required(self):
        for name, cls in self.mappings.items():
            if hasattr(cls, "INPUT_TYPES"):
                result = cls.INPUT_TYPES()
                self.assertIn("required", result,
                              f"{name}: INPUT_TYPES missing 'required'")
                self.assertIsInstance(result["required"], dict,
                                     f"{name}: 'required' must be dict")

    def test_all_return_types_are_tuples(self):
        for name, cls in self.mappings.items():
            if hasattr(cls, "RETURN_TYPES"):
                self.assertIsInstance(cls.RETURN_TYPES, tuple,
                                     f"{name}: RETURN_TYPES must be tuple, "
                                     f"got {type(cls.RETURN_TYPES)}")

    def test_all_have_function(self):
        for name, cls in self.mappings.items():
            self.assertTrue(hasattr(cls, "FUNCTION"),
                            f"{name}: missing FUNCTION attribute")
            func_name = cls.FUNCTION
            self.assertTrue(hasattr(cls, func_name),
                            f"{name}: FUNCTION='{func_name}' but method doesn't exist")

    def test_all_have_category(self):
        for name, cls in self.mappings.items():
            self.assertTrue(hasattr(cls, "CATEGORY"),
                            f"{name}: missing CATEGORY attribute")


@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealSamplerRegistration(unittest.TestCase):
    """Verify custom samplers are properly registered."""

    def test_extra_samplers_loaded(self):
        import comfy.samplers
        sampler_names = comfy.samplers.KSampler.SAMPLERS
        self.assertIsInstance(sampler_names, list)
        self.assertGreater(len(sampler_names), 0)
        print(f"Available samplers: {sampler_names}")


@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealTorchOperations(unittest.TestCase):
    """Tests that require real PyTorch tensors (skipped in mock mode)."""

    def test_ume_image_with_tensor(self):
        import torch
        from modules.common import UmeImage

        tensor = torch.rand(1, 512, 512, 3)
        mask = torch.zeros(1, 512, 512)
        img = UmeImage(image=tensor, mask=mask, mode="inpaint", denoise=0.75)
        self.assertEqual(img.image.shape, (1, 512, 512, 3))
        self.assertEqual(img.mode, "inpaint")
        self.assertAlmostEqual(img.denoise, 0.75)

    def test_generation_context_with_tensor(self):
        import torch
        from modules.common import GenerationContext

        ctx = GenerationContext()
        ctx.image = torch.rand(1, 256, 256, 3)
        self.assertEqual(ctx.image.shape, (1, 256, 256, 3))

    def test_outpaint_padding_real_tensors(self):
        import torch
        from modules.common import apply_outpaint_padding

        image = torch.rand(1, 64, 64, 3)
        mask = torch.ones(1, 64, 64)
        result_img, result_mask = apply_outpaint_padding(
            image, mask, pad_l=16, pad_t=16, pad_r=16, pad_b=16
        )
        self.assertEqual(result_img.shape[1], 96)  # 64 + 16 + 16
        self.assertEqual(result_img.shape[2], 96)


# ═══════════════════════════════════════════════════════════════
# Phase 2 — Coverage Expansion (GPU tensor + real runtime)
# ═══════════════════════════════════════════════════════════════

@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealResizeTensor(unittest.TestCase):
    """resize_tensor with real PyTorch — bypasses DummyTorch skip in mock tests."""

    def test_resize_image_downscale(self):
        import torch
        from modules.common import resize_tensor
        img = torch.rand(1, 256, 256, 3)
        resized = resize_tensor(img, 128, 128)
        self.assertEqual(resized.shape, (1, 128, 128, 3))

    def test_resize_image_upscale(self):
        import torch
        from modules.common import resize_tensor
        img = torch.rand(1, 64, 64, 3)
        resized = resize_tensor(img, 256, 256)
        self.assertEqual(resized.shape, (1, 256, 256, 3))

    def test_resize_mask(self):
        import torch
        from modules.common import resize_tensor
        mask = torch.rand(1, 256, 256)
        resized = resize_tensor(mask, 128, 128, is_mask=True)
        self.assertEqual(resized.shape, (1, 128, 128))

    def test_resize_preserves_batch(self):
        import torch
        from modules.common import resize_tensor
        img = torch.rand(4, 100, 200, 3)
        resized = resize_tensor(img, 50, 50)
        self.assertEqual(resized.shape[0], 4)
        self.assertEqual(resized.shape, (4, 50, 50, 3))

    def test_resize_nearest_mode(self):
        import torch
        from modules.common import resize_tensor
        img = torch.rand(1, 64, 64, 3)
        resized = resize_tensor(img, 32, 32, interp_mode="nearest")
        self.assertEqual(resized.shape, (1, 32, 32, 3))


@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealOutpaintEdgeCases(unittest.TestCase):
    """Edge cases for apply_outpaint_padding with real tensors."""

    def test_asymmetric_padding(self):
        """Asymmetric padding (left=0, right=32, top=0, bottom=32)."""
        import torch
        from modules.common import apply_outpaint_padding
        img = torch.rand(1, 64, 64, 3)
        result_img, result_mask = apply_outpaint_padding(
            img, None, pad_l=0, pad_t=0, pad_r=32, pad_b=32
        )
        self.assertEqual(result_img.shape, (1, 96, 96, 3))
        self.assertEqual(result_mask.shape, (1, 96, 96))

    def test_no_feathering_produces_near_binary_mask(self):
        """Zero feathering should produce a mask without soft gradients."""
        import torch
        from modules.common import apply_outpaint_padding
        img = torch.rand(1, 64, 64, 3)
        _, result_mask = apply_outpaint_padding(
            img, None, pad_l=16, pad_t=16, pad_r=16, pad_b=16, feathering=0
        )
        unique_vals = result_mask.unique()
        # Without feathering, mask should be binary (0.0 and 1.0 only)
        self.assertLessEqual(len(unique_vals), 2)

    def test_large_overlap_parameter(self):
        """Large overlap should still produce valid output."""
        import torch
        from modules.common import apply_outpaint_padding
        img = torch.rand(1, 64, 64, 3)
        result_img, result_mask = apply_outpaint_padding(
            img, None, pad_l=16, pad_t=16, pad_r=16, pad_b=16, overlap=32
        )
        self.assertEqual(result_img.shape[1], 96)
        # Mask values should be in [0, 1]
        self.assertGreaterEqual(result_mask.min().item(), 0.0)
        self.assertLessEqual(result_mask.max().item(), 1.0)


@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealOptimizations(unittest.TestCase):
    """GPU VRAM detection and optimization library checks."""

    def test_gpu_memory_returns_valid_string(self):
        """get_gpu_memory() should return a valid memory string on GPU runner."""
        from modules.optimization_utils import get_gpu_memory
        mem = get_gpu_memory()
        self.assertIn("GB", mem)
        self.assertIn("free", mem)
        print(f"  GPU Memory: {mem}")

    def test_cuda_memory_alias_works(self):
        """get_cuda_memory should still work as backward-compat alias."""
        from modules.optimization_utils import get_cuda_memory, get_gpu_memory
        self.assertIs(get_cuda_memory, get_gpu_memory)

    def test_check_library_triton(self):
        """Triton should be available in the ComfyUI container."""
        from modules.optimization_utils import check_library
        self.assertTrue(check_library("triton"))

    def test_check_library_nonexistent(self):
        """A non-existent library should return False."""
        from modules.optimization_utils import check_library
        self.assertFalse(check_library("nonexistent_fake_module_xyz"))

    def test_sampler_context_detects_optimization(self):
        """SamplerContext should detect at least Triton in GPU container."""
        from modules.optimization_utils import SamplerContext
        with SamplerContext() as ctx:
            self.assertIn(ctx.optimization_name, ("SageAttention", "Triton"))
            print(f"  Active optimization: {ctx.optimization_name}")


@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealExtraSamplers(unittest.TestCase):
    """Verify all 6 custom samplers are individually registered."""

    EXPECTED_SAMPLERS = [
        "sa_solver", "sa_solver_pece",
        "res_multistep", "res_multistep_cfg_pp",
        "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    ]

    def test_all_custom_samplers_registered(self):
        """All 6 custom samplers should appear in KSampler.SAMPLERS."""
        import comfy.samplers
        all_samplers = comfy.samplers.KSampler.SAMPLERS
        for name in self.EXPECTED_SAMPLERS:
            self.assertIn(name, all_samplers, f"Missing custom sampler: {name}")
        print(f"  ✅ All {len(self.EXPECTED_SAMPLERS)} custom samplers registered")

    def test_sa_solver_append_zero(self):
        """append_zero should add a zero at the end of a tensor."""
        import torch
        from modules.extra_samplers import append_zero
        x = torch.tensor([1.0, 0.5, 0.25])
        z = append_zero(x)
        self.assertEqual(z.shape, (4,))
        self.assertAlmostEqual(z[-1].item(), 0.0)

    def test_get_ancestral_step(self):
        """get_ancestral_step should return valid sigma_down and sigma_up."""
        import torch
        from modules.extra_samplers import get_ancestral_step
        sigma_down, sigma_up = get_ancestral_step(
            torch.tensor(1.0), torch.tensor(0.5), eta=1.0
        )
        self.assertGreaterEqual(float(sigma_down), 0.0)
        self.assertGreaterEqual(float(sigma_up), 0.0)

    def test_get_ancestral_step_no_eta(self):
        """With eta=0, sigma_up should be 0 and sigma_down should equal sigma_to."""
        import torch
        from modules.extra_samplers import get_ancestral_step
        sigma_down, sigma_up = get_ancestral_step(
            torch.tensor(1.0), torch.tensor(0.5), eta=0.0
        )
        self.assertAlmostEqual(float(sigma_up), 0.0)
        self.assertAlmostEqual(float(sigma_down), 0.5)

    def test_default_noise_sampler(self):
        """default_noise_sampler should produce correctly shaped noise."""
        import torch
        from modules.extra_samplers import default_noise_sampler
        x = torch.rand(1, 4, 64, 64)
        sampler = default_noise_sampler(x, seed=42)
        noise = sampler(torch.tensor(1.0), torch.tensor(0.5))
        self.assertEqual(noise.shape, x.shape)


@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealCommonUtils(unittest.TestCase):
    """Runtime tests for common.py utility functions."""

    def test_extract_pipeline_params(self):
        """extract_pipeline_params with populated GenerationContext."""
        from modules.common import GenerationContext, extract_pipeline_params
        ctx = GenerationContext()
        ctx.model = "model_placeholder"
        ctx.vae = "vae_placeholder"
        ctx.clip = "clip_placeholder"
        ctx.steps = 30
        ctx.cfg = 7.5
        ctx.sampler_name = "euler_ancestral"
        ctx.scheduler = "karras"
        ctx.seed = 12345
        ctx.positive_prompt = "a cat"
        ctx.negative_prompt = "bad"
        pp = extract_pipeline_params(ctx)
        self.assertEqual(pp.steps, 30)
        self.assertAlmostEqual(pp.cfg, 7.5)
        self.assertEqual(pp.sampler_name, "euler_ancestral")
        self.assertEqual(pp.seed, 12345)
        self.assertEqual(pp.pos_text, "a cat")

    def test_extract_pipeline_params_raises_on_missing(self):
        """extract_pipeline_params should raise ValueError if model/vae/clip are None."""
        from modules.common import GenerationContext, extract_pipeline_params
        ctx = GenerationContext()
        # model, vae, clip are all None
        with self.assertRaises(ValueError):
            extract_pipeline_params(ctx)

    def test_validate_bundle_passes(self):
        """validate_bundle should pass when required attrs are set."""
        from modules.common import validate_bundle, UmeBundle
        bundle = UmeBundle(model="m", clip="c", vae="v", model_name="test")
        # Should not raise
        validate_bundle(bundle, ["model", "clip", "vae"], context="test")

    def test_validate_bundle_raises_on_missing(self):
        """validate_bundle should raise ValueError on missing attrs."""
        from modules.common import validate_bundle, UmeBundle
        bundle = UmeBundle()  # all None
        with self.assertRaises(ValueError):
            validate_bundle(bundle, ["model", "clip"], context="test")

    def test_generation_context_clone_independence(self):
        """Clone should produce independent lists but share tensor refs."""
        import torch
        from modules.common import GenerationContext
        ctx = GenerationContext()
        ctx.image = torch.rand(1, 64, 64, 3)
        ctx.loras = [("lora1", 1.0, 1.0)]
        ctx.controlnets = [("cn1",)]

        cloned = ctx.clone()
        cloned.loras.append(("lora2", 0.5, 0.5))
        cloned.controlnets.append(("cn2",))

        # Original unchanged
        self.assertEqual(len(ctx.loras), 1)
        self.assertEqual(len(ctx.controlnets), 1)
        # Clone has new items
        self.assertEqual(len(cloned.loras), 2)
        self.assertEqual(len(cloned.controlnets), 2)
        # Tensor is shared (shallow copy)
        self.assertTrue(torch.equal(ctx.image, cloned.image))

    def test_generation_context_is_ready(self):
        """is_ready() should return True only when model+vae+clip are set."""
        from modules.common import GenerationContext
        ctx = GenerationContext()
        self.assertFalse(ctx.is_ready())
        ctx.model = "m"
        ctx.vae = "v"
        self.assertFalse(ctx.is_ready())
        ctx.clip = "c"
        self.assertTrue(ctx.is_ready())

    def test_ume_settings_defaults(self):
        """UmeSettings dataclass should have sane defaults."""
        from modules.common import UmeSettings
        s = UmeSettings()
        self.assertEqual(s.width, 1024)
        self.assertEqual(s.height, 1024)
        self.assertEqual(s.steps, 20)
        self.assertEqual(s.sampler_name, "euler")


@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealDetailDaemon(unittest.TestCase):
    """Detail Daemon schedule generation and sigma interpolation."""

    def test_schedule_generation(self):
        """make_detail_daemon_schedule should produce valid numpy arrays."""
        import numpy as np
        from modules.detail_daemon_nodes import make_detail_daemon_schedule
        sched = make_detail_daemon_schedule(
            steps=20, start=0.2, end=0.8, bias=0.5,
            amount=1.0, exponent=1.0, start_offset=0.0,
            end_offset=0.0, fade=0.0, smooth=True
        )
        self.assertEqual(len(sched), 20)
        self.assertTrue(np.all(np.isfinite(sched)))
        # Peak should be <= amount (1.0)
        self.assertLessEqual(float(sched.max()), 1.0 + 1e-6)

    def test_schedule_with_fade(self):
        """Non-zero fade should reduce the schedule amplitude."""
        import numpy as np
        from modules.detail_daemon_nodes import make_detail_daemon_schedule
        no_fade = make_detail_daemon_schedule(
            steps=20, start=0.2, end=0.8, bias=0.5,
            amount=1.0, exponent=1.0, start_offset=0.0,
            end_offset=0.0, fade=0.0, smooth=False
        )
        with_fade = make_detail_daemon_schedule(
            steps=20, start=0.2, end=0.8, bias=0.5,
            amount=1.0, exponent=1.0, start_offset=0.0,
            end_offset=0.0, fade=0.5, smooth=False
        )
        self.assertLessEqual(float(with_fade.max()), float(no_fade.max()))

    def test_get_dd_schedule_with_real_sigmas(self):
        """get_dd_schedule should interpolate correctly with real torch sigmas."""
        import torch
        from modules.detail_daemon_nodes import get_dd_schedule
        sigmas = torch.linspace(14.0, 0.0, 21)  # 20 steps
        dd_schedule = torch.linspace(0.0, 1.0, 20)
        val = get_dd_schedule(7.0, sigmas, dd_schedule)
        self.assertIsInstance(val, float)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_get_dd_schedule_out_of_range(self):
        """Sigma outside the sigmas range should return 0.0."""
        import torch
        from modules.detail_daemon_nodes import get_dd_schedule
        sigmas = torch.linspace(14.0, 0.0, 21)
        dd_schedule = torch.linspace(0.0, 1.0, 20)
        # Sigma way above range
        val = get_dd_schedule(100.0, sigmas, dd_schedule)
        self.assertAlmostEqual(val, 0.0)


@unittest.skipUnless(IN_CONTAINER, "Not in ComfyUI container")
class TestRealNodeContracts(unittest.TestCase):
    """Phase 2: Deep registration validation."""

    @classmethod
    def setUpClass(cls):
        custom_nodes_dir = os.path.dirname(TOOLKIT_ROOT)
        if custom_nodes_dir not in sys.path:
            sys.path.insert(0, custom_nodes_dir)
        toolkit_name = os.path.basename(TOOLKIT_ROOT)
        cls.mod = __import__(toolkit_name)

    def test_display_name_coverage(self):
        """Every NODE_CLASS_MAPPINGS key must have a NODE_DISPLAY_NAME_MAPPINGS entry."""
        for key in self.mod.NODE_CLASS_MAPPINGS:
            self.assertIn(key, self.mod.NODE_DISPLAY_NAME_MAPPINGS,
                          f"Missing display name for: {key}")

    def test_web_directory_defined(self):
        """WEB_DIRECTORY should be defined for frontend JS."""
        self.assertTrue(hasattr(self.mod, "WEB_DIRECTORY"))
        self.assertEqual(self.mod.WEB_DIRECTORY, "./web")

    def test_node_count_minimum(self):
        """Ensure no nodes were accidentally dropped during refactoring."""
        n = len(self.mod.NODE_CLASS_MAPPINGS)
        self.assertGreaterEqual(n, 28, f"Only {n} nodes registered, expected ≥28")
        print(f"  ✅ {n} nodes registered")


# ═══════════════════════════════════════════════════════════════
# Phase 3 — End-to-End Sampling (real model, real GPU)
# ═══════════════════════════════════════════════════════════════

# Check if a test model was downloaded by the CI job
TEST_MODEL_NAME = os.environ.get("TEST_MODEL_NAME", "")
TEST_MODEL_PATH = os.path.join(COMFYUI_ROOT, "models", "checkpoints", TEST_MODEL_NAME) if TEST_MODEL_NAME else ""
HAS_TEST_MODEL = bool(TEST_MODEL_PATH) and os.path.isfile(TEST_MODEL_PATH)


@unittest.skipUnless(IN_CONTAINER and HAS_TEST_MODEL, "Requires ComfyUI container + test model")
class TestEndToEndSampling(unittest.TestCase):
    """Full pipeline: load checkpoint → encode → sample → decode → validate."""

    _checkpoint = None
    _model = None
    _clip = None
    _vae = None

    @classmethod
    def setUpClass(cls):
        """Load the test checkpoint once for all sampling tests."""
        custom_nodes_dir = os.path.dirname(TOOLKIT_ROOT)
        if custom_nodes_dir not in sys.path:
            sys.path.insert(0, custom_nodes_dir)

        import torch
        import comfy.sd
        import comfy.utils

        print(f"  Loading test model: {TEST_MODEL_NAME}...")
        ckpt = comfy.sd.load_checkpoint_guess_config(
            TEST_MODEL_PATH,
            output_vae=True,
            output_clip=True,
        )
        cls._model = ckpt[0]
        cls._clip = ckpt[1]
        cls._vae = ckpt[2]
        print("  ✅ Model loaded successfully")

    def test_txt2img_pipeline(self):
        """txt2img: empty latent → 1 step → VAE decode → valid image tensor."""
        import torch
        import nodes as comfy_nodes

        # Encode prompts
        tokens = self._clip.tokenize("a red circle on white background")
        pos_cond = self._clip.encode_from_tokens_scheduled(tokens)
        tokens = self._clip.tokenize("bad quality")
        neg_cond = self._clip.encode_from_tokens_scheduled(tokens)

        # Create tiny empty latent (64x64 px = 8x8 latent)
        latent = {"samples": torch.zeros(1, 4, 8, 8, device="cpu")}

        # Sample 1 step
        ksampler = comfy_nodes.KSampler()
        result = ksampler.sample(
            self._model, seed=42, steps=1, cfg=7.0,
            sampler_name="euler", scheduler="normal",
            positive=pos_cond, negative=neg_cond,
            latent_image=latent, denoise=1.0
        )
        result_latent = result[0]
        self.assertIn("samples", result_latent)
        self.assertEqual(result_latent["samples"].shape, (1, 4, 8, 8))

        # VAE Decode
        vae_decode = comfy_nodes.VAEDecode()
        image = vae_decode.decode(self._vae, result_latent)[0]
        self.assertEqual(len(image.shape), 4)  # [B, H, W, C]
        self.assertEqual(image.shape[0], 1)
        self.assertEqual(image.shape[3], 3)  # RGB
        print(f"  ✅ txt2img output: {image.shape}")

    def test_img2img_pipeline(self):
        """img2img: encode source → 1 step denoise=0.5 → decode → valid shape."""
        import torch
        import nodes as comfy_nodes

        # Create a fake source image (64x64 RGB)
        source_image = torch.rand(1, 64, 64, 3)

        # Encode to latent
        vae_encode = comfy_nodes.VAEEncode()
        latent = vae_encode.encode(self._vae, source_image)[0]
        self.assertIn("samples", latent)

        # Encode prompts
        tokens = self._clip.tokenize("a blue square")
        pos_cond = self._clip.encode_from_tokens_scheduled(tokens)
        tokens = self._clip.tokenize("")
        neg_cond = self._clip.encode_from_tokens_scheduled(tokens)

        # Sample 1 step with low denoise
        ksampler = comfy_nodes.KSampler()
        result = ksampler.sample(
            self._model, seed=123, steps=1, cfg=5.0,
            sampler_name="euler", scheduler="normal",
            positive=pos_cond, negative=neg_cond,
            latent_image=latent, denoise=0.5
        )
        result_latent = result[0]

        # Decode
        vae_decode = comfy_nodes.VAEDecode()
        image = vae_decode.decode(self._vae, result_latent)[0]
        self.assertEqual(image.shape[0], 1)
        self.assertEqual(image.shape[3], 3)
        print(f"  ✅ img2img output: {image.shape}")

    def test_generation_context_full_pipeline(self):
        """GenerationContext populated from real model → clone → verify."""
        import torch
        from modules.common import GenerationContext

        ctx = GenerationContext()
        ctx.model = self._model
        ctx.clip = self._clip
        ctx.vae = self._vae
        ctx.model_name = TEST_MODEL_NAME
        ctx.steps = 1
        ctx.cfg = 7.0
        ctx.seed = 42
        ctx.width = 64
        ctx.height = 64

        self.assertTrue(ctx.is_ready())
        self.assertEqual(ctx.model_name, TEST_MODEL_NAME)

        # Clone and verify independence
        cloned = ctx.clone()
        cloned.seed = 999
        self.assertEqual(ctx.seed, 42)  # Original unchanged
        self.assertEqual(cloned.seed, 999)
        self.assertIs(cloned.model, ctx.model)  # Shared ref
        print(f"  ✅ GenerationContext with real model: ready={ctx.is_ready()}")


if __name__ == "__main__":
    unittest.main()
