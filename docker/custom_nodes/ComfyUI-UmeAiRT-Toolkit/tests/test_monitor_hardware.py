"""Tests for modules/monitor_hardware.py — Hardware monitoring backend."""
import unittest
from unittest.mock import MagicMock, patch


class TestGPUBackendInterface(unittest.TestCase):
    """Test the abstract GPUBackend interface."""

    def test_import(self):
        from modules.monitor_hardware import GPUBackend
        self.assertIsNotNone(GPUBackend)

    def test_default_methods(self):
        from modules.monitor_hardware import GPUBackend
        backend = GPUBackend()
        self.assertIsNone(backend.get_utilization())
        self.assertIsNone(backend.get_vram())
        self.assertIsNone(backend.get_temperature())
        backend.close()  # should not raise

    def test_abstract_methods_raise(self):
        from modules.monitor_hardware import GPUBackend
        backend = GPUBackend()
        with self.assertRaises(NotImplementedError):
            backend.get_device_type()
        with self.assertRaises(NotImplementedError):
            backend.get_gpu_count()
        with self.assertRaises(NotImplementedError):
            backend.get_name()


class TestNvidiaBackend(unittest.TestCase):
    """Test NvidiaBackend with mocked pynvml."""

    @patch("modules.monitor_hardware.log_node")
    def _make_backend(self, mock_log):
        from modules.monitor_hardware import NvidiaBackend
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetCount.return_value = 2
        mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"handle_{i}"
        mock_nvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 4080"
        mock_nvml.NVML_TEMPERATURE_GPU = 0

        util_mock = MagicMock()
        util_mock.gpu = 75.0
        mock_nvml.nvmlDeviceGetUtilizationRates.return_value = util_mock

        mem_mock = MagicMock()
        mem_mock.used = 4 * 1024**3
        mem_mock.total = 16 * 1024**3
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem_mock

        mock_nvml.nvmlDeviceGetTemperature.return_value = 62.0

        return NvidiaBackend(mock_nvml), mock_nvml

    def test_device_type(self):
        backend, _ = self._make_backend()
        self.assertEqual(backend.get_device_type(), "nvidia")

    def test_gpu_count(self):
        backend, _ = self._make_backend()
        self.assertEqual(backend.get_gpu_count(), 2)

    def test_get_name(self):
        backend, _ = self._make_backend()
        self.assertEqual(backend.get_name(0), "NVIDIA RTX 4080")

    @patch("modules.monitor_hardware.log_node")
    def test_get_name_string(self, mock_log):
        """Test name when pynvml returns str instead of bytes."""
        from modules.monitor_hardware import NvidiaBackend
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "h0"
        mock_nvml.nvmlDeviceGetName.return_value = "RTX 5090"
        backend = NvidiaBackend(mock_nvml)
        self.assertEqual(backend.get_name(0), "RTX 5090")

    def test_utilization(self):
        backend, _ = self._make_backend()
        self.assertEqual(backend.get_utilization(0), 75.0)

    def test_vram(self):
        backend, _ = self._make_backend()
        used, total = backend.get_vram(0)
        self.assertEqual(used, 4 * 1024**3)
        self.assertEqual(total, 16 * 1024**3)

    def test_temperature(self):
        backend, _ = self._make_backend()
        self.assertEqual(backend.get_temperature(0), 62.0)

    def test_close(self):
        backend, mock_nvml = self._make_backend()
        backend.close()
        mock_nvml.nvmlShutdown.assert_called_once()

    @patch("modules.monitor_hardware.log_node")
    def test_error_handling(self, mock_log):
        """Errors in GPU queries should return None, not crash."""
        from modules.monitor_hardware import NvidiaBackend
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "h0"
        mock_nvml.nvmlDeviceGetName.side_effect = Exception("fail")
        mock_nvml.nvmlDeviceGetUtilizationRates.side_effect = Exception("fail")
        mock_nvml.nvmlDeviceGetMemoryInfo.side_effect = Exception("fail")
        mock_nvml.nvmlDeviceGetTemperature.side_effect = Exception("fail")
        backend = NvidiaBackend(mock_nvml)
        self.assertEqual(backend.get_name(0), "NVIDIA GPU")
        self.assertIsNone(backend.get_utilization(0))
        self.assertIsNone(backend.get_vram(0))
        self.assertIsNone(backend.get_temperature(0))


class TestTorchCUDAFallback(unittest.TestCase):
    """Test TorchCUDAFallback with mocked torch."""

    @patch("modules.monitor_hardware.log_node")
    def test_basics(self, mock_log):
        # We need to patch torch where it is used, or since it's locally imported,
        # we can mock sys.modules
        with patch.dict('sys.modules', {'torch': MagicMock()}):
            import sys
            mock_torch = sys.modules['torch']
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_name.return_value = "AMD Radeon RX 7900"
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)

            from modules.monitor_hardware import TorchCUDAFallback
            backend = TorchCUDAFallback()
            self.assertEqual(backend.get_device_type(), "cuda")
            self.assertEqual(backend.get_gpu_count(), 1)
            self.assertEqual(backend.get_name(0), "AMD Radeon RX 7900")
            self.assertIsNone(backend.get_utilization(0))
            self.assertIsNone(backend.get_temperature(0))

            used, total = backend.get_vram(0)
            self.assertEqual(used, 8 * 1024**3)  # total - free
            self.assertEqual(total, 16 * 1024**3)


class TestHardwareMonitor(unittest.TestCase):
    """Test the HardwareMonitor aggregator."""

    @patch("modules.monitor_hardware._init_backend")
    @patch("modules.monitor_hardware.psutil")
    def test_get_status_no_gpu(self, mock_psutil, mock_init):
        mock_init.return_value = None
        mock_psutil.cpu_percent.return_value = 25.0
        ram_mock = MagicMock()
        ram_mock.total = 32 * 1024**3
        ram_mock.used = 16 * 1024**3
        ram_mock.percent = 50.0
        mock_psutil.virtual_memory.return_value = ram_mock

        from modules.monitor_hardware import HardwareMonitor
        monitor = HardwareMonitor()
        status = monitor.get_status()

        self.assertEqual(status["cpu_utilization"], 25.0)
        self.assertEqual(status["ram_used_percent"], 50.0)
        self.assertEqual(status["device_type"], "cpu")
        self.assertEqual(len(status["gpus"]), 0)

    @patch("modules.monitor_hardware._init_backend")
    @patch("modules.monitor_hardware.psutil")
    def test_get_status_with_gpu(self, mock_psutil, mock_init):
        backend = MagicMock()
        backend.get_device_type.return_value = "nvidia"
        backend.get_gpu_count.return_value = 1
        backend.get_name.return_value = "RTX 4080"
        backend.get_utilization.return_value = 80.0
        backend.get_vram.return_value = (8 * 1024**3, 16 * 1024**3)
        backend.get_temperature.return_value = 72.0
        mock_init.return_value = backend

        mock_psutil.cpu_percent.return_value = 10.0
        ram_mock = MagicMock()
        ram_mock.total = 64 * 1024**3
        ram_mock.used = 32 * 1024**3
        ram_mock.percent = 50.0
        mock_psutil.virtual_memory.return_value = ram_mock

        from modules.monitor_hardware import HardwareMonitor
        monitor = HardwareMonitor()
        status = monitor.get_status()

        self.assertEqual(status["device_type"], "nvidia")
        self.assertEqual(len(status["gpus"]), 1)
        gpu = status["gpus"][0]
        self.assertEqual(gpu["gpu_name"], "RTX 4080")
        self.assertEqual(gpu["gpu_utilization"], 80.0)
        self.assertEqual(gpu["gpu_temperature"], 72.0)
        self.assertEqual(gpu["vram_used"], 8 * 1024**3)
        self.assertEqual(gpu["vram_total"], 16 * 1024**3)
        self.assertAlmostEqual(gpu["vram_used_percent"], 50.0)


class TestMonitorService(unittest.TestCase):
    """Test the MonitorService lifecycle."""

    @patch("modules.monitor_hardware.log_node")
    def test_service_creation(self, mock_log):
        from modules.monitor_hardware import MonitorService
        service = MonitorService(rate=2.0)
        self.assertEqual(service.rate, 2.0)
        self.assertFalse(service.is_running)

    @patch("modules.monitor_hardware.log_node")
    def test_set_rate(self, mock_log):
        from modules.monitor_hardware import MonitorService
        service = MonitorService(rate=1.0)
        service.set_rate(0.5)
        self.assertEqual(service.rate, 0.5)
        service.set_rate(-1.0)
        self.assertEqual(service.rate, 0.0)  # clamped to 0

    @patch("modules.monitor_hardware.log_node")
    def test_get_monitor_service_singleton(self, mock_log):
        """get_monitor_service should return the same instance."""
        import modules.monitor_hardware as mh
        old = mh._service_instance
        try:
            mh._service_instance = None  # Reset singleton
            s1 = mh.get_monitor_service(rate=1.0)
            s2 = mh.get_monitor_service(rate=2.0)
            self.assertIs(s1, s2)
        finally:
            mh._service_instance = old  # Restore


class TestGetGpuInfo(unittest.TestCase):
    """Test the get_gpu_info helper."""

    @patch("modules.monitor_hardware._init_backend")
    def test_no_gpu(self, mock_init):
        mock_init.return_value = None
        from modules.monitor_hardware import get_gpu_info
        result = get_gpu_info()
        self.assertEqual(result, [])

    @patch("modules.monitor_hardware._init_backend")
    def test_with_gpus(self, mock_init):
        backend = MagicMock()
        backend.get_gpu_count.return_value = 2
        backend.get_name.side_effect = ["GPU A", "GPU B"]
        backend.get_device_type.return_value = "nvidia"
        mock_init.return_value = backend

        from modules.monitor_hardware import get_gpu_info
        result = get_gpu_info()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "GPU A")
        self.assertEqual(result[1]["name"], "GPU B")
        self.assertEqual(result[0]["index"], 0)
        self.assertEqual(result[1]["index"], 1)


if __name__ == "__main__":
    unittest.main()
