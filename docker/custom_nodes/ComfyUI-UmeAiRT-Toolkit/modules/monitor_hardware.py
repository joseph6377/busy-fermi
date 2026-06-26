"""
UmeAiRT Hardware Monitor — Multi-platform GPU/CPU/RAM monitoring.

Supports:
  - NVIDIA (pynvml)
  - AMD ROCm (pyamdgpuinfo)
  - macOS Apple Silicon (torch.mps)
  - Fallback (torch.cuda for ROCm without pyamdgpuinfo)

Architecture:
  GPUBackend (abstract) → NvidiaBackend / AMDBackend / MPSBackend / TorchCUDAFallback
  HardwareMonitor       → Aggregates CPU, RAM, GPU data
  MonitorService        → Daemon thread, sends data via WebSocket at configurable rate
"""

import atexit
import asyncio
import threading
import time
import platform

import psutil

from .logger import log_node


# ---------------------------------------------------------------------------
# GPU Backend — Abstract base
# ---------------------------------------------------------------------------

class GPUBackend:
    """Abstract interface for GPU monitoring."""

    def get_device_type(self) -> str:
        raise NotImplementedError

    def get_gpu_count(self) -> int:
        raise NotImplementedError

    def get_name(self, index: int = 0) -> str:
        raise NotImplementedError

    def get_utilization(self, index: int = 0):
        """Return GPU utilization % or None if unavailable."""
        return None

    def get_vram(self, index: int = 0):
        """Return (used_bytes, total_bytes) or None if unavailable."""
        return None

    def get_temperature(self, index: int = 0):
        """Return temperature °C or None if unavailable."""
        return None

    def close(self):
        """Cleanup resources."""
        pass


# ---------------------------------------------------------------------------
# NVIDIA Backend (pynvml)
# ---------------------------------------------------------------------------

class NvidiaBackend(GPUBackend):
    def __init__(self, pynvml_module):
        self._nvml = pynvml_module
        self._handles = []
        count = self._nvml.nvmlDeviceGetCount()
        for i in range(count):
            self._handles.append(self._nvml.nvmlDeviceGetHandleByIndex(i))
        log_node(f"🖥️ NVIDIA GPU monitoring: {count} device(s) detected", color="GREEN")

    def get_device_type(self) -> str:
        return "nvidia"

    def get_gpu_count(self) -> int:
        return len(self._handles)

    def get_name(self, index: int = 0) -> str:
        try:
            name = self._nvml.nvmlDeviceGetName(self._handles[index])
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            return name
        except Exception:
            return "NVIDIA GPU"

    def get_utilization(self, index: int = 0):
        try:
            return float(self._nvml.nvmlDeviceGetUtilizationRates(self._handles[index]).gpu)
        except Exception:
            return None

    def get_vram(self, index: int = 0):
        try:
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handles[index])
            return (mem.used, mem.total)
        except Exception:
            return None

    def get_temperature(self, index: int = 0):
        try:
            return float(self._nvml.nvmlDeviceGetTemperature(
                self._handles[index],
                self._nvml.NVML_TEMPERATURE_GPU,
            ))
        except Exception:
            return None

    def close(self):
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AMD Backend (pyamdgpuinfo) — Linux only
# ---------------------------------------------------------------------------

class AMDBackend(GPUBackend):
    def __init__(self, amd_module):
        self._amd = amd_module
        self._count = self._amd.detect_gpus()
        log_node(f"🖥️ AMD GPU monitoring: {self._count} device(s) detected", color="GREEN")

    def get_device_type(self) -> str:
        return "amd"

    def get_gpu_count(self) -> int:
        return self._count

    def get_name(self, index: int = 0) -> str:
        try:
            gpu = self._amd.get_gpu(index)
            return gpu.name or "AMD GPU"
        except Exception:
            return "AMD GPU"

    def get_utilization(self, index: int = 0):
        try:
            gpu = self._amd.get_gpu(index)
            return float(gpu.query_load()) * 100.0
        except Exception:
            return None

    def get_vram(self, index: int = 0):
        try:
            gpu = self._amd.get_gpu(index)
            used = gpu.query_vram_usage()
            total = gpu.memory_info["vram_size"]
            return (used, total)
        except Exception:
            return None

    def get_temperature(self, index: int = 0):
        try:
            gpu = self._amd.get_gpu(index)
            return float(gpu.query_temperature())
        except Exception:
            return None


# ---------------------------------------------------------------------------
# macOS MPS Backend (torch.mps) — Apple Silicon
# ---------------------------------------------------------------------------

class MPSBackend(GPUBackend):
    def __init__(self):
        import torch
        self._torch = torch
        log_node("🖥️ macOS MPS monitoring: Apple Silicon detected", color="GREEN")

    def get_device_type(self) -> str:
        return "mps"

    def get_gpu_count(self) -> int:
        return 1

    def get_name(self, index: int = 0) -> str:
        # Apple doesn't expose chip name easily; use platform
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            brand = result.stdout.strip()
            if brand:
                return f"Apple {brand.split()[-1] if 'Apple' in brand else brand}"
        except Exception:
            pass
        return "Apple Silicon"

    def get_utilization(self, index: int = 0):
        # GPU utilization requires sudo powermetrics — not available
        return None

    def get_vram(self, index: int = 0):
        try:
            used = self._torch.mps.driver_allocated_memory()
            # On Apple Silicon, VRAM = unified RAM. Use total system RAM as reference.
            total = psutil.virtual_memory().total
            return (used, total)
        except Exception:
            return None

    def get_temperature(self, index: int = 0):
        # Temperature requires sudo powermetrics — not available
        return None


# ---------------------------------------------------------------------------
# Torch CUDA Fallback (for ROCm without pyamdgpuinfo)
# ---------------------------------------------------------------------------

class TorchCUDAFallback(GPUBackend):
    def __init__(self):
        import torch
        self._torch = torch
        self._count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if self._count > 0 else "Unknown"
        log_node(f"🖥️ GPU monitoring (torch.cuda fallback): {self._count} device(s) — {device_name}", color="YELLOW")

    def get_device_type(self) -> str:
        return "cuda"

    def get_gpu_count(self) -> int:
        return self._count

    def get_name(self, index: int = 0) -> str:
        try:
            return self._torch.cuda.get_device_name(index)
        except Exception:
            return "GPU"

    def get_utilization(self, index: int = 0):
        # torch.cuda doesn't reliably expose utilization on ROCm
        return None

    def get_vram(self, index: int = 0):
        try:
            free, total = self._torch.cuda.mem_get_info(index)
            used = total - free
            return (used, total)
        except Exception:
            return None

    def get_temperature(self, index: int = 0):
        return None


# ---------------------------------------------------------------------------
# Backend auto-detection (cascade)
# ---------------------------------------------------------------------------

def detect_gpu_backend() -> GPUBackend | None:
    """Auto-detect the best available GPU monitoring backend."""

    # 1. NVIDIA — pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            return NvidiaBackend(pynvml)
    except ImportError:
        log_node("ℹ️ pynvml not installed — NVIDIA GPU monitoring unavailable", color="YELLOW")
    except Exception as e:
        log_node(f"⚠️ pynvml init failed: {e}", color="YELLOW")

    # 2. AMD — pyamdgpuinfo (Linux only)
    if platform.system() == "Linux":
        try:
            import pyamdgpuinfo
            if pyamdgpuinfo.detect_gpus() > 0:
                return AMDBackend(pyamdgpuinfo)
        except ImportError:
            pass
        except Exception as e:
            log_node(f"⚠️ pyamdgpuinfo init failed: {e}", color="YELLOW")

    # 3. macOS — MPS
    if platform.system() == "Darwin":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return MPSBackend()
        except Exception:
            pass

    # 4. Fallback — torch.cuda (covers ROCm via HIP without pyamdgpuinfo)
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return TorchCUDAFallback()
    except Exception:
        pass

    log_node("ℹ️ No GPU detected — monitoring CPU/RAM only", color="YELLOW")
    return None


# ---------------------------------------------------------------------------
# Hardware Monitor (aggregation layer)
# ---------------------------------------------------------------------------

_gpu_backend: GPUBackend | None = None


def _init_backend():
    """Lazy-init the GPU backend (called once)."""
    global _gpu_backend
    if _gpu_backend is None:
        _gpu_backend = detect_gpu_backend()
    return _gpu_backend


class HardwareMonitor:
    """Aggregates CPU, RAM, and GPU metrics into a single status dict."""

    def __init__(self):
        self.backend = _init_backend()

    def get_status(self) -> dict:
        # CPU
        cpu = psutil.cpu_percent(interval=None)

        # RAM
        ram = psutil.virtual_memory()

        # GPUs
        gpus = []
        device_type = "cpu"

        if self.backend is not None:
            device_type = self.backend.get_device_type()
            for i in range(self.backend.get_gpu_count()):
                gpu_data = {
                    "gpu_name": self.backend.get_name(i),
                    "gpu_utilization": self.backend.get_utilization(i),
                    "gpu_temperature": self.backend.get_temperature(i),
                    "vram_total": None,
                    "vram_used": None,
                    "vram_used_percent": None,
                }
                vram = self.backend.get_vram(i)
                if vram is not None:
                    used, total = vram
                    gpu_data["vram_used"] = used
                    gpu_data["vram_total"] = total
                    if total and total > 0:
                        gpu_data["vram_used_percent"] = round(used / total * 100, 1)

                gpus.append(gpu_data)

        return {
            "cpu_utilization": cpu,
            "ram_total": ram.total,
            "ram_used": ram.used,
            "ram_used_percent": ram.percent,
            "device_type": device_type,
            "gpus": gpus,
        }


def get_gpu_info() -> list[dict]:
    """Return list of detected GPUs (for the /umeairt/monitor/gpu-info route)."""
    backend = _init_backend()
    if backend is None:
        return []

    gpus = []
    for i in range(backend.get_gpu_count()):
        gpus.append({
            "index": i,
            "name": backend.get_name(i),
            "device_type": backend.get_device_type(),
        })
    return gpus


# ---------------------------------------------------------------------------
# Monitor Service (daemon thread + WebSocket broadcast)
# ---------------------------------------------------------------------------

class MonitorService:
    """Background thread that periodically sends hardware metrics via WebSocket."""

    def __init__(self, rate: float = 1.0):
        self.rate = rate
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._monitor = HardwareMonitor()
        self._server = None

    def _get_server(self):
        """Lazy import of PromptServer to avoid import-time issues."""
        if self._server is None:
            try:
                import server
                self._server = server.PromptServer.instance
            except Exception:
                pass
        return self._server

    def _loop(self):
        """Main monitoring loop — runs in daemon thread."""
        asyncio.run(self._async_loop())

    async def _async_loop(self):
        while not self._stop_event.is_set():
            if self.rate <= 0:
                await asyncio.sleep(0.5)
                continue

            try:
                data = self._monitor.get_status()
                srv = self._get_server()
                if srv is not None:
                    srv.send_sync("umeairt.monitor", data)
            except Exception as e:
                log_node(f"⚠️ Monitor error: {e}", color="RED")

            await asyncio.sleep(self.rate)

    def start(self):
        """Start the monitoring thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log_node(f"📊 Hardware monitor started (rate: {self.rate}s)", color="GREEN")

    def stop(self):
        """Stop the monitoring thread."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=3)
        self._thread = None
        log_node("📊 Hardware monitor stopped", color="YELLOW")

    def set_rate(self, rate: float):
        """Update the refresh rate (takes effect on next cycle)."""
        self.rate = max(0.0, rate)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ---------------------------------------------------------------------------
# Module-level singleton (lazy — not started until __init__.py calls .start())
# ---------------------------------------------------------------------------

_service_instance: MonitorService | None = None


def get_monitor_service(rate: float = 1.0) -> MonitorService:
    """Get or create the singleton MonitorService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = MonitorService(rate=rate)
        atexit.register(_service_instance.stop)
    return _service_instance
