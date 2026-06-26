# 📊 Hardware Monitor

Built-in system resource monitoring that replaces [Crystools](https://github.com/crystian/ComfyUI-Crystools) — no external custom node needed.

## Overview

The Hardware Monitor displays **CPU, RAM, GPU utilization, VRAM, and Temperature** directly in the ComfyUI top bar. It also provides a **contextual progress bar** that shows what the pipeline is currently doing.

## Settings

Available in **Settings → UmeAiRT → Monitor**:

| Setting | Default | Description |
|---------|---------|-------------|
| Enable Hardware Monitor | `true` | Show/hide the monitoring bar. Disabling also stops the backend thread. |
| Monitor Style | `Micro Gauges` | Visual style: Glassmorphism, Accent Strip, or Micro Gauges. |
| Refresh Rate | `1s` | Update frequency (0.5s–5s). Lower = more responsive but slightly more CPU usage. |

## Styles

### Micro Gauges (Default)
Compact rectangles with gradient fill, label at bottom-left, value at top-right.

### Glassmorphism Pills
Frosted glass effect with backdrop blur, colored border, and thin fill bar at the bottom.

### Accent Strip
Ultra-minimal — colored label text with a 2px accent line. Best for small screens.

## Progress Bar

During pipeline execution, a progress pill appears showing:

| Phase | Label Example |
|-------|---------------|
| Image generation | `Generating 45%` |
| Upscaling (tile 1) | `Upscaling 30%` |
| Upscaling (tile 2+) | `Upscaling T2 60%` |
| Face detailing | `Detailing 80%` |
| Detail refining | `Refining 55%` |
| Video generation | `Video Gen 20%` |
| Frame interpolation | `Interpolating 90%` |
| Saving | `Saving 100%` |

> **Tip**: Hover the progress pill for a detailed tooltip showing the current step, total steps, percentage, and internal node type.

After the pipeline finishes, the bar shows `Idle` after a brief 1.5s delay.

## Platform Support

| Platform | GPU% | VRAM | Temperature | Library |
|----------|------|------|-------------|---------|
| NVIDIA (Windows/Linux) | ✅ | ✅ | ✅ | `pynvml` (auto-installed) |
| AMD ROCm (Linux) | ✅ | ✅ | ✅ | `pyamdgpuinfo` (manual install) |
| Apple Silicon (macOS) | ❌ | ✅ | ❌ | `torch.mps` (built-in) |
| CUDA Fallback | ❌ | ✅ | ❌ | `torch.cuda` (built-in) |

Multi-GPU setups (RunPod, etc.) show separate metrics per GPU.

On macOS, VRAM reports unified memory usage via `torch.mps.driver_allocated_memory()`.

## VRAM Tracking

- **Peak VRAM**: The monitor tracks peak VRAM usage per GPU.
- **Reset**: Double-click the VRAM pill to reset the peak counter.
- **Tooltip**: Hover to see current used/total, peak, and GPU name.

## Technical Details

### Backend (`modules/monitor_hardware.py`)

- **GPUBackend** abstract class with 4 implementations (cascade auto-detection)
- **HardwareMonitor** aggregates CPU, RAM, and GPU data
- **MonitorService** daemon thread broadcasts data via `PromptServer.send_sync('umeairt.monitor', data)`

### Frontend (`web/umeairt_monitor.js` + `web/umeairt_monitor.css`)

- ComfyUI extension registered via `app.registerExtension()`
- WebSocket listener for `umeairt.monitor` (hardware data)
- WebSocket listeners for `executing` and `progress` (pipeline progress)
- Node type resolved via `app.graph.getNodeById()` for contextual labels

### API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/umeairt/monitor/gpu-info` | GET | Returns list of detected GPUs |
| `/umeairt/monitor/settings` | PATCH | Update rate and enabled state |
