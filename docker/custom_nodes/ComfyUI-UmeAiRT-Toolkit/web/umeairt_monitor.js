import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// UmeAiRT Hardware Monitor — Frontend Extension
// Multi-platform GPU/CPU/RAM monitoring with 3 switchable styles
// ---------------------------------------------------------------------------

const METRIC_COLORS = {
    cpu:      "#0AA015",
    ram:      "#07630D",
    gpu:      "#2471A3",
    vram:     "#154360",
    temp:     "#52BE80",
    tempHot:  "#E74C3C",
    progress: "#CD8B62",
    mps:      "#9B59B6",
};

const STYLE_MAP = {
    "Glassmorphism": "glassmorphism",
    "Accent Strip": "strip",
    "Micro Gauges": "gauges",
};

// Contextual labels for progress bar based on executing node type
const NODE_PROGRESS_LABELS = {
    // Image generation
    "UmeAiRT_BlockSampler":              "Generating",
    // Upscaling
    "UmeAiRT_PipelineUltimateUpscale":   "Upscaling",
    "UmeAiRT_PipelineSeedVR2Upscale":    "Upscaling",
    // Detailing
    "UmeAiRT_PipelineSubjectDetailer":   "Detailing",
    "UmeAiRT_Detailer_Daemon":           "Detailing",
    "UmeAiRT_DetailRefiner":             "Refining",
    // Video
    "UmeAiRT_VideoGenerator":            "Video Gen",
    "UmeAiRT_LTXVideoGenerator":         "Video Gen",
    "UmeAiRT_LTXVideoExtender":          "Extending",
    "UmeAiRT_LTXVideoEnhancer":          "Enhancing",
    "UmeAiRT_VideoLooper":               "Looping",
    "UmeAiRT_VideoExtender":             "Extending",
    // Post-processing
    "UmeAiRT_VideoFrameInterpolation":   "Interpolating",
    "UmeAiRT_VideoSmartUpscale":         "Upscaling",
    // Analysis
    "UmeAiRT_ImageToPrompt":             "Analyzing",
    // Output
    "UmeAiRT_PipelineImageSaver":        "Saving",
    "UmeAiRT_VideoOutput":               "Encoding",
    // Loader
    "UmeAiRT_BundleLoader":              "Loading",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatBytes(bytes) {
    if (bytes == null || bytes < 0) return "—";
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return gb.toFixed(1) + " GB";
    const mb = bytes / (1024 * 1024);
    return mb.toFixed(0) + " MB";
}

function getTemperatureColor(temp) {
    if (temp == null || temp < 0) return METRIC_COLORS.temp;
    const pct = Math.min(100, Math.max(0, temp));
    return `color-mix(in srgb, ${METRIC_COLORS.tempHot} ${pct}%, ${METRIC_COLORS.temp})`;
}

function getWarningState(value) {
    if (value == null || value < 0) return "normal";
    if (value >= 95) return "critical";
    if (value >= 80) return "warning";
    return "normal";
}

// ---------------------------------------------------------------------------
// Tooltip Manager
// ---------------------------------------------------------------------------

class TooltipManager {
    constructor() {
        this._el = null;
    }

    show(targetEl, html) {
        this.hide();
        const tip = document.createElement("div");
        tip.className = "umeairt-monitor__tooltip";
        tip.innerHTML = html;
        document.body.appendChild(tip);
        this._el = tip;

        const rect = targetEl.getBoundingClientRect();
        tip.style.left = rect.left + "px";
        tip.style.top = (rect.bottom + 6) + "px";

        // Keep in viewport
        requestAnimationFrame(() => {
            if (!this._el) return;
            const tipRect = this._el.getBoundingClientRect();
            if (tipRect.right > window.innerWidth - 8) {
                this._el.style.left = (window.innerWidth - tipRect.width - 8) + "px";
            }
        });
    }

    hide() {
        if (this._el) {
            this._el.remove();
            this._el = null;
        }
    }
}

// ---------------------------------------------------------------------------
// Monitor UI
// ---------------------------------------------------------------------------

class UmeAiRTMonitor {
    constructor() {
        this.idExtensionName = "UmeAiRT.monitor";
        this.rootElement = null;
        this.items = {};
        this.peakVRAM = {};
        this.tooltip = new TooltipManager();
        this.currentStyle = "gauges";
        this.gpuInfoLoaded = false;
        this.enabled = true;
        // Progress tracking
        this._currentNodeId = null;
        this._currentNodeType = null;
        this._currentNodeLabel = "Processing";
        this._progressValue = 0;
        this._progressMax = 0;
        // Tile tracking (for UltimateSD Upscale, SeedVR2, etc.)
        this._tileCount = 0;
        this._prevProgressValue = 0;
    }

    // -- Settings ----------------------------------------------------------

    registerSettings() {
        app.ui.settings.addSetting({
            id: "UmeAiRT.Monitor.Enabled",
            name: "Enable Hardware Monitor",
            category: ["UmeAiRT", "Monitor", "Enable"],
            tooltip: "Show CPU, RAM, GPU, VRAM and Temperature in the top bar",
            type: "boolean",
            defaultValue: true,
            onChange: async (value) => {
                this.enabled = value;
                if (this.rootElement) {
                    this.rootElement.style.display = value ? "flex" : "none";
                }
                try {
                    await api.fetchApi("/umeairt/monitor/settings", {
                        method: "PATCH",
                        body: JSON.stringify({ enabled: value }),
                    });
                } catch (e) { /* server may not be ready */ }
            },
        });

        app.ui.settings.addSetting({
            id: "UmeAiRT.Monitor.Style",
            name: "Monitor Style",
            category: ["UmeAiRT", "Monitor", "Style"],
            tooltip: "Visual style for the monitoring bar",
            type: "combo",
            options: Object.keys(STYLE_MAP),
            defaultValue: "Micro Gauges",
            onChange: (value) => {
                this.setStyle(STYLE_MAP[value] || "gauges");
            },
        });

        app.ui.settings.addSetting({
            id: "UmeAiRT.Monitor.RefreshRate",
            name: "Refresh Rate (seconds)",
            category: ["UmeAiRT", "Monitor", "Refresh Rate"],
            tooltip: "How often hardware metrics are updated (lower = more CPU usage)",
            type: "slider",
            attrs: { min: 0.5, max: 5, step: 0.5 },
            defaultValue: 1,
            onChange: async (value) => {
                try {
                    await api.fetchApi("/umeairt/monitor/settings", {
                        method: "PATCH",
                        body: JSON.stringify({ rate: parseFloat(value) }),
                    });
                } catch (e) { /* server may not be ready */ }
            },
        });
    }

    // -- Style switching ---------------------------------------------------

    setStyle(style) {
        if (!this.rootElement) return;
        Object.values(STYLE_MAP).forEach(s => {
            this.rootElement.classList.remove("umeairt-monitor--" + s);
        });
        this.rootElement.classList.add("umeairt-monitor--" + style);
        this.currentStyle = style;
    }

    // -- DOM creation ------------------------------------------------------

    createRootElement() {
        const root = document.createElement("div");
        root.id = "umeairt-monitor-root";
        root.className = "umeairt-monitor umeairt-monitor--" + this.currentStyle;
        return root;
    }

    createMonitorItem(key, label, color) {
        const item = document.createElement("div");
        item.className = "umeairt-monitor__item";
        item.dataset.metric = key;
        item.dataset.state = "normal";

        const dot = document.createElement("span");
        dot.className = "umeairt-monitor__dot";
        dot.style.setProperty("--metric-color", color);

        const labelEl = document.createElement("span");
        labelEl.className = "umeairt-monitor__label";
        labelEl.textContent = label;
        labelEl.style.setProperty("--metric-color", color);

        const valueEl = document.createElement("span");
        valueEl.className = "umeairt-monitor__value";
        valueEl.textContent = "—";

        const bar = document.createElement("div");
        bar.className = "umeairt-monitor__bar";

        const fill = document.createElement("div");
        fill.className = "umeairt-monitor__fill";
        fill.dataset.metric = key.startsWith("temp") ? "temp" : key;
        fill.style.setProperty("--metric-color", color);
        fill.style.backgroundColor = color;
        bar.appendChild(fill);

        item.append(dot, labelEl, valueEl, bar);

        // Tooltip events
        item.addEventListener("mouseenter", () => this.showTooltip(item, key));
        item.addEventListener("mouseleave", () => this.tooltip.hide());

        // Double-click to reset peak VRAM
        if (key.startsWith("vram") || key === "mps") {
            item.addEventListener("dblclick", () => {
                const gpuIdx = key.replace("vram", "").replace("mps", "0") || "0";
                this.peakVRAM[gpuIdx] = 0;
            });
            item.style.cursor = "pointer";
        }

        this.items[key] = { el: item, valueEl, fill, bar, label, color };
        return item;
    }

    createSeparator() {
        const sep = document.createElement("div");
        sep.className = "umeairt-monitor__sep";
        return sep;
    }

    createProgressItem() {
        const item = document.createElement("div");
        item.className = "umeairt-monitor__item umeairt-monitor__progress";
        item.dataset.metric = "progress";
        item.dataset.state = "normal";
        item.dataset.active = "false";

        const dot = document.createElement("span");
        dot.className = "umeairt-monitor__dot";
        dot.style.setProperty("--metric-color", METRIC_COLORS.progress);

        const labelEl = document.createElement("span");
        labelEl.className = "umeairt-monitor__label";
        labelEl.textContent = "Idle";
        labelEl.style.setProperty("--metric-color", METRIC_COLORS.progress);

        const valueEl = document.createElement("span");
        valueEl.className = "umeairt-monitor__value";
        valueEl.textContent = "";

        const bar = document.createElement("div");
        bar.className = "umeairt-monitor__bar";

        const fill = document.createElement("div");
        fill.className = "umeairt-monitor__fill";
        fill.style.setProperty("--metric-color", METRIC_COLORS.progress);
        fill.style.backgroundColor = METRIC_COLORS.progress;
        bar.appendChild(fill);

        item.append(dot, labelEl, valueEl, bar);

        // Tooltip for progress bar
        item.addEventListener("mouseenter", () => this.showProgressTooltip(item));
        item.addEventListener("mouseleave", () => this.tooltip.hide());

        this.items.progress = { el: item, valueEl, fill, bar, labelEl, label: "Idle", color: METRIC_COLORS.progress };
        return item;
    }

    // -- Build the full bar ------------------------------------------------

    async buildMonitorBar() {
        if (!this.rootElement) return;

        // Static items: CPU + RAM
        this.rootElement.appendChild(this.createMonitorItem("cpu", "CPU", METRIC_COLORS.cpu));
        this.rootElement.appendChild(this.createMonitorItem("ram", "RAM", METRIC_COLORS.ram));

        // Fetch GPU info from backend
        try {
            const resp = await api.fetchApi("/umeairt/monitor/gpu-info", { method: "GET" });
            if (resp.status === 200) {
                const gpus = await resp.json();
                const multi = gpus.length > 1;

                if (gpus.length > 0) {
                    this.rootElement.appendChild(this.createSeparator());
                }

                gpus.forEach((gpu, i) => {
                    const suffix = multi ? "₊" + i : "";
                    const idSuffix = multi ? i.toString() : "";

                    if (gpu.device_type === "mps") {
                        // macOS — only MPS memory, no GPU% or temp
                        this.rootElement.appendChild(
                            this.createMonitorItem("mps" + idSuffix, "MPS" + suffix, METRIC_COLORS.mps)
                        );
                    } else {
                        // GPU utilization
                        this.rootElement.appendChild(
                            this.createMonitorItem("gpu" + idSuffix, "GPU" + suffix, METRIC_COLORS.gpu)
                        );
                        // VRAM
                        this.rootElement.appendChild(
                            this.createMonitorItem("vram" + idSuffix, "VRAM" + suffix, METRIC_COLORS.vram)
                        );
                        // Temperature
                        this.rootElement.appendChild(
                            this.createMonitorItem("temp" + idSuffix, "Temp" + suffix, METRIC_COLORS.temp)
                        );
                    }

                    if (multi && i < gpus.length - 1) {
                        this.rootElement.appendChild(this.createSeparator());
                    }
                });

                this.gpuInfoLoaded = true;
            }
        } catch (e) {
            console.warn("[UmeAiRT Monitor] Could not fetch GPU info:", e);
        }

        // Progress bar (last item)
        this.rootElement.appendChild(this.createSeparator());
        this.rootElement.appendChild(this.createProgressItem());
    }

    // -- Data update -------------------------------------------------------

    updateDisplay(data) {
        if (!this.enabled) return;

        // CPU
        this.updateItem("cpu", data.cpu_utilization, "%");

        // RAM
        this.updateItem("ram", data.ram_used_percent, "%");

        // GPUs
        if (data.gpus && data.gpus.length > 0) {
            const multi = data.gpus.length > 1;

            data.gpus.forEach((gpu, i) => {
                const idSuffix = multi ? i.toString() : "";

                if (data.device_type === "mps") {
                    // macOS MPS — show memory in absolute value
                    const mpsKey = "mps" + idSuffix;
                    const item = this.items[mpsKey];
                    if (item && gpu.vram_used != null) {
                        item.valueEl.textContent = formatBytes(gpu.vram_used);
                        const pct = gpu.vram_used_percent || 0;
                        item.fill.style.width = Math.floor(pct) + "%";
                        item.el.dataset.state = getWarningState(pct);
                        // Peak tracking
                        const pk = this.peakVRAM[i] || 0;
                        if (gpu.vram_used > pk) this.peakVRAM[i] = gpu.vram_used;
                    }
                } else {
                    // GPU utilization
                    const gpuKey = "gpu" + idSuffix;
                    if (gpu.gpu_utilization != null) {
                        this.updateItem(gpuKey, gpu.gpu_utilization, "%");
                    } else {
                        this.hideItem(gpuKey);
                    }

                    // VRAM
                    const vramKey = "vram" + idSuffix;
                    const vramItem = this.items[vramKey];
                    if (vramItem && gpu.vram_used_percent != null) {
                        vramItem.valueEl.textContent = Math.floor(gpu.vram_used_percent) + "%";
                        vramItem.fill.style.width = Math.floor(gpu.vram_used_percent) + "%";
                        vramItem.el.dataset.state = getWarningState(gpu.vram_used_percent);
                        vramItem.el.dataset.hidden = "false";
                        // Peak tracking
                        const pk = this.peakVRAM[i] || 0;
                        if (gpu.vram_used > pk) this.peakVRAM[i] = gpu.vram_used;
                    } else {
                        this.hideItem(vramKey);
                    }

                    // Temperature
                    const tempKey = "temp" + idSuffix;
                    const tempItem = this.items[tempKey];
                    if (tempItem && gpu.gpu_temperature != null) {
                        tempItem.valueEl.textContent = Math.floor(gpu.gpu_temperature) + "°";
                        tempItem.fill.style.width = Math.min(100, Math.floor(gpu.gpu_temperature)) + "%";
                        tempItem.fill.style.backgroundColor = getTemperatureColor(gpu.gpu_temperature);
                        tempItem.el.dataset.state = gpu.gpu_temperature >= 90 ? "critical" : gpu.gpu_temperature >= 75 ? "warning" : "normal";
                        tempItem.el.dataset.hidden = "false";
                    } else {
                        this.hideItem(tempKey);
                    }
                }
            });
        }
    }

    updateItem(key, value, symbol) {
        const item = this.items[key];
        if (!item) return;

        if (value == null || value < 0) {
            this.hideItem(key);
            return;
        }

        item.el.dataset.hidden = "false";
        item.valueEl.textContent = Math.floor(value) + (symbol || "");
        item.fill.style.width = Math.min(100, Math.floor(value)) + "%";
        item.el.dataset.state = getWarningState(value);
    }

    hideItem(key) {
        const item = this.items[key];
        if (item) {
            item.el.dataset.hidden = "true";
        }
    }

    // -- Progress bar ------------------------------------------------------

    /**
     * Resolve the node type from a node ID using the ComfyUI graph.
     */
    resolveNodeType(nodeId) {
        if (!nodeId || !app.graph) return null;
        try {
            const node = app.graph.getNodeById(parseInt(nodeId));
            if (node) {
                return node.comfyClass || node.type || null;
            }
        } catch (e) { /* ignore */ }
        return null;
    }

    /**
     * Called when ComfyUI starts executing a node.
     */
    onNodeExecuting(nodeId) {
        this._currentNodeId = nodeId;
        if (!nodeId) return;

        const nodeType = this.resolveNodeType(nodeId);
        this._currentNodeType = nodeType;

        // Reset tile counter when a new node starts executing
        this._tileCount = 0;
        this._prevProgressValue = 0;

        if (nodeType && NODE_PROGRESS_LABELS[nodeType]) {
            this._currentNodeLabel = NODE_PROGRESS_LABELS[nodeType];
        } else if (nodeType) {
            // Non-UmeAiRT node — use generic "Sampling" for KSampler-like,
            // or "Processing" for others
            const lower = nodeType.toLowerCase();
            if (lower.includes("sampler") || lower.includes("ksampler")) {
                this._currentNodeLabel = "Sampling";
            } else {
                this._currentNodeLabel = "Processing";
            }
        } else {
            this._currentNodeLabel = "Processing";
        }
    }

    showProgress(value, max) {
        const item = this.items.progress;
        if (!item) return;

        // Detect tile reset: progress value drops significantly while same node executes
        // e.g. tile 1 finishes at 20/20, tile 2 starts at 1/20
        if (value < this._prevProgressValue && this._prevProgressValue > 0 && max > 0) {
            this._tileCount++;
        }
        this._prevProgressValue = value;

        this._progressValue = value;
        this._progressMax = max;

        item.el.dataset.active = "true";
        const pct = max > 0 ? Math.floor((value / max) * 100) : 0;

        // Build label: "Upscaling" or "Upscaling T2" for multi-tile
        let displayLabel = this._currentNodeLabel;
        if (this._tileCount > 0) {
            displayLabel += " T" + (this._tileCount + 1);
        }

        item.labelEl.textContent = displayLabel;
        item.valueEl.textContent = `${pct}%`;
        item.fill.style.width = pct + "%";
    }

    hideProgress() {
        const item = this.items.progress;
        if (!item) return;
        // Small delay to let the user see 100%
        setTimeout(() => {
            item.el.dataset.active = "false";
            item.fill.style.width = "0%";
            item.labelEl.textContent = "Idle";
            item.valueEl.textContent = "";
            this._currentNodeId = null;
            this._currentNodeType = null;
            this._currentNodeLabel = "Processing";
            this._tileCount = 0;
            this._prevProgressValue = 0;
        }, 1500);
    }

    showProgressTooltip(el) {
        const v = this._progressValue;
        const m = this._progressMax;
        const nodeType = this._currentNodeType || "Unknown";
        const label = this._currentNodeLabel;
        const pct = m > 0 ? Math.floor((v / m) * 100) : 0;

        let html = `<div class="umeairt-monitor__tooltip-title">${label}</div>`;
        if (this._tileCount > 0) {
            html += `<div class="umeairt-monitor__tooltip-row">Tile: <span>${this._tileCount + 1}</span></div>`;
        }
        html += `<div class="umeairt-monitor__tooltip-row">Step: <span>${v} / ${m}</span></div>`;
        html += `<div class="umeairt-monitor__tooltip-row">Progress: <span>${pct}%</span></div>`;
        html += `<div class="umeairt-monitor__tooltip-hint">${nodeType}</div>`;
        this.tooltip.show(el, html);
    }

    // -- Tooltips ----------------------------------------------------------

    showTooltip(el, key) {
        let html = "";
        const data = this._lastData;

        if (key === "cpu") {
            const val = data ? Math.floor(data.cpu_utilization) : "—";
            html = `<div class="umeairt-monitor__tooltip-title">CPU</div>`;
            html += `<div class="umeairt-monitor__tooltip-row">Usage: <span>${val}%</span></div>`;
        } else if (key === "ram") {
            if (data) {
                html = `<div class="umeairt-monitor__tooltip-title">System RAM</div>`;
                html += `<div class="umeairt-monitor__tooltip-row">Used: <span>${formatBytes(data.ram_used)}</span> / ${formatBytes(data.ram_total)}</div>`;
                html += `<div class="umeairt-monitor__tooltip-row">Usage: <span>${Math.floor(data.ram_used_percent)}%</span></div>`;
            }
        } else if (key.startsWith("gpu") || key.startsWith("vram") || key.startsWith("temp") || key.startsWith("mps")) {
            // Find GPU index
            const idxStr = key.replace(/^(gpu|vram|temp|mps)/, "");
            const idx = idxStr ? parseInt(idxStr) : 0;
            if (data && data.gpus && data.gpus[idx]) {
                const gpu = data.gpus[idx];
                html = `<div class="umeairt-monitor__tooltip-title">${gpu.gpu_name || "GPU"}</div>`;

                if (gpu.gpu_utilization != null) {
                    html += `<div class="umeairt-monitor__tooltip-row">GPU: <span>${Math.floor(gpu.gpu_utilization)}%</span></div>`;
                }
                if (gpu.vram_used != null && gpu.vram_total != null) {
                    html += `<div class="umeairt-monitor__tooltip-row">VRAM: <span>${formatBytes(gpu.vram_used)}</span> / ${formatBytes(gpu.vram_total)}</div>`;
                    const peak = this.peakVRAM[idx] || 0;
                    if (peak > 0) {
                        html += `<div class="umeairt-monitor__tooltip-row">Peak: <span>${formatBytes(peak)}</span></div>`;
                        html += `<div class="umeairt-monitor__tooltip-hint">Double-click VRAM to reset peak</div>`;
                    }
                }
                if (gpu.gpu_temperature != null) {
                    html += `<div class="umeairt-monitor__tooltip-row">Temperature: <span>${Math.floor(gpu.gpu_temperature)}°C</span></div>`;
                }
                if (data.device_type === "mps") {
                    html += `<div class="umeairt-monitor__tooltip-hint">Apple Silicon — Unified Memory</div>`;
                }
            }
        }

        if (html) {
            this.tooltip.show(el, html);
        }
    }

    // -- WebSocket listeners -----------------------------------------------

    registerListeners() {
        // Hardware monitoring data
        api.addEventListener("umeairt.monitor", (event) => {
            if (!event?.detail) return;
            this._lastData = event.detail;
            this.updateDisplay(event.detail);
        });

        // Track which node is currently executing (for contextual progress labels)
        api.addEventListener("executing", (event) => {
            const detail = event?.detail;
            // Extract node ID — ComfyUI may send {node: "id"} or just "id"
            const nodeId = detail?.node !== undefined ? detail.node : detail;
            if (!nodeId) {
                // null/undefined node = pipeline execution finished
                this.hideProgress();
                return;
            }
            this.onNodeExecuting(nodeId);
        });

        // Sampling progress (native ComfyUI events)
        api.addEventListener("progress", (event) => {
            if (!event?.detail) return;
            const { value, max } = event.detail;
            if (value != null && max != null) {
                this.showProgress(value, max);
            }
        });

        api.addEventListener("executed", () => {
            // Don't hide on individual node executed — wait for full pipeline finish
            // The 'executing' event with null node signals pipeline completion
        });

        api.addEventListener("execution_error", () => {
            this.hideProgress();
        });
    }

    // -- Setup (entry point) -----------------------------------------------

    async setup() {
        // Load stylesheet
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = new URL("umeairt_monitor.css", import.meta.url).href;
        document.head.appendChild(link);

        // Read current settings
        try {
            this.enabled = app.extensionManager.setting.get("UmeAiRT.Monitor.Enabled") ?? true;
            const styleName = app.extensionManager.setting.get("UmeAiRT.Monitor.Style") ?? "Micro Gauges";
            this.currentStyle = STYLE_MAP[styleName] || "gauges";
        } catch (e) {
            // Settings may not be ready yet
        }

        // Register settings
        this.registerSettings();

        // Create root container
        this.rootElement = this.createRootElement();

        // Insert into top bar
        try {
            if (app.menu?.settingsGroup?.element) {
                app.menu.settingsGroup.element.before(this.rootElement);
            } else {
                // Fallback: insert after queue button
                const queueBtn = document.getElementById("queue-button");
                if (queueBtn) {
                    queueBtn.insertAdjacentElement("afterend", this.rootElement);
                }
            }
        } catch (e) {
            console.warn("[UmeAiRT Monitor] Could not insert into top bar:", e);
        }

        // Build monitoring items
        await this.buildMonitorBar();

        // Apply initial enabled state
        if (!this.enabled && this.rootElement) {
            this.rootElement.style.display = "none";
        }

        // Register WebSocket listeners
        this.registerListeners();
    }
}

// ---------------------------------------------------------------------------
// Register extension
// ---------------------------------------------------------------------------

const monitor = new UmeAiRTMonitor();

app.registerExtension({
    name: monitor.idExtensionName,
    setup: () => monitor.setup(),
});
