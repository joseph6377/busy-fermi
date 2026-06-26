import { app } from "../../scripts/app.js";

/**
 * UmeAiRT LoRA Info Extension
 * ---------------------------
 * Adds context menu entries on LoRA nodes to:
 *   - Show full LoRA metadata (base model, triggers, network info)
 *   - Copy trigger words to clipboard
 *
 * Compatible with ComfyUI Nodes 2.0 (Vue frontend).
 */

const _infoCache = new Map();

app.registerExtension({
    name: "UmeAiRT.LoraInfo",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData.name || !nodeData.name.startsWith("UmeAiRT_LoraBlock_")) {
            return;
        }

        // Pre-fetch LoRA info when a node is created/loaded
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            // Background pre-fetch for all set LoRAs
            for (const w of this.widgets || []) {
                if (w.name?.endsWith("_name") && w.value && w.value !== "None") {
                    _prefetch(w.value, w);
                }
            }
            // Also pre-fetch on combo change
            for (const w of this.widgets || []) {
                if (w.name?.endsWith("_name")) {
                    const orig = w.callback;
                    const widget = w;
                    w.callback = function (value) {
                        if (orig) orig.apply(this, arguments);
                        if (value && value !== "None") _prefetch(value, widget);
                    };
                }
            }
        };

        // Pre-fetch on workflow load
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            for (const w of this.widgets || []) {
                if (w.name?.endsWith("_name") && w.value && w.value !== "None") {
                    _prefetch(w.value, w);
                }
            }
        };

        // Context menu entries
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            if (origGetExtraMenuOptions) {
                origGetExtraMenuOptions.apply(this, arguments);
            }

            const loraWidgets = (this.widgets || []).filter(
                (w) => w.name?.endsWith("_name") && w.value && w.value !== "None"
            );

            if (loraWidgets.length === 0) return;

            options.push(null); // separator

            for (const w of loraWidgets) {
                const loraName = w.value;
                const slot = w.name.match(/lora_(\d+)_name/);
                const label = slot ? `LoRA ${slot[1]}` : "LoRA";

                options.push({
                    content: `ℹ️ ${label} Info`,
                    callback: () => _showInfo(loraName, label),
                });
                options.push({
                    content: `📋 ${label} — Copy Trigger Words`,
                    callback: () => _copyTriggers(loraName, label),
                });
            }
        };
    },
});


/** Background pre-fetch into cache + set tooltip (visible on re-render/R key) */
async function _prefetch(loraName, widget) {
    if (_infoCache.has(loraName)) {
        if (widget) widget.tooltip = _formatTooltip(_infoCache.get(loraName));
        return;
    }
    try {
        const r = await fetch(`/umeairt/lora-info?filename=${encodeURIComponent(loraName)}`);
        if (r.ok) {
            const data = await r.json();
            _infoCache.set(loraName, data);
            if (widget) widget.tooltip = _formatTooltip(data);
        }
    } catch { /* silent */ }
}

function _formatTooltip(data) {
    const triggers = data.trigger_words?.length
        ? data.trigger_words.join(", ")
        : "(none)";
    return [
        `📂 ${data.filename}`,
        `Base: ${data.base_model}  |  Size: ${data.file_size_mb} MB`,
        `Network: dim=${data.network_dim || "?"} / alpha=${data.network_alpha || "?"}`,
        data.resolution ? `Resolution: ${data.resolution}` : null,
        `🏷️ Triggers: ${triggers}`,
    ].filter(Boolean).join("\n");
}


/** Fetch (or use cache) and return data */
async function _getData(loraName) {
    if (_infoCache.has(loraName)) return _infoCache.get(loraName);
    const r = await fetch(`/umeairt/lora-info?filename=${encodeURIComponent(loraName)}`);
    if (!r.ok) return null;
    const data = await r.json();
    _infoCache.set(loraName, data);
    return data;
}


/** Show LoRA info dialog */
async function _showInfo(loraName, label) {
    const data = await _getData(loraName);
    if (!data) {
        alert(`⚠️ Could not load info for ${loraName}`);
        return;
    }

    const triggers = data.trigger_words?.length
        ? data.trigger_words.join(", ")
        : "(none found in metadata)";

    const lines = [
        `━━━ ${label} ━━━`,
        ``,
        `📂  ${data.filename}`,
        `📐  Base Model: ${data.base_model}`,
        `💾  Size: ${data.file_size_mb} MB`,
        `🔧  Network: dim=${data.network_dim || "?"} / alpha=${data.network_alpha || "?"}`,
    ];

    if (data.resolution) lines.push(`📏  Resolution: ${data.resolution}`);
    if (data.training_comment) lines.push(`💬  ${data.training_comment}`);

    lines.push(``);
    lines.push(`🏷️  Trigger Words:`);
    lines.push(triggers);

    if (data.trigger_words?.length) {
        lines.push(``);
        try {
            await navigator.clipboard.writeText(data.trigger_words.join(", "));
            lines.push(`✅ Trigger words copied to clipboard!`);
        } catch {
            lines.push(`(select and copy manually)`);
        }
    }

    alert(lines.join("\n"));
}


/** Copy trigger words only */
async function _copyTriggers(loraName, label) {
    const data = await _getData(loraName);
    if (!data) {
        alert(`⚠️ Could not load info for ${loraName}`);
        return;
    }

    if (!data.trigger_words?.length) {
        alert(`${label}: No trigger words found in metadata.`);
        return;
    }

    const text = data.trigger_words.join(", ");
    try {
        await navigator.clipboard.writeText(text);
        alert(`✅ ${label} — Trigger words copied!\n\n${text}`);
    } catch {
        alert(`${label} — Copy manually:\n\n${text}`);
    }
}
