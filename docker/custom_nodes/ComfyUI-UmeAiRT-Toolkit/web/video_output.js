import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "UmeAiRT.VideoOutput",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UmeAiRT_VideoOutput") {
            // Override onResize to completely bypass ComfyUI's native aspect-ratio locking
            nodeType.prototype.onResize = function (size) {
                // Simply accept the new size without enforcing any ratio
                this.size[0] = size[0];
                this.size[1] = size[1];
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                // Removed forced height resizing so the user's manual layout is preserved
            };
        }
    }
});
