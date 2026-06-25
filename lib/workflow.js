const fieldNames = new Set([
  "text", "seed", "noise_seed", "steps", "cfg", "sampler_name", "scheduler",
  "width", "height", "batch_size", "ckpt_name", "lora_name", "vae_name",
  "control_net_name", "unet_name", "clip_name", "noise_seed", "guidance", "denoise"
]);

export function validateWorkflow(workflow) {
  if (!workflow || Array.isArray(workflow) || typeof workflow !== "object") {
    throw new Error("Workflow must be a ComfyUI API-format object");
  }
  const entries = Object.entries(workflow);
  if (!entries.length) throw new Error("Workflow contains no nodes");
  for (const [id, node] of entries) {
    if (!node || typeof node !== "object" || typeof node.class_type !== "string") {
      throw new Error(`Node ${id} is not in ComfyUI API format; use Export (API)`);
    }
    if (!node.inputs || typeof node.inputs !== "object") {
      throw new Error(`Node ${id} has no inputs object`);
    }
  }
  return workflow;
}

export function inspectWorkflow(workflow) {
  validateWorkflow(workflow);
  const fields = [];
  const models = [];

  for (const [nodeId, node] of Object.entries(workflow)) {
    for (const [key, value] of Object.entries(node.inputs)) {
      if (!fieldNames.has(key) || (typeof value !== "string" && typeof value !== "number")) continue;
      const field = {
        nodeId,
        classType: node.class_type,
        title: node._meta?.title || node.class_type,
        key,
        value
      };
      fields.push(field);
      if (key.endsWith("_name") && key !== "sampler_name") models.push(field);
    }
  }

  return { nodeCount: Object.keys(workflow).length, fields, models };
}

export function setWorkflowInput(workflow, nodeId, key, value) {
  validateWorkflow(workflow);
  if (!workflow[nodeId]) throw new Error(`Unknown workflow node ${nodeId}`);
  if (!(key in workflow[nodeId].inputs)) throw new Error(`Node ${nodeId} has no input ${key}`);
  const copy = structuredClone(workflow);
  copy[nodeId].inputs[key] = value;
  return copy;
}

export function validateGenerationLimits(workflow, limits) {
  const { fields } = inspectWorkflow(workflow);
  const checks = {
    width: limits.maxWidth,
    height: limits.maxHeight,
    steps: limits.maxSteps,
    batch_size: limits.maxBatchSize
  };
  for (const field of fields) {
    const maximum = checks[field.key];
    if (maximum && Number(field.value) > maximum) {
      throw new Error(`${field.key} on node ${field.nodeId} exceeds limit ${maximum}`);
    }
  }
}
