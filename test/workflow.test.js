import test from "node:test";
import assert from "node:assert/strict";
import { inspectWorkflow, setWorkflowInput, validateGenerationLimits, validateWorkflow } from "../lib/workflow.js";

const workflow = {
  "1": {
    class_type: "CheckpointLoaderSimple",
    inputs: { ckpt_name: "model.safetensors" },
    _meta: { title: "Load Checkpoint" }
  },
  "2": {
    class_type: "KSampler",
    inputs: { seed: 1, steps: 20, cfg: 7, sampler_name: "euler", scheduler: "normal" }
  }
};

test("validates and inspects API workflows", () => {
  assert.equal(validateWorkflow(workflow), workflow);
  const inspected = inspectWorkflow(workflow);
  assert.equal(inspected.nodeCount, 2);
  assert.equal(inspected.models[0].key, "ckpt_name");
});

test("updates only the selected input", () => {
  const updated = setWorkflowInput(workflow, "2", "steps", 30);
  assert.equal(updated["2"].inputs.steps, 30);
  assert.equal(workflow["2"].inputs.steps, 20);
});

test("enforces generation limits", () => {
  assert.throws(
    () => validateGenerationLimits(setWorkflowInput(workflow, "2", "steps", 100), { maxSteps: 60 }),
    /exceeds limit/
  );
});

test("rejects editor-format workflows", () => {
  assert.throws(() => validateWorkflow({ nodes: [] }), /Export \(API\)/);
});
