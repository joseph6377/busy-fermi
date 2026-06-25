import "dotenv/config";
import express from "express";
import cors from "cors";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadConfig, publicConfig } from "./lib/config.js";
import { ensureDirectories } from "./lib/files.js";
import { importLocalModel, listModels } from "./lib/models.js";
import { addMockJob, endMockSession, getSession, startMockSession } from "./lib/session.js";
import { inspectWorkflow, validateGenerationLimits, validateWorkflow } from "./lib/workflow.js";
import { cancelJob, getJob, submitJob } from "./lib/runpod.js";
import { saveJobOutputs } from "./lib/outputs.js";

const root = path.dirname(fileURLToPath(import.meta.url));
const config = loadConfig(root);
await ensureDirectories(config);

const app = express();
app.disable("x-powered-by");
app.use(cors({ origin: /^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/ }));
app.use(express.json({ limit: config.maxRequestBytes }));
app.use(express.static(path.join(root, "public")));
app.use("/outputs", express.static(config.outputDir));

app.get("/api/config", (_request, response) => response.json(publicConfig(config)));
app.get("/api/models", async (_request, response) => response.json({ models: await listModels(config) }));
app.post("/api/models/import-local", async (request, response) => {
  const result = await importLocalModel(config, request.body.sourcePath, request.body.modelType);
  response.status(result.duplicate ? 200 : 201).json(result);
});
app.post("/api/workflows/inspect", (request, response) => {
  response.json(inspectWorkflow(validateWorkflow(request.body.workflow)));
});
app.get("/api/cloud-session", async (_request, response) => response.json(await getSession(config)));
app.post("/api/cloud-session", async (request, response) => {
  if (!config.mockRunpod) throw new Error("Real RunPod session creation is not enabled in this MVP");
  const allModels = await listModels(config);
  const ids = new Set(request.body.modelIds || []);
  const selected = allModels.filter((model) => ids.has(model.id));
  if (selected.length !== ids.size) throw new Error("One or more selected models are missing");
  response.status(201).json(await startMockSession(config, selected));
});
app.delete("/api/cloud-session", async (_request, response) => {
  if (!config.mockRunpod) throw new Error("Real RunPod session deletion is not enabled in this MVP");
  response.json(await endMockSession(config));
});
app.post("/api/jobs", async (request, response) => {
  validateWorkflow(request.body.workflow);
  validateGenerationLimits(request.body.workflow, config.limits);
  const size = Buffer.byteLength(JSON.stringify(request.body));
  if (size > config.maxRequestBytes) throw new Error("Generation request exceeds configured size limit");
  if (config.mockRunpod) {
    response.status(202).json(await addMockJob(config));
    return;
  }
  const job = await submitJob(config, {
    workflow: request.body.workflow,
    images: request.body.images || []
  });
  response.status(202).json(job);
});
app.get("/api/jobs/:id", async (request, response) => {
  if (config.mockRunpod) {
    const session = await getSession(config);
    const job = session.jobs?.find((candidate) => candidate.id === request.params.id);
    if (!job) throw new Error("Mock job not found");
    response.json(job);
    return;
  }
  const job = await getJob(config, request.params.id);
  if (job.status === "COMPLETED" && Array.isArray(job.output?.images)) {
    job.saved = await saveJobOutputs(config, job);
  }
  response.json(job);
});
app.post("/api/jobs/:id/cancel", async (request, response) => {
  if (config.mockRunpod) {
    response.json({ id: request.params.id, status: "CANCELLED" });
    return;
  }
  response.json(await cancelJob(config, request.params.id));
});

app.use((error, _request, response, _next) => {
  const status = error.type === "entity.too.large" ? 413 : 400;
  console.error(error.message);
  response.status(status).json({ error: error.message });
});

app.listen(config.port, config.host, () => {
  console.log(`Comfy RunPod UI: http://${config.host}:${config.port}`);
  console.log(`Mode: ${config.mockRunpod ? "mock (no paid calls)" : "real"}`);
});
