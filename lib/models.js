import crypto from "node:crypto";
import { constants } from "node:fs";
import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { readJson, safeFilename, writeJsonAtomic } from "./files.js";

const allowedTypes = new Set([
  "checkpoints", "loras", "vae", "controlnet", "clip", "clip_vision",
  "diffusion_models", "text_encoders", "upscale_models"
]);

export async function listModels(config) {
  const manifest = await readJson(config.manifestPath, { version: 1, models: [] });
  return manifest.models;
}

export async function importLocalModel(config, sourcePath, modelType) {
  if (!allowedTypes.has(modelType)) throw new Error(`Unsupported model type: ${modelType}`);
  const source = path.resolve(sourcePath);
  const stat = await fs.stat(source);
  if (!stat.isFile()) throw new Error("Model source must be a file");

  const filename = safeFilename(path.basename(source));
  const targetDirectory = path.join(config.localModelDir, modelType);
  const target = path.join(targetDirectory, filename);
  await fs.mkdir(targetDirectory, { recursive: true });

  const hash = crypto.createHash("sha256");
  for await (const chunk of createReadStream(source)) hash.update(chunk);
  const sha256 = hash.digest("hex");

  if (source !== target) await fs.copyFile(source, target, constants.COPYFILE_EXCL);
  const manifest = await readJson(config.manifestPath, { version: 1, models: [] });
  const duplicate = manifest.models.find((model) => model.sha256 === sha256);
  if (duplicate) {
    if (source !== target) await fs.rm(target, { force: true });
    return { model: duplicate, duplicate: true };
  }

  const model = {
    id: crypto.randomUUID(),
    source: "local",
    sourceUrl: null,
    filename,
    modelType,
    localPath: path.relative(config.root, target),
    sizeBytes: stat.size,
    sha256,
    installedAt: new Date().toISOString()
  };
  manifest.models.push(model);
  await writeJsonAtomic(config.manifestPath, manifest);
  return { model, duplicate: false };
}
