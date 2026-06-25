import "dotenv/config";
import fs from "node:fs/promises";
import { createWriteStream } from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { loadConfig } from "../lib/config.js";
import { ensureDirectories } from "../lib/files.js";
import { importLocalModel } from "../lib/models.js";

const config = loadConfig(path.resolve(path.dirname(new URL(import.meta.url).pathname), ".."));
const token = process.env.HF_TOKEN?.trim();
if (!token) throw new Error("HF_TOKEN is missing");

const files = [
  {
    modelType: "diffusion_models",
    filename: "flux-2-klein-9b-fp8.safetensors",
    url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors"
  },
  {
    modelType: "text_encoders",
    filename: "qwen_3_8b_fp8mixed.safetensors",
    url: "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"
  },
  {
    modelType: "vae",
    filename: "flux2-vae.safetensors",
    url: "https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/main/split_files/vae/flux2-vae.safetensors"
  }
];

await ensureDirectories(config);

for (const file of files) {
  const directory = path.join(config.localModelDir, file.modelType);
  const finalPath = path.join(directory, file.filename);
  const partialPath = `${finalPath}.part`;
  await fs.mkdir(directory, { recursive: true });

  try {
    await fs.access(finalPath);
    console.log(`${file.filename}: already downloaded`);
    await importLocalModel(config, finalPath, file.modelType);
    continue;
  } catch {}

  let offset = 0;
  try {
    offset = (await fs.stat(partialPath)).size;
  } catch {}

  const headers = { authorization: `Bearer ${token}` };
  if (offset) headers.range = `bytes=${offset}-`;
  console.log(`${file.filename}: ${offset ? `resuming at ${(offset / 1024 ** 3).toFixed(2)} GiB` : "starting"}`);

  const response = await fetch(file.url, { headers, redirect: "follow" });
  if (!response.ok && response.status !== 206) {
    throw new Error(`${file.filename}: download failed (${response.status})`);
  }
  if (offset && response.status !== 206) {
    offset = 0;
    await fs.rm(partialPath, { force: true });
  }

  const expectedRemaining = Number(response.headers.get("content-length") || 0);
  const stream = createWriteStream(partialPath, { flags: offset ? "a" : "w" });
  await pipeline(Readable.fromWeb(response.body), stream);
  const finalSize = (await fs.stat(partialPath)).size;
  if (expectedRemaining && finalSize !== offset + expectedRemaining) {
    throw new Error(`${file.filename}: incomplete download`);
  }

  await fs.rename(partialPath, finalPath);
  console.log(`${file.filename}: downloaded ${(finalSize / 1024 ** 3).toFixed(2)} GiB; hashing`);
  await importLocalModel(config, finalPath, file.modelType);
  console.log(`${file.filename}: verified and registered`);
}

console.log("FLUX.2 Klein 9B model set is ready locally.");
