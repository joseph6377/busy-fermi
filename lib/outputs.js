import fs from "node:fs/promises";
import path from "node:path";
import { safeFilename, writeJsonAtomic } from "./files.js";

const mimeExtensions = {
  "image/png": ".png",
  "image/jpeg": ".jpg",
  "image/webp": ".webp"
};

function parseDataUrl(value) {
  const match = /^data:(image\/(?:png|jpeg|webp));base64,([a-zA-Z0-9+/=\s]+)$/.exec(value);
  if (!match) return null;
  return { mime: match[1], bytes: Buffer.from(match[2].replace(/\s/g, ""), "base64") };
}

function dateDirectory(date = new Date()) {
  return date.toISOString().slice(0, 10);
}

export async function saveJobOutputs(config, job) {
  const images = Array.isArray(job.output?.images) ? job.output.images : [];
  const directory = path.join(config.outputDir, dateDirectory());
  await fs.mkdir(directory, { recursive: true });
  const saved = [];

  for (const [index, image] of images.entries()) {
    const raw = typeof image === "string" ? image : image?.image || image?.data;
    const parsed = typeof raw === "string" ? parseDataUrl(raw) : null;
    if (!parsed) continue;
    const suppliedName = typeof image === "object" ? image.filename || image.name : "";
    const extension = mimeExtensions[parsed.mime];
    const base = safeFilename(suppliedName || `output${extension}`).replace(/\.[^.]+$/, "");
    const filename = safeFilename(`${job.id}_${index}_${base}${extension}`);
    const destination = path.join(directory, filename);
    try {
      await fs.writeFile(destination, parsed.bytes, { flag: "wx" });
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
    }
    saved.push({
      filename,
      path: path.relative(config.root, destination),
      url: `/outputs/${dateDirectory()}/${encodeURIComponent(filename)}`,
      sizeBytes: parsed.bytes.length
    });
  }

  const metadata = {
    jobId: job.id,
    status: job.status,
    delayTime: job.delayTime,
    executionTime: job.executionTime,
    saved,
    savedAt: new Date().toISOString()
  };
  await writeJsonAtomic(path.join(directory, `${safeFilename(job.id)}.json`), metadata);
  return saved;
}
