import fs from "node:fs/promises";
import path from "node:path";
import { safeFilename, writeJsonAtomic } from "./files.js";

const mimeExtensions = {
  "image/png": ".png",
  "image/jpeg": ".jpg",
  "image/webp": ".webp"
};

function detectMimeType(buffer) {
  if (buffer.length >= 4 && buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47) {
    return "image/png";
  }
  if (buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return "image/jpeg";
  }
  if (buffer.length >= 12 && buffer.toString("ascii", 0, 4) === "RIFF" && buffer.toString("ascii", 8, 12) === "WEBP") {
    return "image/webp";
  }
  return null;
}

function parseDataUrl(value) {
  const match = /^data:(image\/(?:png|jpeg|webp));base64,([a-zA-Z0-9+/=\s]+)$/.exec(value);
  if (!match) return null;
  return { mime: match[1], bytes: Buffer.from(match[2].replace(/\s/g, ""), "base64") };
}

function parseRawBase64(value, filenameHint = "") {
  const cleanStr = value.replace(/\s/g, "");
  if (!/^[a-zA-Z0-9+/=]+$/.test(cleanStr)) {
    return null;
  }
  try {
    const bytes = Buffer.from(cleanStr, "base64");
    if (bytes.length === 0) return null;

    let mime = detectMimeType(bytes);
    if (!mime && filenameHint) {
      const ext = path.extname(filenameHint).toLowerCase();
      if (ext === ".png") mime = "image/png";
      else if (ext === ".jpg" || ext === ".jpeg") mime = "image/jpeg";
      else if (ext === ".webp") mime = "image/webp";
    }
    if (!mime) {
      mime = "image/png";
    }
    return { mime, bytes };
  } catch {
    return null;
  }
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
    const suppliedName = typeof image === "object" ? image.filename || image.name : "";
    let parsed = typeof raw === "string" ? parseDataUrl(raw) : null;
    if (!parsed && typeof raw === "string") {
      parsed = parseRawBase64(raw, suppliedName);
    }
    if (!parsed) continue;
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
