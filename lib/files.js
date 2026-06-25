import fs from "node:fs/promises";
import path from "node:path";

export async function ensureDirectories(config) {
  await Promise.all([
    fs.mkdir(config.outputDir, { recursive: true }),
    fs.mkdir(config.localModelDir, { recursive: true })
  ]);
}

export async function readJson(file, fallback) {
  try {
    return JSON.parse(await fs.readFile(file, "utf8"));
  } catch (error) {
    if (error.code === "ENOENT") return structuredClone(fallback);
    throw error;
  }
}

export async function writeJsonAtomic(file, value) {
  const temporary = `${file}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });
  await fs.rename(temporary, file);
}

export function safeFilename(value) {
  const name = path.basename(String(value || "file"));
  return name.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^\.+/, "") || "file";
}
