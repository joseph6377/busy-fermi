import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { saveJobOutputs } from "../lib/outputs.js";

test("saves all valid base64 image outputs and metadata", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "comfy-runpod-"));
  const config = { root, outputDir: path.join(root, "outputs") };
  const onePixelPng = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
  const saved = await saveJobOutputs(config, {
    id: "job-1",
    status: "COMPLETED",
    output: {
      images: [
        { filename: "result.png", image: `data:image/png;base64,${onePixelPng}` },
        { filename: "../../escape.png", image: `data:image/png;base64,${onePixelPng}` }
      ]
    }
  });
  assert.equal(saved.length, 2);
  assert.ok(saved.every((item) => item.path.startsWith("outputs/")));
  assert.ok(saved.every((item) => !item.filename.includes("..")));
  await fs.rm(root, { recursive: true, force: true });
});

test("saves base64 video outputs and metadata", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "comfy-runpod-"));
  const config = { root, outputDir: path.join(root, "outputs") };
  // Mock base64 video payload
  const mockVideoBase64 = "AAAAIGZ0eXBtcDQyAAAAAG1wNDJpc29tYXZjMQAAAAhkYXRhAAAAAA==";
  const saved = await saveJobOutputs(config, {
    id: "job-2",
    status: "COMPLETED",
    output: {
      images: [
        { filename: "result.mp4", image: `data:video/mp4;base64,${mockVideoBase64}` }
      ]
    }
  });
  assert.equal(saved.length, 1);
  assert.equal(saved[0].filename.endsWith(".mp4"), true);
  await fs.rm(root, { recursive: true, force: true });
});
