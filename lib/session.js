import crypto from "node:crypto";
import fs from "node:fs/promises";
import { readJson, writeJsonAtomic } from "./files.js";

const empty = { active: false };

export async function getSession(config) {
  return readJson(config.sessionPath, empty);
}

export async function startMockSession(config, models) {
  const current = await getSession(config);
  if (current.active) throw new Error("A cloud session is already active");
  const requiredBytes = models.reduce((sum, model) => sum + Number(model.sizeBytes || 0), 0);
  const session = {
    active: true,
    mock: true,
    id: crypto.randomUUID(),
    volumeId: `mock-volume-${Date.now()}`,
    ownedByApplication: true,
    createdAt: new Date().toISOString(),
    modelIds: models.map((model) => model.id),
    requiredBytes,
    provisionedBytes: requiredBytes + config.tempVolumeSafetyBytes,
    jobs: []
  };
  await writeJsonAtomic(config.sessionPath, session);
  return session;
}

export async function endMockSession(config) {
  const session = await getSession(config);
  if (!session.active) return empty;
  if (!session.mock || !session.ownedByApplication) {
    throw new Error("Refusing to delete a volume not created by this application");
  }
  if (session.jobs.some((job) => ["IN_QUEUE", "IN_PROGRESS"].includes(job.status))) {
    throw new Error("Cannot end session while a job is active");
  }
  await fs.rm(config.sessionPath, { force: true });
  return empty;
}

export async function addMockJob(config) {
  const session = await getSession(config);
  if (!session.active) throw new Error("Start a cloud session first");
  const job = { id: `mock-job-${Date.now()}`, status: "COMPLETED", createdAt: new Date().toISOString() };
  session.jobs.push(job);
  await writeJsonAtomic(config.sessionPath, session);
  return job;
}
