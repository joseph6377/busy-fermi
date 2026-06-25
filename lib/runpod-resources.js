import fs from "node:fs";
import path from "node:path";
import { S3Client, HeadObjectCommand } from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";
import { getSession } from "./session.js";
import { writeJsonAtomic } from "./files.js";

async function rest(config, pathname, { method = "GET", body } = {}) {
  const response = await fetch(`https://rest.runpod.io/v1${pathname}`, {
    method,
    headers: {
      authorization: `Bearer ${config.runpodApiKey}`,
      ...(body ? { "content-type": "application/json" } : {})
    },
    body: body ? JSON.stringify(body) : undefined
  });
  if (response.status === 204) return {};
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.message || `RunPod REST request failed (${response.status})`);
  return data;
}

export function getEndpointResource(config) {
  return rest(config, `/endpoints/${encodeURIComponent(config.runpodEndpointId)}`);
}

export function createNetworkVolume(config, { dataCenterId, name, size }) {
  return rest(config, "/networkvolumes", {
    method: "POST",
    body: { dataCenterId, name, size }
  });
}

export function attachNetworkVolume(config, volumeId) {
  return rest(config, `/endpoints/${encodeURIComponent(config.runpodEndpointId)}`, {
    method: "PATCH",
    body: { networkVolumeId: volumeId, networkVolumeIds: [volumeId] }
  });
}

export function detachNetworkVolume(config) {
  return rest(config, `/endpoints/${encodeURIComponent(config.runpodEndpointId)}`, {
    method: "PATCH",
    body: { networkVolumeId: "", networkVolumeIds: [] }
  });
}

export function deleteNetworkVolume(config, volumeId) {
  return rest(config, `/networkvolumes/${encodeURIComponent(volumeId)}`, { method: "DELETE" });
}

function s3Client(config, dataCenterId) {
  if (!config.s3AccessKeyId || !config.s3SecretAccessKey) {
    throw new Error("RunPod S3 API credentials are required");
  }
  return new S3Client({
    endpoint: config.s3Endpoint || `https://s3api-${dataCenterId.toLowerCase()}.runpod.io/`,
    region: dataCenterId,
    forcePathStyle: true,
    credentials: {
      accessKeyId: config.s3AccessKeyId,
      secretAccessKey: config.s3SecretAccessKey
    }
  });
}

export async function uploadModel(config, volume, model, onProgress = () => {}) {
  const client = s3Client(config, volume.dataCenterId);
  const key = `models/${model.modelType}/${model.filename}`;
  const localPath = path.resolve(config.root, model.localPath);
  const upload = new Upload({
    client,
    params: { Bucket: volume.id, Key: key, Body: fs.createReadStream(localPath) },
    queueSize: 3,
    partSize: 64 * 1024 * 1024,
    leavePartsOnError: false
  });
  upload.on("httpUploadProgress", (progress) => onProgress({
    modelId: model.id,
    loaded: progress.loaded || 0,
    total: progress.total || model.sizeBytes
  }));
  await upload.done();
  const head = await client.send(new HeadObjectCommand({ Bucket: volume.id, Key: key }));
  if (Number(head.ContentLength) !== Number(model.sizeBytes)) {
    throw new Error(`Uploaded size mismatch for ${model.filename}`);
  }
  return { key, sizeBytes: Number(head.ContentLength) };
}

export async function endOwnedCloudSession(config) {
  const session = await getSession(config);
  if (!session.active) return { active: false };
  if (!session.ownedByApplication || !session.volumeId) {
    throw new Error("Refusing to delete a volume not created by this application");
  }
  if (session.jobs?.some((job) => ["IN_QUEUE", "IN_PROGRESS"].includes(job.status))) {
    throw new Error("Cannot delete cloud storage while a job is active");
  }
  await detachNetworkVolume(config);
  await deleteNetworkVolume(config, session.volumeId);
  await writeJsonAtomic(config.sessionPath, { active: false });
  return { active: false };
}
