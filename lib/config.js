import path from "node:path";

const number = (name, fallback) => {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isFinite(value)) throw new Error(`${name} must be a number`);
  return value;
};

const boolean = (name, fallback) => {
  const value = String(process.env[name] ?? fallback).toLowerCase();
  return ["1", "true", "yes", "on"].includes(value);
};

export function loadConfig(root = process.cwd()) {
  return {
    root,
    host: process.env.HOST || "127.0.0.1",
    port: number("PORT", 3000),
    mockRunpod: boolean("MOCK_RUNPOD", true),
    runpodApiKey: process.env.RUNPOD_API_KEY || "",
    runpodEndpointId: process.env.RUNPOD_ENDPOINT_ID || "",
    endpoints: {
      flux: process.env.RUNPOD_ENDPOINT_FLUX || process.env.RUNPOD_ENDPOINT_ID || "",
      flux1: process.env.RUNPOD_ENDPOINT_FLUX1 || process.env.RUNPOD_ENDPOINT_FLUX || process.env.RUNPOD_ENDPOINT_ID || "",
      flux2: process.env.RUNPOD_ENDPOINT_FLUX2 || process.env.RUNPOD_ENDPOINT_FLUX || process.env.RUNPOD_ENDPOINT_ID || "",
      sdxl: process.env.RUNPOD_ENDPOINT_SDXL || ""
    },
    runpodDataCenterId: process.env.RUNPOD_DATACENTER_ID || "",
    s3Endpoint: process.env.RUNPOD_S3_ENDPOINT || "",
    s3AccessKeyId: process.env.RUNPOD_S3_ACCESS_KEY_ID || "",
    s3SecretAccessKey: process.env.RUNPOD_S3_SECRET_ACCESS_KEY || "",
    outputDir: path.resolve(root, process.env.OUTPUT_DIR || "outputs"),
    localModelDir: path.resolve(root, process.env.LOCAL_MODEL_DIR || "local_models"),
    manifestPath: path.resolve(root, process.env.MODEL_MANIFEST || "model_manifest.json"),
    sessionPath: path.resolve(root, process.env.SESSION_STATE || "session_state.json"),
    maxRequestBytes: number("MAX_REQUEST_MB", 8) * 1024 * 1024,
    limits: {
      maxWidth: number("MAX_WIDTH", 2048),
      maxHeight: number("MAX_HEIGHT", 2048),
      maxSteps: number("MAX_STEPS", 60),
      maxBatchSize: number("MAX_BATCH_SIZE", 1)
    },
    tempVolumeSafetyBytes: number("TEMP_VOLUME_SAFETY_GB", 2) * 1024 ** 3,
    autoDeleteVolume: boolean("AUTO_DELETE_VOLUME_AFTER_SUCCESS", true)
  };
}

export function publicConfig(config) {
  const hasEndpoint = Boolean(config.runpodEndpointId || config.endpoints.flux || config.endpoints.sdxl);
  return {
    ready: config.mockRunpod || Boolean(config.runpodApiKey && hasEndpoint),
    mockRunpod: config.mockRunpod,
    endpointConfigured: hasEndpoint,
    maxRequestMB: Math.round(config.maxRequestBytes / 1024 / 1024),
    limits: config.limits,
    autoDeleteVolume: config.autoDeleteVolume,
    endpoints: config.endpoints
  };
}
