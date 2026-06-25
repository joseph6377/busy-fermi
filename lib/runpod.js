const terminalStates = new Set(["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]);

function endpointUrl(config, operation, jobId) {
  const suffix = jobId ? `/${operation}/${encodeURIComponent(jobId)}` : `/${operation}`;
  return `https://api.runpod.ai/v2/${config.runpodEndpointId}${suffix}`;
}

async function request(config, operation, { method = "GET", jobId, body } = {}) {
  if (!config.runpodApiKey || !config.runpodEndpointId) {
    throw new Error("RunPod API key and endpoint ID are required");
  }
  const response = await fetch(endpointUrl(config, operation, jobId), {
    method,
    headers: {
      authorization: `Bearer ${config.runpodApiKey}`,
      ...(body ? { "content-type": "application/json" } : {})
    },
    body: body ? JSON.stringify(body) : undefined
  });
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`RunPod returned a non-JSON response (${response.status})`);
  }
  if (!response.ok) {
    throw new Error(data.error || data.message || `RunPod request failed (${response.status})`);
  }
  return data;
}

export function submitJob(config, input) {
  return request(config, "run", { method: "POST", body: { input } });
}

export function getJob(config, jobId) {
  return request(config, "status", { jobId });
}

export function cancelJob(config, jobId) {
  return request(config, "cancel", { method: "POST", jobId });
}

export function isTerminalStatus(status) {
  return terminalStates.has(status);
}
