const terminalStates = new Set(["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]);

function endpointUrl(config, operation, jobId, endpointId) {
  const activeId = endpointId || config.runpodEndpointId;
  const suffix = jobId ? `/${operation}/${encodeURIComponent(jobId)}` : `/${operation}`;
  return `https://api.runpod.ai/v2/${activeId}${suffix}`;
}

async function request(config, operation, { method = "GET", jobId, body, endpointId } = {}) {
  const activeId = endpointId || config.runpodEndpointId;
  if (!config.runpodApiKey || !activeId) {
    throw new Error("RunPod API key and endpoint ID are required");
  }
  const response = await fetch(endpointUrl(config, operation, jobId, endpointId), {
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

export function submitJob(config, input, endpointId) {
  return request(config, "run", { method: "POST", body: { input }, endpointId });
}

export function getJob(config, jobId, endpointId) {
  return request(config, "status", { jobId, endpointId });
}

export function cancelJob(config, jobId, endpointId) {
  return request(config, "cancel", { method: "POST", jobId, endpointId });
}

export function isTerminalStatus(status) {
  return terminalStates.has(status);
}
