let workflow = null;
let models = [];
let activeJobId = null;

const $ = (selector) => document.querySelector(selector);
const message = (text, error = false) => {
  $("#message").textContent = text;
  $("#message").style.color = error ? "#ff9db5" : "#a8c0ff";
};
const api = async (url, options = {}) => {
  const response = await fetch(url, {
    ...options,
    headers: { "content-type": "application/json", ...(options.headers || {}) }
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed: ${response.status}`);
  return body;
};

async function refreshModels() {
  ({ models } = await api("/api/models"));
  $("#models").innerHTML = models.length ? models.map((model) => `
    <div class="model-row">
      <input type="checkbox" value="${model.id}" checked>
      <span><strong>${model.filename}</strong><br><small>${model.modelType} · ${(model.sizeBytes / 1024 ** 3).toFixed(2)} GB</small></span>
    </div>`).join("") : '<div class="muted">No local models imported.</div>';
}

async function refreshSession() {
  const session = await api("/api/cloud-session");
  $("#session").textContent = session.active
    ? `Active ${session.mock ? "mock " : ""}volume ${session.volumeId} · ${(session.provisionedBytes / 1024 ** 3).toFixed(2)} GB`
    : "No chargeable cloud session.";
  $("#generate").disabled = !session.active || !workflow;
  $("#start-session").disabled = session.active;
  $("#end-session").disabled = !session.active;
}

$("#model-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const result = await api("/api/models/import-local", {
      method: "POST",
      body: JSON.stringify({ sourcePath: $("#model-path").value, modelType: $("#model-type").value })
    });
    message(result.duplicate ? "Model already exists in the library." : "Model imported.");
    await refreshModels();
  } catch (error) { message(error.message, true); }
});

$("#workflow-file").addEventListener("change", async (event) => {
  try {
    workflow = JSON.parse(await event.target.files[0].text());
    const result = await api("/api/workflows/inspect", {
      method: "POST", body: JSON.stringify({ workflow })
    });
    $("#workflow-summary").textContent = `${result.nodeCount} nodes · ${result.fields.length} editable fields · ${result.models.length} model references`;
    $("#fields").innerHTML = result.fields.slice(0, 20).map((field) =>
      `<div><strong>${field.title}</strong><br><small>${field.key}: ${String(field.value)}</small></div>`
    ).join("");
    await refreshSession();
  } catch (error) { workflow = null; message(error.message, true); }
});

$("#start-session").addEventListener("click", async () => {
  try {
    const modelIds = [...document.querySelectorAll(".model-row input:checked")].map((input) => input.value);
    await api("/api/cloud-session", { method: "POST", body: JSON.stringify({ modelIds }) });
    message("Temporary cloud session started.");
    await refreshSession();
  } catch (error) { message(error.message, true); }
});

$("#generate").addEventListener("click", async () => {
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ workflow, images: [] }) });
    activeJobId = job.id;
    $("#generate").disabled = true;
    message(`Job ${job.id}: ${job.status || "IN_QUEUE"}`);
    await pollJob(job.id);
  } catch (error) { message(error.message, true); }
});

async function pollJob(jobId) {
  const terminal = new Set(["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]);
  while (activeJobId === jobId) {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    message(`Job ${job.id}: ${job.status}${job.saved?.length ? ` · saved ${job.saved.length} image(s)` : ""}`);
    if (terminal.has(job.status)) {
      activeJobId = null;
      await refreshSession();
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

$("#end-session").addEventListener("click", async () => {
  try {
    await api("/api/cloud-session", { method: "DELETE" });
    message("Session ended and application-owned temporary volume deleted.");
    await refreshSession();
  } catch (error) { message(error.message, true); }
});

const config = await api("/api/config");
$("#mode").textContent = config.mockRunpod ? "MOCK · NO PAID CALLS" : "RUNPOD LIVE";
await Promise.all([refreshModels(), refreshSession()]);
