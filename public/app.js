let workflow = null;
let activeJobId = null;
let config = null;
let timerInterval = null;
let startTime = null;
let loraUrls = [];
let activePreset = "txt2img";

const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d");
let isDrawing = false;
let lastX = 0;
let lastY = 0;

const $ = (selector) => document.querySelector(selector);

const message = (text, error = false) => {
  const $msg = $("#message");
  $msg.textContent = text;
  $msg.style.color = error ? "#ff7b7b" : "#c5a880";
  $msg.classList.add("active");
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

// Start and Stop execution timer
function startTimer() {
  stopTimer();
  startTime = Date.now();
  $("#progress-timer").textContent = "0.0s";
  timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    $("#progress-timer").textContent = `${elapsed}s`;
  }, 100);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

// Accordion support helper
function updateAccordionHeight() {
  const content = $("#accordion-content");
  const chevron = $("#accordion-toggle .chevron");
  if (!chevron || !content) return;
  if (chevron.classList.contains("open")) {
    content.style.maxHeight = `${content.scrollHeight}px`;
  } else {
    content.style.maxHeight = "0px";
  }
}

function renderFieldHtml(field) {
  const isPrompt = field.key === "text";
  const inputId = `field-${field.nodeId}-${field.key}`;
  const inputHtml = typeof field.value === "number"
    ? `<input type="number" id="${inputId}" class="field-input" data-node-id="${field.nodeId}" data-key="${field.key}" value="${field.value}">`
    : `<textarea id="${inputId}" class="field-input ${isPrompt ? 'prominent-prompt' : ''}" data-node-id="${field.nodeId}" data-key="${field.key}">${field.value}</textarea>`;
  
  return `<div class="${isPrompt ? 'prominent-label' : ''}">
    <label>
      <span><strong>${isPrompt ? '✨ ' : ''}${field.title}</strong> (${field.key})</span>
      ${inputHtml}
    </label>
  </div>`;
}

// Render dynamic fields (split into prompt and advanced)
function renderWorkflowFields(fields) {
  const promptFields = fields.filter((f) => f.key === "text");
  const advancedFields = fields.filter((f) => f.key !== "text");

  $("#fields").innerHTML = promptFields.map(renderFieldHtml).join("");
  $("#advanced-fields").innerHTML = advancedFields.map(renderFieldHtml).join("");
  
  updateAccordionHeight();
}

// Load a preset workflow
async function loadPreset() {
  try {
    message("Loading preset...");
    const model = $("#base-model-select").value;
    let url = "";
    if (model === "flux2") {
      if (activePreset === "txt2img") {
        url = "/samples/flux2-klein-9b-text-to-image-api.json";
      } else if (activePreset === "img2img") {
        url = "/samples/flux2-klein-9b-image-to-image-api.json";
      } else if (activePreset === "inpaint") {
        url = "/samples/flux2-klein-9b-inpaint-api.json";
      }
    } else {
      if (activePreset === "txt2img") {
        url = "/samples/flux1-dev-text-to-image-api.json";
      } else if (activePreset === "img2img") {
        url = "/samples/flux1-dev-image-to-image-api.json";
      } else if (activePreset === "inpaint") {
        url = "/samples/flux1-dev-image-to-image-api.json";
      }
    }
      
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch preset: ${response.statusText}`);
    
    workflow = await response.json();
    
    // Inspect workflow
    const result = await api("/api/workflows/inspect", {
      method: "POST",
      body: JSON.stringify({ workflow })
    });
    
    $("#workflow-summary").textContent = `${result.nodeCount} nodes · ${result.fields.length} editable fields · ${result.models.length} model references`;
    
    // Show/hide image and mask upload groups
    const hasLoadImage = workflow && Object.values(workflow).some((node) => node.class_type === "LoadImage");
    const hasLoadImageMask = workflow && Object.values(workflow).some((node) => node.class_type === "LoadImageMask");
    $("#img2img-upload-group").style.display = (hasLoadImage || hasLoadImageMask) ? "block" : "none";
    $("#mask-upload-subgroup").style.display = hasLoadImageMask ? "block" : "none";
    
    // Render editable fields
    renderWorkflowFields(result.fields);
    
    $("#generate").disabled = false;
    message("Preset loaded successfully.");
  } catch (error) {
    workflow = null;
    $("#generate").disabled = true;
    message(error.message, true);
  }
}

// Setup Presets Toggles
$("#preset-txt2img").addEventListener("click", () => {
  $("#preset-txt2img").classList.add("active");
  $("#preset-img2img").classList.remove("active");
  $("#preset-inpaint").classList.remove("active");
  activePreset = "txt2img";
  loadPreset();
});

$("#preset-img2img").addEventListener("click", () => {
  $("#preset-img2img").classList.add("active");
  $("#preset-txt2img").classList.remove("active");
  $("#preset-inpaint").classList.remove("active");
  activePreset = "img2img";
  loadPreset();
});

$("#preset-inpaint").addEventListener("click", () => {
  $("#preset-inpaint").classList.add("active");
  $("#preset-txt2img").classList.remove("active");
  $("#preset-img2img").classList.remove("active");
  activePreset = "inpaint";
  loadPreset();
});

// Setup base model change handler
$("#base-model-select").addEventListener("change", () => {
  loadPreset();
});

// Custom JSON file upload
$("#workflow-file").addEventListener("change", async (event) => {
  try {
    message("Loading custom workflow...");
    workflow = JSON.parse(await event.target.files[0].text());
    
    // Clear active preset classes
    $("#preset-txt2img").classList.remove("active");
    $("#preset-img2img").classList.remove("active");
    $("#preset-inpaint").classList.remove("active");
    
    const result = await api("/api/workflows/inspect", {
      method: "POST", body: JSON.stringify({ workflow })
    });
    
    $("#workflow-summary").textContent = `${result.nodeCount} nodes · ${result.fields.length} editable fields · ${result.models.length} model references`;
    
    // Show/hide image and mask upload groups
    const hasLoadImage = workflow && Object.values(workflow).some((node) => node.class_type === "LoadImage");
    const hasLoadImageMask = workflow && Object.values(workflow).some((node) => node.class_type === "LoadImageMask");
    $("#img2img-upload-group").style.display = (hasLoadImage || hasLoadImageMask) ? "block" : "none";
    $("#mask-upload-subgroup").style.display = hasLoadImageMask ? "block" : "none";
    
    // Render editable fields
    renderWorkflowFields(result.fields);
    
    $("#generate").disabled = false;
    message("Custom workflow loaded successfully.");
  } catch (error) {
    workflow = null;
    $("#generate").disabled = true;
    message(error.message, true);
  }
});

function readImageAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
}

// Render dynamic LoRA list
function renderLoras() {
  $("#lora-list").innerHTML = loraUrls.map((url, index) => `
    <div style="display: flex; justify-content: space-between; align-items: center; background: var(--bg-input); border: 1px solid var(--border-color); padding: 8px 12px; border-radius: 6px;">
      <span style="font-size: 0.8rem; word-break: break-all; color: var(--color-text-bright);">${url}</span>
      <button type="button" class="btn-remove-lora" data-index="${index}" style="width: auto; padding: 6px 10px; font-size: 0.75rem; background: #3a1d28; color: #ffb6c8; border-radius: 4px; border: 0; cursor: pointer;">Remove</button>
    </div>
  `).join("");
  
  document.querySelectorAll(".btn-remove-lora").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const index = parseInt(e.target.getAttribute("data-index"));
      loraUrls.splice(index, 1);
      renderLoras();
    });
  });
  
  updateAccordionHeight();
}

// Handle Add LoRA Click
$("#btn-add-lora").addEventListener("click", () => {
  const url = $("#lora-url-input").value.trim();
  if (!url) return;
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    message("Please enter a valid HTTP/HTTPS URL", true);
    return;
  }
  loraUrls.push(url);
  $("#lora-url-input").value = "";
  renderLoras();
});

// Dynamic LoRA Node Injection Compiler
function injectLorasIntoWorkflow(workflow, urls) {
  if (!urls || urls.length === 0) return workflow;
  
  const copy = JSON.parse(JSON.stringify(workflow));
  
  // Find base model and clip loader node IDs
  let modelNodeId = null;
  let clipNodeId = null;
  for (const id in copy) {
    if (copy[id].class_type === "UNETLoader" || copy[id].class_type === "CheckpointLoaderSimple") {
      modelNodeId = id;
    }
    if (copy[id].class_type === "CLIPLoader" || copy[id].class_type === "CheckpointLoaderSimple") {
      clipNodeId = id;
    }
  }
  
  if (!modelNodeId || !clipNodeId) {
    console.warn("Could not find base model/clip loader nodes in workflow for LoRA injection.");
    return copy;
  }
  
  let lastModelOutput = [modelNodeId, 0];
  let lastClipOutput = [clipNodeId, 0];
  
  if (copy[modelNodeId].class_type === "CheckpointLoaderSimple") {
    lastClipOutput = [modelNodeId, 1];
  }
  
  // Insert LoRA nodes sequentially
  urls.forEach((url, index) => {
    const loraNodeId = `lora_download_${index + 1}`;
    copy[loraNodeId] = {
      class_type: "LoadLoraFromURL",
      inputs: {
        model: [lastModelOutput[0], lastModelOutput[1]],
        clip: [lastClipOutput[0], lastClipOutput[1]],
        url: url,
        strength_model: 1.0,
        strength_clip: 1.0
      },
      "_meta": {
        "title": `Dynamic LoRA ${index + 1}`
      }
    };
    lastModelOutput = [loraNodeId, 0];
    lastClipOutput = [loraNodeId, 1];
  });
  
  // Update downstream references
  for (const id in copy) {
    if (id.startsWith("lora_download_")) continue;
    
    const inputs = copy[id].inputs;
    if (!inputs) continue;
    
    for (const key in inputs) {
      const val = inputs[key];
      if (Array.isArray(val) && val.length === 2) {
        const sourceNodeId = val[0];
        const outputIndex = val[1];
        
        // Handle Model output mapping
        if (sourceNodeId === modelNodeId) {
          const classType = copy[modelNodeId].class_type;
          if (classType === "CheckpointLoaderSimple" && outputIndex === 0) {
            inputs[key] = [lastModelOutput[0], lastModelOutput[1]];
          } else if (classType === "UNETLoader" && outputIndex === 0) {
            inputs[key] = [lastModelOutput[0], lastModelOutput[1]];
          }
        }
        
        // Handle CLIP output mapping (disjoint from Model mapping)
        if (sourceNodeId === clipNodeId) {
          const classType = copy[clipNodeId].class_type;
          if (classType === "CheckpointLoaderSimple" && outputIndex === 1) {
            inputs[key] = [lastClipOutput[0], lastClipOutput[1]];
          } else if (classType === "CLIPLoader" && outputIndex === 0) {
            inputs[key] = [lastClipOutput[0], lastClipOutput[1]];
          }
        }
      }
    }
  }
  
  return copy;
}

// Generate execution handler
$("#generate").addEventListener("click", async () => {
  try {
    let images = [];
    const imageFile = $("#input-image-file").files[0];
    if (imageFile) {
      const base64 = await readImageAsBase64(imageFile);
      const filename = imageFile.name || "input_image.png";
      images.push({
        name: filename,
        image: base64
      });
      
      // Update any LoadImage nodes in the workflow to match this filename
      if (workflow) {
        for (const nodeId in workflow) {
          if (workflow[nodeId].class_type === "LoadImage") {
            workflow[nodeId].inputs = workflow[nodeId].inputs || {};
            workflow[nodeId].inputs.image = filename;
          }
        }
      }
    }
    
    const maskFile = $("#input-mask-file") ? $("#input-mask-file").files[0] : null;
    const hasLoadImageMask = workflow && Object.values(workflow).some((node) => node.class_type === "LoadImageMask");
    if (maskFile) {
      const base64 = await readImageAsBase64(maskFile);
      const filename = maskFile.name || "mask_image.png";
      images.push({
        name: filename,
        image: base64
      });
      
      // Update any LoadImageMask nodes in the workflow to match this filename
      if (workflow) {
        for (const nodeId in workflow) {
          if (workflow[nodeId].class_type === "LoadImageMask") {
            workflow[nodeId].inputs = workflow[nodeId].inputs || {};
            workflow[nodeId].inputs.image = filename;
          }
        }
      }
    } else if (hasLoadImageMask) {
      // Extract the drawn mask from the canvas
      const maskBase64 = maskCanvas.toDataURL("image/png");
      images.push({
        name: "mask_image.png",
        image: maskBase64
      });
      
      // Update any LoadImageMask nodes in the workflow to match this filename
      if (workflow) {
        for (const nodeId in workflow) {
          if (workflow[nodeId].class_type === "LoadImageMask") {
            workflow[nodeId].inputs = workflow[nodeId].inputs || {};
            workflow[nodeId].inputs.image = "mask_image.png";
          }
        }
      }
    }
    
    // Read and update editable text/number inputs
    if (workflow) {
      document.querySelectorAll(".field-input").forEach((input) => {
        const nodeId = input.getAttribute("data-node-id");
        const key = input.getAttribute("data-key");
        let value = input.value;
        if (input.type === "number") {
          value = Number(value);
        }
        if (workflow[nodeId] && workflow[nodeId].inputs) {
          workflow[nodeId].inputs[key] = value;
        }
      });
    }
    
    // Inject LoRAs if specified
    let compiledWorkflow = workflow;
    if (loraUrls.length > 0) {
      compiledWorkflow = injectLorasIntoWorkflow(workflow, loraUrls);
    }
    
    const baseModel = $("#base-model-select").value;
    
    message("Submitting job to RunPod...");
    const job = await api("/api/jobs", { 
      method: "POST", 
      body: JSON.stringify({ 
        workflow: compiledWorkflow, 
        images,
        baseModel
      }) 
    });
    
    activeJobId = job.id;
    $("#generate").disabled = true;
    
    // Show and reset status card
    $("#status-card").style.display = "block";
    $("#status-badge").className = "status-badge running";
    $("#status-badge").textContent = job.status || "IN_QUEUE";
    $("#progress-fill").style.width = "10%";
    $("#progress-phase").textContent = "Job submitted...";
    
    // Clear previous gallery
    $("#gallery-card").style.display = "none";
    $("#gallery-container").innerHTML = "";
    
    startTimer();
    message(`Job ${job.id} started.`);
    await pollJob(job.id);
  } catch (error) {
    message(error.message, true);
    $("#generate").disabled = false;
  }
});

// Polling status and progress updates
async function pollJob(jobId) {
  const terminal = new Set(["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]);
  while (activeJobId === jobId) {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      
      const status = job.status || "IN_QUEUE";
      $("#status-badge").textContent = status;
      
      if (status === "COMPLETED") {
        $("#status-badge").className = "status-badge completed";
      } else if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(status)) {
        $("#status-badge").className = "status-badge failed";
      } else {
        $("#status-badge").className = "status-badge running";
      }
      
      let progress = 10;
      let phase = "Queueing...";
      
      if (status === "IN_QUEUE") {
        progress = 15;
        phase = "Waiting in Queue...";
      } else if (status === "IN_PROGRESS") {
        progress = 40;
        phase = "Executing on RunPod GPU...";
        
        if (Array.isArray(job.stream) && job.stream.length > 0) {
          const latest = job.stream[job.stream.length - 1];
          if (latest && typeof latest === "object") {
            if (latest.step !== undefined && latest.max_steps !== undefined) {
              progress = Math.round((latest.step / latest.max_steps) * 100);
              phase = `Generating (Step ${latest.step}/${latest.max_steps})`;
            } else if (latest.message) {
              phase = latest.message;
            }
          }
        }
      } else if (status === "COMPLETED") {
        progress = 100;
        phase = "Finished successfully!";
      } else {
        progress = 100;
        phase = `Execution ${status.toLowerCase()}`;
      }
      
      $("#progress-fill").style.width = `${progress}%`;
      $("#progress-phase").textContent = phase;
      
      message(`Job ${job.id}: ${status}${job.saved?.length ? ` · saved ${job.saved.length} image(s)` : ""}`);
      
      if (terminal.has(status)) {
        activeJobId = null;
        stopTimer();
        $("#generate").disabled = false;
        
        if (status === "COMPLETED" && Array.isArray(job.saved) && job.saved.length > 0) {
          $("#gallery-card").style.display = "block";
          $("#gallery-container").innerHTML = job.saved.map((item) => `
            <div class="gallery-card">
              <img src="${item.url}" alt="${item.filename}">
              <div class="gallery-info">
                <div class="gallery-filename">${item.filename}</div>
                <div class="gallery-meta">${(item.sizeBytes / 1024 / 1024).toFixed(2)} MB</div>
              </div>
            </div>
          `).join("");
        } else if (status !== "COMPLETED") {
          message(`Job failed with status: ${status}. Error details: ${job.error || job.message || "Unknown error"}`, true);
        }
        return;
      }
    } catch (err) {
      console.error("Error polling job:", err);
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

// Accordion Toggle Event Listener
$("#accordion-toggle").addEventListener("click", () => {
  const content = $("#accordion-content");
  const chevron = $("#accordion-toggle .chevron");
  const isOpen = chevron.classList.toggle("open");
  if (isOpen) {
    content.style.maxHeight = `${content.scrollHeight}px`;
  } else {
    content.style.maxHeight = "0px";
  }
});

// Painter Brush and Canvas logic
function clearMask() {
  const canvas = $("#painter-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  maskCtx.fillStyle = "#000000";
  maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
}

function getCoords(event) {
  const canvas = $("#painter-canvas");
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height
  };
}

function startDrawing(e) {
  isDrawing = true;
  const coords = getCoords(e);
  lastX = coords.x;
  lastY = coords.y;
  draw(e);
}

function draw(e) {
  if (!isDrawing) return;
  const canvas = $("#painter-canvas");
  const coords = getCoords(e);
  const x = coords.x;
  const y = coords.y;
  
  const brushSize = parseInt($("#brush-size-slider").value);
  
  // Draw on visible canvas
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = "rgba(255, 0, 0, 0.4)";
  ctx.lineWidth = brushSize;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  
  // Draw on hidden mask canvas
  maskCtx.strokeStyle = "#ffffff";
  maskCtx.lineWidth = brushSize;
  maskCtx.lineCap = "round";
  maskCtx.lineJoin = "round";
  maskCtx.beginPath();
  maskCtx.moveTo(lastX, lastY);
  maskCtx.lineTo(x, y);
  maskCtx.stroke();
  
  lastX = x;
  lastY = y;
}

function stopDrawing() {
  isDrawing = false;
}

// Add Drawing Event Listeners
const painterCanvas = $("#painter-canvas");
painterCanvas.addEventListener("pointerdown", startDrawing);
painterCanvas.addEventListener("pointermove", draw);
window.addEventListener("pointerup", stopDrawing);

$("#btn-clear-mask").addEventListener("click", clearMask);

$("#brush-size-slider").addEventListener("input", (e) => {
  $("#brush-size-val").textContent = `${e.target.value}px`;
});

// Image loading and canvas mapping
$("#input-image-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) {
    $("#mask-paint-container").style.display = "none";
    return;
  }
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = $("#painter-img");
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
});

$("#painter-img").addEventListener("load", () => {
  const img = $("#painter-img");
  const canvas = $("#painter-canvas");
  
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  
  maskCanvas.width = img.naturalWidth;
  maskCanvas.height = img.naturalHeight;
  
  clearMask();
  
  const hasLoadImageMask = workflow && Object.values(workflow).some((node) => node.class_type === "LoadImageMask");
  if (hasLoadImageMask) {
    $("#mask-paint-container").style.display = "block";
    updateAccordionHeight();
  }
});

// Initial boot logic
config = await api("/api/config");
$("#mode").textContent = config.mockRunpod ? "MOCK MODE" : "RUNPOD LIVE";

// Default load Text to Image preset
activePreset = "txt2img";
loadPreset();
