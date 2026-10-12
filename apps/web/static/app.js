const fileInput = document.getElementById("file");
const sourcePreview = document.getElementById("source-preview");
const preEditor = document.getElementById("pre-editor");
const timelineViewport = document.getElementById("timeline-viewport");
const timelineContent = document.getElementById("timeline-content");
const timelineRuler = document.getElementById("timeline-ruler");
const timelineSections = document.getElementById("timeline-sections");
const timelinePlayhead = document.getElementById("timeline-playhead");
const sectionList = document.getElementById("section-list");
const zoomSlider = document.getElementById("timeline-zoom");
const startBtn = document.getElementById("start-btn");
const workflow = document.getElementById("workflow");
// Keep DL available in the backend while its interface is temporarily hidden.
const DL_ALGO_VISIBLE = false;
if (DL_ALGO_VISIBLE) {
  workflow.appendChild(document.getElementById("dl-workflow-option").content.cloneNode(true));
}
let dlCapability = null;
let potsAvailable = false;
let potsStatus = "Checking potted-shots mode…";

async function loadShotSelections() {
  try {
    const response = await fetch("/api/shot-selections");
    if (!response.ok) throw new Error("Restart the app server to enable Potted shots only.");
    const data = await response.json();
    const capability = data.selections.find((item) => item.id === "pots_only");
    potsAvailable = Boolean(capability?.available);
    potsStatus = capability?.reason || "Potted shots only is unavailable.";
  } catch (error) {
    potsAvailable = false;
    potsStatus = error.message || String(error);
  }
  const option = document.getElementById("pots-workflow-option");
  option.disabled = !potsAvailable;
  option.title = potsStatus;
  updateWorkflow();
}

function updateStartAvailability() {
  startBtn.disabled = !sourceDuration || workflow.disabled
    || (workflow.value === "pots_only" && !potsAvailable)
    || (workflow.value === "dl_algo" && !dlCapability?.available);
}

async function loadAlgorithms() {
  if (!DL_ALGO_VISIBLE) return;
  try {
    const response = await fetch("/api/algorithms");
    if (!response.ok) throw new Error("Could not check DL Algo readiness.");
    const metadata = await response.json();
    dlCapability = metadata.algorithms.find((item) => item.id === "dl_algo")
      || { available: false, reason: "DL Algo is not available yet." };
  } catch (error) {
    dlCapability = { available: false, reason: error.message || String(error) };
  }
  updateWorkflow();
}

function updateWorkflow() {
  const classic = workflow.value === "classic";
  const dl = DL_ALGO_VISIBLE && workflow.value === "dl_algo";
  const pots = workflow.value === "pots_only";
  document.getElementById("workflow-heading").textContent = classic
    ? "Upload & analyze" : dl ? "DL Algo" : pots ? "Potted shots only" : "Create snooker highlights";
  document.getElementById("workflow-description").textContent = classic
    ? "Detect shots, then open the previous editor with preview, timeline, shot controls, and export buttons."
    : pots ? "Keep confirmed scoring pots using stable scores and the player's turn. Misses and fouls are excluded; unreadable scores are flagged for review."
    : dl ? "Create highlights with DL Algo. Results are saved as a separate job."
      : "Upload the entire match. Remove waiting and setup footage, then export with mix transitions.";
  const status = document.getElementById("algorithm-status");
  status.classList.toggle("hidden", !dl && !pots);
  status.textContent = pots ? potsStatus : dlCapability?.available
    ? "Experimental DL Algo is ready for comparison. Accuracy on new matches has not been verified."
    : dlCapability?.reason || "Checking DL Algo readiness…";
  document.getElementById("pre-editor-heading").textContent = classic
    ? "Prepare match before analysis" : "Optional source trimming";
  document.getElementById("pre-editor-description").textContent = classic
    ? "Split around frame breaks and delete those sections, or keep the whole video. Your original video is never changed."
    : "Keep the whole video and start. Shot detection, cutting, and export run automatically.";
  startBtn.textContent = classic ? "Prepare & start analysis"
    : dl ? "Create with DL Algo" : pots ? "Create pots-only video" : "Create video automatically";
  updateStartAvailability();
}

workflow.addEventListener("change", () => {
  updateWorkflow();
  if (workflow.value === "pots_only") loadShotSelections();
  if (workflow.value === "dl_algo") loadAlgorithms();
});

let sourceDuration = 0;
let sections = [];
let selectedSectionId = null;
let nextSectionId = 1;
let timelineZoom = 1;
let editHistory = [];
let redoHistory = [];
let localVideoUrl = null;
let draggingPlayhead = false;
let dragPointerClientX = 0;
let dragScrollFrame = 0;
updateWorkflow();
loadShotSelections();
loadAlgorithms();
setInterval(() => {
  if (workflow.value === "dl_algo" && !dlCapability?.available && !document.hidden) {
    loadAlgorithms();
  }
}, 10000);

function fmt(t) {
  if (!Number.isFinite(t)) return "00:00.000";
  const hours = Math.floor(t / 3600);
  const minutes = Math.floor((t % 3600) / 60);
  const seconds = t % 60;
  const base = `${String(minutes).padStart(2, "0")}:${seconds.toFixed(3).padStart(6, "0")}`;
  return hours ? `${String(hours).padStart(2, "0")}:${base}` : base;
}

function fmtRuler(t) {
  if (!Number.isFinite(t)) return "00:00";
  const rounded = Math.max(0, Math.round(t));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const seconds = rounded % 60;
  if (hours) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function cloneSections(value = sections) {
  return value.map((section) => ({ ...section }));
}

function saveHistory() {
  editHistory.push({
    sections: cloneSections(),
    selectedSectionId,
  });
  if (editHistory.length > 100) editHistory.shift();
  redoHistory = [];
}

function niceTickInterval(raw) {
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / power;
  const step = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return step * power;
}

function playheadPixel() {
  const time = Math.max(0, Math.min(sourceDuration, sourcePreview.currentTime || 0));
  return sourceDuration ? (time / sourceDuration) * timelineContent.clientWidth : 0;
}

function keepPlayheadVisible(center = false) {
  const pixel = playheadPixel();
  const padding = 36;
  const visibleStart = timelineViewport.scrollLeft;
  const visibleEnd = visibleStart + timelineViewport.clientWidth;
  if (center) {
    timelineViewport.scrollLeft = Math.max(0, pixel - timelineViewport.clientWidth / 2);
  } else if (pixel < visibleStart + padding) {
    timelineViewport.scrollLeft = Math.max(0, pixel - padding);
  } else if (pixel > visibleEnd - padding) {
    timelineViewport.scrollLeft = pixel - timelineViewport.clientWidth + padding;
  }
}

function updatePlayhead({ ensureVisible = false } = {}) {
  const time = Math.max(0, Math.min(sourceDuration, sourcePreview.currentTime || 0));
  timelinePlayhead.style.left = `${sourceDuration ? (time / sourceDuration) * 100 : 0}%`;
  document.getElementById("source-clock").textContent = `${fmt(time)} / ${fmt(sourceDuration)}`;
  if (ensureVisible && !draggingPlayhead) keepPlayheadVisible();
}

function renderRuler() {
  const targetTicks = Math.max(2, Math.floor(timelineContent.clientWidth / 110));
  const interval = niceTickInterval(sourceDuration / targetTicks);
  const ticks = [];
  for (let time = 0; time <= sourceDuration + interval * 0.25; time += interval) {
    const bounded = Math.min(time, sourceDuration);
    const left = sourceDuration ? (bounded / sourceDuration) * 100 : 0;
    const endpointClass = bounded >= sourceDuration ? " last" : "";
    ticks.push(`<span class="pre-tick${endpointClass}" style="left:${left}%"><i></i><b>${fmtRuler(bounded)}</b></span>`);
    if (bounded >= sourceDuration) break;
  }
  timelineRuler.innerHTML = ticks.join("");
}

function renderEditor() {
  if (!sourceDuration) return;
  const viewportWidth = Math.max(1, timelineViewport.clientWidth || 800);
  timelineContent.style.width = `${Math.round(viewportWidth * timelineZoom)}px`;
  renderRuler();

  timelineSections.innerHTML = sections.map((section, index) => {
    const left = (section.start / sourceDuration) * 100;
    const width = ((section.end - section.start) / sourceDuration) * 100;
    const classes = [
      "pre-section",
      section.deleted ? "deleted" : "kept",
      section.id === selectedSectionId ? "selected" : "",
    ].filter(Boolean).join(" ");
    return `<button type="button" class="${classes}" data-section-id="${section.id}"
      style="left:${left}%;width:${width}%" title="Section ${index + 1}: ${fmt(section.start)} → ${fmt(section.end)}">
      <span>${index + 1}</span>
    </button>`;
  }).join("");

  sectionList.innerHTML = sections.map((section, index) => {
    const state = section.deleted ? "DELETE" : "KEEP";
    const classes = [
      "section-chip",
      section.deleted ? "deleted" : "",
      section.id === selectedSectionId ? "selected" : "",
    ].filter(Boolean).join(" ");
    return `<button type="button" class="${classes}" data-section-id="${section.id}">
      ${index + 1}. ${fmt(section.start)}–${fmt(section.end)} · ${state}
    </button>`;
  }).join("");

  const kept = sections.filter((section) => !section.deleted);
  const keptDuration = kept.reduce((sum, section) => sum + section.end - section.start, 0);
  document.getElementById("edit-summary").textContent = (
    `${kept.length} kept · ${fmt(keptDuration)} · removed ${fmt(sourceDuration - keptDuration)}`
  );
  document.getElementById("zoom-label").textContent = `${timelineZoom}×`;
  zoomSlider.value = String(timelineZoom);
  document.getElementById("undo-edit").disabled = editHistory.length === 0;
  document.getElementById("redo-edit").disabled = redoHistory.length === 0;

  const selected = sections.find((section) => section.id === selectedSectionId);
  document.getElementById("delete-section").disabled = !selected || selected.deleted;
  document.getElementById("restore-section").disabled = !selected || !selected.deleted;
  updatePlayhead();
}

function initializeEditor(duration) {
  sourceDuration = duration;
  sections = [{ id: nextSectionId++, start: 0, end: duration, deleted: false }];
  selectedSectionId = sections[0].id;
  timelineZoom = 1;
  editHistory = [];
  redoHistory = [];
  preEditor.classList.remove("hidden");
  updateStartAvailability();
  window.requestAnimationFrame(renderEditor);
}

function sectionAt(time) {
  return sections.find((section) => time >= section.start && time <= section.end + 1e-6);
}

function splitAtPlayhead() {
  const time = Math.max(0, Math.min(sourceDuration, sourcePreview.currentTime));
  const index = sections.findIndex(
    (section) => time > section.start + 0.05 && time < section.end - 0.05,
  );
  if (index < 0) return;
  saveHistory();
  const original = sections[index];
  const left = { ...original, end: time };
  const right = { ...original, id: nextSectionId++, start: time };
  sections.splice(index, 1, left, right);
  selectedSectionId = right.id;
  renderEditor();
}

function setSelectedDeleted(deleted) {
  const selected = sections.find((section) => section.id === selectedSectionId);
  if (!selected || selected.deleted === deleted) return;
  saveHistory();
  selected.deleted = deleted;
  renderEditor();
}

function currentEditState() {
  return {
    sections: cloneSections(),
    selectedSectionId,
  };
}

function restoreEditState(state) {
  sections = cloneSections(state.sections);
  selectedSectionId = state.selectedSectionId;
  renderEditor();
}

function undoEdit() {
  const previous = editHistory.pop();
  if (!previous) return;
  redoHistory.push(currentEditState());
  restoreEditState(previous);
}

function redoEdit() {
  const next = redoHistory.pop();
  if (!next) return;
  editHistory.push(currentEditState());
  restoreEditState(next);
}

function setZoom(requestedZoom) {
  const nextZoom = Math.max(1, Math.min(64, Math.round(requestedZoom)));
  if (nextZoom === timelineZoom) return;
  const anchorTime = Math.max(0, Math.min(sourceDuration, sourcePreview.currentTime || 0));
  timelineZoom = nextZoom;
  renderEditor();
  const anchorPixel = (anchorTime / sourceDuration) * timelineContent.clientWidth;
  timelineViewport.scrollLeft = Math.max(0, anchorPixel - timelineViewport.clientWidth / 2);
}

function selectSectionAndSeek(sectionId, seekTime = null) {
  const section = sections.find((item) => item.id === sectionId);
  if (!section) return;
  selectedSectionId = section.id;
  sourcePreview.currentTime = Math.max(
    section.start,
    Math.min(section.end, seekTime === null ? section.start : seekTime),
  );
  renderEditor();
}

function handleTimelineClick(event) {
  if (event.target.closest("#timeline-playhead")) return;
  const segmentButton = event.target.closest("[data-section-id]");
  const rect = timelineContent.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  const time = ratio * sourceDuration;
  const id = segmentButton
    ? Number(segmentButton.dataset.sectionId)
    : sectionAt(time)?.id;
  if (id) selectSectionAndSeek(id, time);
}

function seekFromPlayheadClientX(clientX) {
  const viewportRect = timelineViewport.getBoundingClientRect();
  const boundedClientX = Math.max(viewportRect.left, Math.min(viewportRect.right, clientX));
  const contentRect = timelineContent.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (boundedClientX - contentRect.left) / contentRect.width));
  const time = ratio * sourceDuration;
  sourcePreview.currentTime = time;
  const selected = sectionAt(time);
  if (selected) selectedSectionId = selected.id;
  updatePlayhead();
}

function runDragAutoScroll() {
  if (!draggingPlayhead) return;
  const viewportRect = timelineViewport.getBoundingClientRect();
  const edgeSize = Math.min(64, viewportRect.width * 0.12);
  let velocity = 0;
  if (dragPointerClientX < viewportRect.left + edgeSize) {
    const depth = (viewportRect.left + edgeSize - dragPointerClientX) / edgeSize;
    velocity = -Math.max(2, 30 * Math.min(1, depth));
  } else if (dragPointerClientX > viewportRect.right - edgeSize) {
    const depth = (dragPointerClientX - (viewportRect.right - edgeSize)) / edgeSize;
    velocity = Math.max(2, 30 * Math.min(1, depth));
  }

  if (velocity) {
    const before = timelineViewport.scrollLeft;
    timelineViewport.scrollLeft += velocity;
    if (timelineViewport.scrollLeft !== before) seekFromPlayheadClientX(dragPointerClientX);
  }
  dragScrollFrame = window.requestAnimationFrame(runDragAutoScroll);
}

function finishPlayheadDrag(event = null) {
  if (!draggingPlayhead) return;
  draggingPlayhead = false;
  timelinePlayhead.classList.remove("dragging");
  window.cancelAnimationFrame(dragScrollFrame);
  dragScrollFrame = 0;
  if (event && timelinePlayhead.hasPointerCapture(event.pointerId)) {
    timelinePlayhead.releasePointerCapture(event.pointerId);
  }
  renderEditor();
}

timelinePlayhead.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  event.stopPropagation();
  draggingPlayhead = true;
  dragPointerClientX = event.clientX;
  timelinePlayhead.classList.add("dragging");
  timelinePlayhead.setPointerCapture(event.pointerId);
  seekFromPlayheadClientX(event.clientX);
  window.cancelAnimationFrame(dragScrollFrame);
  dragScrollFrame = window.requestAnimationFrame(runDragAutoScroll);
});
timelinePlayhead.addEventListener("pointermove", (event) => {
  if (!draggingPlayhead) return;
  dragPointerClientX = event.clientX;
  seekFromPlayheadClientX(event.clientX);
});
timelinePlayhead.addEventListener("pointerup", finishPlayheadDrag);
timelinePlayhead.addEventListener("pointercancel", finishPlayheadDrag);
timelinePlayhead.addEventListener("click", (event) => event.stopPropagation());

function mergedKeepRanges() {
  const kept = sections.filter((section) => !section.deleted).sort((a, b) => a.start - b.start);
  const merged = [];
  for (const section of kept) {
    const previous = merged[merged.length - 1];
    if (previous && Math.abs(previous.end - section.start) <= 0.001) {
      previous.end = section.end;
    } else {
      merged.push({ start: section.start, end: section.end });
    }
  }
  return merged;
}

async function restartJob(jobId) {
  if (!confirm(`Restart analysis for job ${jobId}? Saved checkpoints and manual edits will be preserved.`)) return;
  try {
    const res = await fetch(`/api/jobs/${jobId}/restart`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || err.message || "Failed to restart job");
    }
    refreshJobs();
  } catch (error) {
    alert(`Could not restart job: ${error.message || error}`);
  }
}

const capcutJobMessages = new Map();

function escapeJobText(value) {
  const span = document.createElement("span");
  span.textContent = String(value || "");
  return span.innerHTML;
}

async function continueJob(jobId, button) {
  button.disabled = true;
  button.textContent = "Continuing…";
  try {
    const response = await fetch(`/api/jobs/${jobId}/continue`, { method: "POST" });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Could not continue processing");
    }
    await refreshJobs();
  } catch (error) {
    button.disabled = false;
    button.textContent = "Continue";
    alert(error.message || error);
  }
}

async function refreshJobs() {
  const el = document.getElementById("jobs");
  // Preserve a pending handoff and its result while the periodic job poll runs.
  if (el.querySelector("[data-capcut-active]")) return;
  try {
    const [res, capcut] = await Promise.all([fetch("/api/jobs"), window.snookerCapCut.load()]);
    const jobs = (await res.json()).filter(
      (job) => DL_ALGO_VISIBLE || job.algorithm !== "dl_algo",
    );
    if (!jobs.length) {
      el.innerHTML = "<p class='meta'>No jobs yet.</p>";
      return;
    }
    el.innerHTML = jobs
      .slice(0, 20)
      .map(
        (job) => `
      <div class="job-item">
        <div>
          <strong>${job.job_id}</strong>
          <div class="meta">${job.selection === "pots_only" ? "Potted shots only · " : ""}${job.algorithm === "dl_algo" ? "DL Algo · " : ""}${job.status || ""} · ${(job.progress * 100 || 0).toFixed(0)}% · ${escapeJobText(job.message)}</div>
          ${job.error ? `<div class="meta" role="alert">Error: ${escapeJobText(job.error)}</div>` : ""}
          ${Number.isFinite(job.checkpoint?.source_time) ? `<div class="meta">Saved through ${fmt(job.checkpoint.source_time)}${job.checkpoint.duration ? ` of ${fmt(job.checkpoint.duration)}` : ""}</div>` : ""}
        </div>
        <div class="job-actions">
          ${job.can_continue ? `<button type="button" class="btn-continue primary" data-job-id="${job.job_id}" title="Resume saved work without uploading again">Continue</button>` : ""}
          <button type="button" class="btn-restart" data-job-id="${job.job_id}" ${job.running ? "disabled" : ""}>Restart analysis</button>
          ${job.download_url ? `<a href="${job.download_url}">Download video</a>` : ""}
          ${["completed", "ready_for_review"].includes(job.status) ? `<button type="button" class="btn-capcut" data-job-id="${job.job_id}" ${capcut.available ? "" : "disabled"}>Open in CapCut</button>` : ""}
          <a href="/review/${job.job_id}?editor=classic">${job.algorithm === "dl_algo" ? "Open DL Algo editor" : "Open classic editor"}</a>
        </div>
      </div>`,
      )
      .join("");
    el.querySelectorAll(".btn-capcut").forEach((button) => {
      button.title = capcut.available
        ? "Create a new CapCut project with separate editable shots"
        : capcut.reason;
      const message = capcutJobMessages.get(button.dataset.jobId);
      if (message) {
        const status = document.createElement("p");
        status.className = "capcut-status muted";
        status.setAttribute("role", "status");
        status.textContent = message;
        button.closest(".job-item").querySelector("div").appendChild(status);
      }
    });
  } catch (error) {
    el.innerHTML = `<p class="meta">Could not load jobs: ${error}</p>`;
  }
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  document.getElementById("file-name").textContent = file ? file.name : "Choose a snooker video…";
  startBtn.disabled = true;
  preEditor.classList.add("hidden");
  sourceDuration = 0;
  sections = [];
  if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
  localVideoUrl = file ? URL.createObjectURL(file) : null;
  if (localVideoUrl) sourcePreview.src = localVideoUrl;
});

sourcePreview.addEventListener("loadedmetadata", () => {
  if (Number.isFinite(sourcePreview.duration) && sourcePreview.duration > 0) {
    initializeEditor(sourcePreview.duration);
  }
});
sourcePreview.addEventListener("timeupdate", () => {
  updatePlayhead({ ensureVisible: !sourcePreview.paused });
});
sourcePreview.addEventListener("seeked", () => updatePlayhead({ ensureVisible: true }));
document.getElementById("split-section").addEventListener("click", splitAtPlayhead);
document.getElementById("delete-section").addEventListener("click", () => setSelectedDeleted(true));
document.getElementById("restore-section").addEventListener("click", () => setSelectedDeleted(false));
document.getElementById("undo-edit").addEventListener("click", undoEdit);
document.getElementById("redo-edit").addEventListener("click", redoEdit);
timelineContent.addEventListener("click", handleTimelineClick);
sectionList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-section-id]");
  if (button) selectSectionAndSeek(Number(button.dataset.sectionId));
});
document.getElementById("zoom-out").addEventListener("click", () => setZoom(timelineZoom / 2));
document.getElementById("zoom-in").addEventListener("click", () => setZoom(timelineZoom * 2));
document.getElementById("zoom-fit").addEventListener("click", () => setZoom(1));
zoomSlider.addEventListener("input", () => setZoom(Number(zoomSlider.value)));
timelineViewport.addEventListener("wheel", (event) => {
  if (!event.ctrlKey) return;
  event.preventDefault();
  setZoom(event.deltaY < 0 ? timelineZoom * 2 : timelineZoom / 2);
}, { passive: false });
document.addEventListener("keydown", (event) => {
  if (!sourceDuration || preEditor.classList.contains("hidden")) return;
  if (
    event.target.matches('textarea, select, input:not([type="range"])')
    || event.target.isContentEditable
  ) return;
  if (event.key.toLowerCase() !== "z") return;

  const commandKey = event.ctrlKey || event.metaKey;
  event.preventDefault();
  if (commandKey && event.shiftKey) {
    redoEdit();
  } else if (commandKey) {
    undoEdit();
  } else if (!event.repeat) {
    splitAtPlayhead();
  }
});
window.addEventListener("resize", renderEditor);
window.addEventListener("beforeunload", () => {
  if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
});

document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file || !sourceDuration) return;
  const keepRanges = mergedKeepRanges();
  const wrap = document.getElementById("progress-wrap");
  const fill = document.getElementById("progress-fill");
  const text = document.getElementById("progress-text");
  const mode = "strict";
  const autoExport = workflow.value !== "classic";
  const algorithm = DL_ALGO_VISIBLE && workflow.value === "dl_algo" ? "dl_algo" : "classic";
  if (algorithm === "dl_algo") {
    await loadAlgorithms();
    if (!dlCapability?.available) {
      text.textContent = dlCapability?.reason || "DL Algo is not ready yet.";
      wrap.classList.remove("hidden");
      return;
    }
  }
  if (!keepRanges.length) {
    text.textContent = "Keep at least one section before starting analysis.";
    wrap.classList.remove("hidden");
    return;
  }

  wrap.classList.remove("hidden");
  startBtn.disabled = true;
  workflow.disabled = true;
  text.textContent = "Uploading original video…";
  fill.style.width = "4%";

  try {
    const formData = new FormData();
    formData.append("file", file);
    const upload = await fetch("/api/upload", { method: "POST", body: formData });
    if (!upload.ok) throw new Error(await upload.text());
    let { path } = await upload.json();

    const fullSourceKept = (
      keepRanges.length === 1
      && keepRanges[0].start <= 0.001
      && keepRanges[0].end >= sourceDuration - 0.001
    );
    if (!fullSourceKept) {
      text.textContent = "Creating cleaned match from kept sections…";
      fill.style.width = "7%";
      const preprocess = await fetch("/api/preprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_path: path, ranges: keepRanges }),
      });
      if (!preprocess.ok) throw new Error(await preprocess.text());
      ({ path } = await preprocess.json());
    }

    text.textContent = "Starting AI analysis…";
    fill.style.width = "10%";
    const start = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_path: path, mode, algorithm,
        selection: workflow.value === "pots_only" ? "pots_only" : "all", resume: true, auto_export: autoExport }),
    });
    if (!start.ok) throw new Error(await start.text());
    const { job_id: jobId } = await start.json();
    text.textContent = `Job ${jobId} running…`;

    const poll = setInterval(async () => {
      const response = await fetch(`/api/jobs/${jobId}/progress`);
      const metadata = await response.json();
      const progress = Math.max(10, (metadata.progress || 0) * 100);
      fill.style.width = `${progress}%`;
      text.textContent = `${metadata.status}: ${metadata.message || ""}`;
      if (["ready_for_review", "completed", "failed", "interrupted", "cancelled"].includes(metadata.status)) {
        clearInterval(poll);
        workflow.disabled = false;
        updateStartAvailability();
        refreshJobs();
        if (["failed", "interrupted", "cancelled"].includes(metadata.status)) {
          text.textContent = `Processing stopped: ${metadata.error || metadata.message}. Use Continue in Recent jobs.`;
        } else if (metadata.status === "completed") {
          text.textContent = metadata.message || "Finished video ready.";
          if (metadata.download_url) {
            const download = document.createElement("a");
            download.href = metadata.download_url;
            download.textContent = " Download finished video";
            text.appendChild(download);
          }
        } else {
          window.location.href = `/review/${jobId}?editor=classic`;
        }
      }
    }, 1500);
  } catch (error) {
    text.textContent = `Error: ${error.message || error}`;
    workflow.disabled = false;
    updateStartAvailability();
  }
});

document.getElementById("jobs").addEventListener("click", (event) => {
  const continueButton = event.target.closest(".btn-continue");
  if (continueButton?.dataset.jobId) {
    continueJob(continueButton.dataset.jobId, continueButton);
    return;
  }
  const capcutButton = event.target.closest(".btn-capcut");
  if (capcutButton?.dataset.jobId) {
    const item = capcutButton.closest(".job-item");
    item.dataset.capcutActive = "true";
    let status = item.querySelector(".capcut-status");
    if (!status) {
      status = document.createElement("p");
      status.className = "capcut-status muted";
      status.setAttribute("role", "status");
      item.querySelector("div").appendChild(status);
    }
    window.snookerCapCut.open(capcutButton.dataset.jobId, capcutButton, status).finally(() => {
      capcutJobMessages.set(capcutButton.dataset.jobId, status.textContent);
      delete item.dataset.capcutActive;
    });
    return;
  }
  const btn = event.target.closest(".btn-restart");
  if (btn && btn.dataset.jobId) {
    restartJob(btn.dataset.jobId);
  }
});

refreshJobs();
setInterval(refreshJobs, 10000);
