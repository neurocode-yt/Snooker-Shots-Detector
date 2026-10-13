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
const UI = window.snookerUI;
const workflowRadios = [...document.querySelectorAll('[name="highlight-workflow"]')];
function syncWorkflowChoices() {
  workflowRadios.forEach((radio) => {
    radio.checked = radio.value === workflow.value;
    radio.disabled = workflow.disabled || Boolean([...workflow.options].find((option) => option.value === radio.value)?.disabled);
  });
}
workflowRadios.forEach((radio) => radio.addEventListener("change", () => {
  workflow.value = radio.value;
  workflow.dispatchEvent(new Event("change"));
}));
// Keep DL available in the backend while its interface is temporarily hidden.
const DL_ALGO_VISIBLE = false;
if (DL_ALGO_VISIBLE) {
  workflow.appendChild(document.getElementById("dl-workflow-option").content.cloneNode(true));
}
let dlCapability = null;
let potsAvailable = false;
let pointsAvailable = false;
let pointsStatus = "Checking points highlights…";
let potsStatus = "Checking potted-shots mode…";

async function loadShotSelections() {
  try {
    const response = await fetch("/api/shot-selections");
    if (!response.ok) throw new Error("Restart the app server to enable Potted shots only.");
    const data = await response.json();
    const capability = data.selections.find((item) => item.id === "pots_only");
    potsAvailable = Boolean(capability?.available);
    potsStatus = capability?.reason || "Potted shots only is unavailable.";
    const points = data.selections.find((item) => item.id === "pots_and_fouls");
    pointsAvailable = Boolean(points?.available);
    pointsStatus = points?.reason || "Restart the app server to enable pots + fouls highlights.";
  } catch (error) {
    potsAvailable = false;
    potsStatus = error.message || String(error);
    pointsAvailable = false;
    pointsStatus = "Restart the app server to enable pots + fouls highlights.";
  }
  const option = document.getElementById("pots-workflow-option");
  option.disabled = !potsAvailable;
  option.title = potsStatus;
  document.getElementById("points-workflow-option").disabled = !pointsAvailable;
  document.getElementById("points-workflow-option").title = pointsStatus;
  updateWorkflow();
  refreshJobs();
}

function updateStartAvailability() {
  startBtn.disabled = (!sourceDuration && !previewUnavailable) || !fileInput.files[0] || workflow.disabled
    || (workflow.value === "pots_only" && !potsAvailable)
    || (workflow.value === "pots_and_fouls" && !pointsAvailable)
    || (workflow.value === "dl_algo" && !dlCapability?.available);
  syncWorkflowChoices();
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
  const points = workflow.value === "pots_and_fouls";
  document.getElementById("workflow-heading").textContent = classic
    ? "Prepare your match" : dl ? "DL Algo" : "Create highlights";
  document.getElementById("workflow-description").textContent = classic
    ? "Detect shots, then choose and trim your highlights in the editor."
    : points ? "Keep pots, fouls and confirmed points increases. Shots without points are skipped; unclear scores are flagged for review."
    : pots ? "Keep confirmed scoring pots using stable scores and the player's turn. Misses and fouls are excluded; unreadable scores are flagged for review."
    : dl ? "Create highlights with DL Algo. Results are saved as a separate job."
      : "Detect shots and create a finished highlights video automatically.";
  const status = document.getElementById("algorithm-status");
  status.classList.toggle("hidden", !dl && (!pots || potsAvailable) && (!points || pointsAvailable));
  status.textContent = points ? pointsStatus : pots ? potsStatus : dlCapability?.available
    ? "Experimental DL Algo is ready for comparison. Accuracy on new matches has not been verified."
    : dlCapability?.reason || "Checking DL Algo readiness…";
  document.getElementById("pre-editor-heading").textContent = classic
    ? "Prepare match before analysis" : "Optional source trimming";
  document.getElementById("pre-editor-description").textContent = classic
    ? "Split around frame breaks and delete those sections, or keep the whole video. Your original video is never changed."
    : "Keep the whole video and start. Shot detection, cutting, and export run automatically.";
  startBtn.textContent = classic ? "Prepare & start analysis"
    : dl ? "Create with DL Algo" : points ? "Create pots + fouls highlights" : pots ? "Create pots-only highlights" : "Create highlights";
  updateStartAvailability();
}

workflow.addEventListener("change", () => {
  updateWorkflow();
  if (["pots_only", "pots_and_fouls"].includes(workflow.value)) loadShotSelections();
  if (workflow.value === "dl_algo") loadAlgorithms();
});

let sourceDuration = 0;
let previewUnavailable = false;
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
    sourceDuration - keptDuration < .001 ? "Full match selected" : `${UI.time(keptDuration)} kept`
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
  document.getElementById("edit-summary").textContent = "Full match selected";
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
const pointsJobMessages = new Map();
let visibleJobs = 20;

function videoName(job) {
  return UI.filename(job);
}

async function createPointsHighlights(jobId, button) {
  const item = button.closest(".job-item");
  item.dataset.pointsActive = "true";
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    const response = await fetch(`/api/jobs/${jobId}/points-highlights`, {method: "POST"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not create points highlights");
    pointsJobMessages.set(jobId, data.status === "ready"
      ? "Points highlights are ready in the jobs list."
      : "Creating points highlights from saved shots; the original video stays available.");
  } catch (error) {
    pointsJobMessages.set(jobId, error.message || String(error));
  } finally {
    delete item.dataset.pointsActive;
    await refreshJobs();
  }
}

function escapeJobText(value) {
  return UI.escape(value);
}

async function continueJob(jobId, button) {
  const item = button.closest(".job-item");
  item.dataset.continueActive = "true";
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
  } finally {
    delete item.dataset.continueActive;
    refreshJobs();
  }
}

let libraryJobs = [];
let libraryFilter = "all";
let libraryCapcut = {available: false, reason: "Checking CapCut…"};
let libraryLoading = false;

function renderJobs() {
  const el = document.getElementById("jobs");
  if (el.querySelector("[data-capcut-active], [data-points-active], [data-continue-active], .menu[open]")) return;
  const query = document.getElementById("job-search").value;
  const jobs = libraryJobs.filter((job) => UI.matches(job, libraryFilter, query));
  document.getElementById("jobs-count").textContent = libraryJobs.length;
  document.querySelectorAll("[data-filter-count]").forEach((count) => {
    count.textContent = libraryJobs.filter((job) => UI.matches(job, count.dataset.filterCount)).length;
  });
  document.querySelectorAll("[data-job-filter]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.jobFilter === libraryFilter)));
  const more = document.getElementById("show-more-jobs");
  more.classList.toggle("hidden", jobs.length <= visibleJobs);
  more.textContent = `Show ${Math.min(20, jobs.length - visibleJobs)} more videos`;
  if (!jobs.length) {
    el.innerHTML = `<div class="empty-state"><strong>${libraryJobs.length ? "No matching videos" : "Your highlights will appear here"}</strong><p>${libraryJobs.length ? "Try another name or switch the filter." : "Choose a match above to create your first video."}</p></div>`;
    return;
  }
  const focused = document.activeElement?.closest("[data-job-id]");
  const focusId = focused?.dataset.jobId;
  const focusClass = focused?.className;
  el.innerHTML = jobs.slice(0, visibleJobs).map((job) => {
    const id = UI.escape(job.job_id);
    const state = UI.status(job);
    const actions = UI.actions(job, pointsAvailable);
    const progress = Math.max(0, Math.min(100, Math.round(Number(job.progress || 0) * 100)));
    const summary = job.pot_selection_summary || {};
    const unknown = Number(summary.unknown || 0);
    const shotCount = Number(job.shots_detected || 0);
    const countText = Number.isFinite(summary.included) ? `${summary.included} of ${shotCount || Object.values(summary).filter(Number.isInteger).slice(0, 5).reduce((a, b) => a + b, 0)} shots selected` : shotCount ? `${shotCount} shots detected` : "";
    const date = job.created_at ? new Intl.DateTimeFormat("en-GB", {day: "numeric", month: "short", year: "numeric"}).format(new Date(job.created_at * 1000)) : "";
    const editor = `/review/${encodeURIComponent(job.job_id)}?editor=classic`;
    const primary = UI.active(job) ? (actions.editor ? `<a class="button secondary" href="${editor}">View shots</a>` : "") : actions.resume ? `<button class="btn-continue primary" data-job-id="${id}">Continue processing</button>`
      : job.download_url ? `<a class="button primary" href="${UI.escape(job.download_url)}">Download video <span aria-hidden="true">↓</span></a>`
      : actions.editor ? `<a class="button primary" href="${editor}">Review shots</a>` : "";
    return `<article class="job-item" data-library-job="${id}">
      <div class="job-icon ${state.tone}" aria-hidden="true">${UI.active(job) ? "◷" : UI.stopped(job) ? "!" : "▶"}</div>
      <div class="job-copy"><div class="job-title-row"><h3 title="${UI.escape(videoName(job))}">${UI.escape(UI.title(job))}</h3><span class="status-badge ${state.tone}">${state.label}</span></div>
        <div class="job-meta"><span class="mode-badge">${UI.escape(UI.mode(job))}</span>${countText ? `<span>${countText}</span>` : ""}<span>${date}</span></div>
        ${UI.active(job) ? `<div class="job-progress"><span>${UI.escape(UI.phase(job))}</span><span>${progress}%</span><div role="progressbar" aria-label="Video processing" aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100"><i style="width:${progress}%"></i></div></div>` : ""}
        ${UI.stopped(job) ? '<p class="job-warning">Processing stopped. Saved work is preserved.</p>' : job.export_needs_update ? '<p class="job-warning">Edits saved. Open the editor to export the updated video.</p>' : unknown ? `<p class="job-warning">${unknown} ${unknown === 1 ? "shot needs" : "shots need"} a score check in the editor.</p>` : ""}
        ${pointsJobMessages.has(job.job_id) ? `<p class="job-feedback" role="status">${UI.escape(pointsJobMessages.get(job.job_id))}</p>` : ""}
        ${capcutJobMessages.has(job.job_id) ? `<p class="capcut-status job-feedback" role="status">${UI.escape(capcutJobMessages.get(job.job_id))}</p>` : ""}
      </div>
      <div class="job-actions">${primary}
        ${job.download_url && !UI.active(job) && actions.editor ? `<a class="button secondary" href="${editor}">Open editor</a>` : ""}
        ${actions.points ? `<button class="btn-points secondary" data-job-id="${id}" title="Create a separate version from saved shots">Pots + fouls</button>` : ""}
        <details class="menu job-menu"><summary class="button icon-button" aria-label="More options for ${UI.escape(UI.title(job))}">•••</summary><div class="menu-panel">
          ${actions.editor && !job.download_url && primary ? `<a href="${editor}">Open editor</a>` : ""}
          ${UI.finished(job) ? `<button class="btn-capcut" data-job-id="${id}" ${libraryCapcut.available ? "" : "disabled"} title="${UI.escape(libraryCapcut.reason || "Open as editable clips")}">Open in CapCut</button>` : ""}
          <button class="btn-restart" data-job-id="${id}" ${actions.restart ? "" : "disabled"}>Restart analysis</button>
          <div class="menu-meta"><strong>Processing details</strong><span>${UI.escape(job.message || "")}</span>${job.error ? `<span class="error-detail">${UI.escape(job.error)}</span>` : ""}${Number.isFinite(job.checkpoint?.source_time) ? `<span>Saved through ${fmt(job.checkpoint.source_time)}${job.checkpoint.duration ? ` of ${fmt(job.checkpoint.duration)}` : ""}</span>` : ""}<span class="job-id">${id}</span></div>
        </div></details>
      </div></article>`;
  }).join("");
  el.querySelectorAll(".job-item").forEach((item) => item.querySelectorAll("a, button, summary").forEach((control) => {
    control.dataset.jobId = item.dataset.libraryJob;
  }));
  el.querySelectorAll(".menu").forEach((menu) => menu.addEventListener("toggle", () => { if (!menu.open) renderJobs(); }));
  if (focusId && focusClass) {
    [...el.querySelectorAll("[data-job-id]")].find((node) => node.dataset.jobId === focusId && node.className === focusClass)?.focus({preventScroll: true});
  }
}

async function refreshJobs() {
  if (libraryLoading) return;
  libraryLoading = true;
  try {
    const [res, capcut] = await Promise.all([fetch("/api/jobs"), window.snookerCapCut.load()]);
    if (!res.ok) throw new Error("The video library could not be loaded. Retrying shortly.");
    libraryJobs = (await res.json()).filter((job) => DL_ALGO_VISIBLE || job.algorithm !== "dl_algo")
      .sort((left, right) => Number(right.created_at || 0) - Number(left.created_at || 0));
    libraryCapcut = capcut;
    document.getElementById("library-status").classList.add("hidden");
    renderJobs();
  } catch (error) {
    const status = document.getElementById("library-status");
    status.textContent = error.message || String(error);
    status.classList.remove("hidden");
  } finally { libraryLoading = false; }
}

const dropZone = document.getElementById("file-drop-zone");
["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  if (!fileInput.disabled) dropZone.classList.add("drag-over");
}));
["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, () => dropZone.classList.remove("drag-over")));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  if (fileInput.disabled || !event.dataTransfer.files.length) return;
  const files = new DataTransfer();
  files.items.add(event.dataTransfer.files[0]);
  fileInput.files = files.files;
  fileInput.dispatchEvent(new Event("change"));
});
document.getElementById("job-search").addEventListener("input", () => { visibleJobs = 20; renderJobs(); });
document.querySelectorAll("[data-job-filter]").forEach((button) => button.addEventListener("click", () => {
  libraryFilter = button.dataset.jobFilter;
  visibleJobs = 20;
  document.querySelectorAll(".job-menu[open]").forEach((menu) => { menu.open = false; });
  renderJobs();
}));
document.addEventListener("click", (event) => {
  document.querySelectorAll(".menu[open]").forEach((menu) => {
    if (!menu.contains(event.target)) menu.open = false;
  });
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") document.querySelectorAll(".menu[open]").forEach((menu) => { menu.open = false; menu.querySelector("summary").focus(); });
});

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  previewUnavailable = false;
  document.getElementById("file-name").textContent = file ? file.name : "Choose a video or drop it here";
  document.getElementById("file-meta").textContent = file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · Reading video…` : "MP4, MOV, MKV, AVI or WebM";
  document.getElementById("file-action").textContent = file ? "Change video" : "Browse files";
  dropZone.classList.toggle("has-file", Boolean(file));
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
    previewUnavailable = false;
    initializeEditor(sourcePreview.duration);
    const file = fileInput.files[0];
    document.getElementById("file-meta").textContent = `${UI.time(sourcePreview.duration)} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
    document.getElementById("trim-options").open = false;
  }
});
sourcePreview.addEventListener("error", () => {
  if (!fileInput.files[0]) return;
  previewUnavailable = true;
  preEditor.classList.add("hidden");
  document.getElementById("file-meta").textContent = "Browser preview unavailable. You can still process the full match.";
  updateStartAvailability();
});
document.getElementById("trim-options").addEventListener("toggle", (event) => {
  if (event.currentTarget.open) requestAnimationFrame(renderEditor);
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
  if (!sourceDuration || preEditor.classList.contains("hidden") || !document.getElementById("trim-options").open) return;
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
  if (!file || (!sourceDuration && !previewUnavailable)) return;
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
  if (!keepRanges.length && !previewUnavailable) {
    text.textContent = "Keep at least one section before starting analysis.";
    wrap.classList.remove("hidden");
    return;
  }

  wrap.classList.remove("hidden");
  startBtn.disabled = true;
  workflow.disabled = true;
  fileInput.disabled = true;
  syncWorkflowChoices();
  text.textContent = "Uploading original video…";
  fill.style.width = "4%";

  try {
    const formData = new FormData();
    formData.append("file", file);
    const upload = await fetch("/api/upload", { method: "POST", body: formData });
    if (!upload.ok) throw new Error(await upload.text());
    let { path } = await upload.json();

    const fullSourceKept = previewUnavailable || (
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

    text.textContent = "Finding shots…";
    fill.style.width = "10%";
    const start = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_path: path, mode, algorithm,
        selection: ["pots_only", "pots_and_fouls"].includes(workflow.value) ? workflow.value : "all", resume: true, auto_export: autoExport }),
    });
    if (!start.ok) throw new Error(await start.text());
    const { job_id: jobId } = await start.json();
    text.textContent = "Your match is processing. You can follow it in Your videos below.";

    let pollBusy = false;
    const poll = setInterval(async () => {
      if (pollBusy) return;
      pollBusy = true;
      try {
        const response = await fetch(`/api/jobs/${jobId}/progress`);
        if (!response.ok) throw new Error("Unable to fetch progress");
        const metadata = await response.json();
        const progress = Math.max(10, (metadata.progress || 0) * 100);
        fill.style.width = `${progress}%`;
        text.textContent = `${UI.phase(metadata)} · ${Math.round(progress)}%`;
        fill.parentElement.setAttribute("aria-valuenow", String(Math.round(progress)));
        if (["ready_for_review", "completed", "failed", "interrupted", "cancelled"].includes(metadata.status)) {
          clearInterval(poll);
          workflow.disabled = false;
          fileInput.disabled = false;
          updateStartAvailability();
          refreshJobs();
          if (["failed", "interrupted", "cancelled"].includes(metadata.status)) {
            text.textContent = "Processing stopped. Saved work is preserved; use Continue processing in Your videos.";
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

      } catch (error) {
        text.textContent = "Connection lost. Your match may still be processing; checking again shortly.";
      } finally { pollBusy = false; }
    }, 1500);
  } catch (error) {
    text.textContent = `Error: ${error.message || error}`;
    workflow.disabled = false;
    fileInput.disabled = false;
    updateStartAvailability();
  }
});

document.getElementById("jobs").addEventListener("click", (event) => {
  const pointsButton = event.target.closest(".btn-points");
  if (pointsButton?.dataset.jobId) {
    createPointsHighlights(pointsButton.dataset.jobId, pointsButton);
    return;
  }
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
      item.querySelector(".job-copy").appendChild(status);
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
document.getElementById("show-more-jobs").addEventListener("click", () => {
  visibleJobs += 20;
  renderJobs();
});
setInterval(refreshJobs, 10000);
