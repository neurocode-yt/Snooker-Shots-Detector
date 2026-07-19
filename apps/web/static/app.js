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

let sourceDuration = 0;
let sections = [];
let selectedSectionId = null;
let nextSectionId = 1;
let timelineZoom = 1;
let editHistory = [];
let redoHistory = [];
let localVideoUrl = null;
let draggingPlayhead = false;

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

function updatePlayhead() {
  const time = Math.max(0, Math.min(sourceDuration, sourcePreview.currentTime || 0));
  timelinePlayhead.style.left = `${sourceDuration ? (time / sourceDuration) * 100 : 0}%`;
  document.getElementById("source-clock").textContent = `${fmt(time)} / ${fmt(sourceDuration)}`;
}

function renderRuler() {
  const interval = niceTickInterval(sourceDuration / Math.max(8, timelineZoom * 8));
  const ticks = [];
  for (let time = 0; time <= sourceDuration + interval * 0.25; time += interval) {
    const bounded = Math.min(time, sourceDuration);
    const left = sourceDuration ? (bounded / sourceDuration) * 100 : 0;
    ticks.push(`<span class="pre-tick" style="left:${left}%"><i></i><b>${fmtRuler(bounded)}</b></span>`);
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
  startBtn.disabled = false;
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

function seekFromPlayheadPointer(event) {
  const viewportRect = timelineViewport.getBoundingClientRect();
  const edgeSize = 36;
  if (event.clientX < viewportRect.left + edgeSize) {
    timelineViewport.scrollLeft = Math.max(0, timelineViewport.scrollLeft - 24);
  } else if (event.clientX > viewportRect.right - edgeSize) {
    timelineViewport.scrollLeft += 24;
  }
  const contentRect = timelineContent.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - contentRect.left) / contentRect.width));
  const time = ratio * sourceDuration;
  sourcePreview.currentTime = time;
  const selected = sectionAt(time);
  if (selected) selectedSectionId = selected.id;
  updatePlayhead();
}

timelinePlayhead.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  event.stopPropagation();
  draggingPlayhead = true;
  timelinePlayhead.classList.add("dragging");
  timelinePlayhead.setPointerCapture(event.pointerId);
  seekFromPlayheadPointer(event);
});
timelinePlayhead.addEventListener("pointermove", (event) => {
  if (draggingPlayhead) seekFromPlayheadPointer(event);
});
timelinePlayhead.addEventListener("pointerup", (event) => {
  if (!draggingPlayhead) return;
  draggingPlayhead = false;
  timelinePlayhead.classList.remove("dragging");
  if (timelinePlayhead.hasPointerCapture(event.pointerId)) {
    timelinePlayhead.releasePointerCapture(event.pointerId);
  }
  renderEditor();
});
timelinePlayhead.addEventListener("pointercancel", () => {
  draggingPlayhead = false;
  timelinePlayhead.classList.remove("dragging");
});
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

async function refreshJobs() {
  const el = document.getElementById("jobs");
  try {
    const res = await fetch("/api/jobs");
    const jobs = await res.json();
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
          <div class="meta">${job.status || ""} · ${(job.progress * 100 || 0).toFixed(0)}% · ${job.message || ""}</div>
        </div>
        <a href="/review/${job.job_id}">Review</a>
      </div>`,
      )
      .join("");
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
sourcePreview.addEventListener("timeupdate", updatePlayhead);
sourcePreview.addEventListener("seeked", updatePlayhead);
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
  const mode = document.getElementById("mode").value;
  if (!keepRanges.length) {
    text.textContent = "Keep at least one section before starting analysis.";
    wrap.classList.remove("hidden");
    return;
  }

  wrap.classList.remove("hidden");
  startBtn.disabled = true;
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
      body: JSON.stringify({ source_path: path, mode, resume: true }),
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
      if (["ready_for_review", "completed", "failed"].includes(metadata.status)) {
        clearInterval(poll);
        startBtn.disabled = false;
        refreshJobs();
        if (metadata.status === "failed") {
          text.textContent = `Failed: ${metadata.error || metadata.message}`;
        } else {
          text.textContent = "Done — opening review…";
          window.location.href = `/review/${jobId}`;
        }
      }
    }, 1500);
  } catch (error) {
    text.textContent = `Error: ${error.message || error}`;
    startBtn.disabled = false;
  }
});

refreshJobs();
setInterval(refreshJobs, 10000);
