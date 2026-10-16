const jobId = window.SNOOKER_JOB_ID;
const UI = window.snookerUI;
const video = document.getElementById("video");
const selectedVideo = document.getElementById("selected-video");
const previewPlayBtn = document.getElementById("preview-play");
const previewSeek = document.getElementById("preview-seek");
const previewClock = document.getElementById("preview-clock");
const previewCount = document.getElementById("preview-count");
const sourceEditor = document.getElementById("source-editor");
const highlightsPanel = document.getElementById("highlights-panel");
const shotFilter = document.getElementById("shot-filter");
let currentEditorView = "highlights";
function setEditorView(view, fromShot = false) {
  const source = view === "source";
  if (source && currentEditorView !== "source" && !fromShot && selectedVideo.readyState >= HTMLMediaElement.HAVE_METADATA) seekTo(selectedVideo.currentTime);
  if (source) selectedVideo.pause(); else video.pause();
  highlightsPanel.classList.toggle("hidden", source);
  sourceEditor.classList.toggle("hidden", !source);
  currentEditorView = view;
  ["highlights", "source"].forEach((name) => {
    const tab = document.getElementById(`${name}-tab`);
    tab.setAttribute("aria-selected", String((name === "source") === source));
    tab.tabIndex = (name === "source") === source ? 0 : -1;
  });
}
["highlights", "source"].forEach((name) => {
  const tab = document.getElementById(`${name}-tab`);
  tab.addEventListener("click", () => setEditorView(name));
  tab.addEventListener("keydown", (event) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const target = event.key === "Home" ? "highlights" : event.key === "End" ? "source" : name === "source" ? "highlights" : "source";
      setEditorView(target);
      document.getElementById(`${target}-tab`).focus();
    }
  });
});
const shotList = document.getElementById("shot-list");
const timeline = document.getElementById("timeline");
const stats = document.getElementById("stats");
const toast = document.getElementById("toast");
const exportCombinedBtn = document.getElementById("export-combined-btn");
const exportClipsBtn = document.getElementById("export-clips-btn");
const openExportBtn = document.getElementById("open-export-btn");
const openCapCutBtn = document.getElementById("open-capcut-btn");
const makeShortsBtn = document.getElementById("make-shorts-btn");
makeShortsBtn.addEventListener("click", () => window.snookerShorts.open(jobId, makeShortsBtn));
const capCutStatus = document.getElementById("capcut-status");
const exportStatus = document.getElementById("export-status");
const combinedDownloadLink = document.getElementById("combined-download-link");

window.snookerCapCut.load().then((capability) => {
  openCapCutBtn.title = capability.available
    ? "Create a new CapCut project with included live shots as separate editable clips"
    : capability.reason;
  if (meta.job_id) syncJobState();
});
openCapCutBtn.addEventListener("click", () => {
  const request = window.snookerCapCut.open(jobId, openCapCutBtn, capCutStatus);
  syncJobState();
  request.finally(syncJobState);
});

let shots = [];
let activeId = null;
let duration = 0;
let meta = {};
let exportBusy = false;
let pointsCapabilityAvailable = false;
let editBusy = false;
let analysisRequestBusy = false;
let metadataEpoch = 0;
let loadedTimelineMeta = null;
let statusTimer = null;
let statusInFlight = false;
let statusRefreshQueued = false;
let timelineBusy = false;
let timelineRefreshRequired = false;
let pendingSeek = null;
let playbackStopAt = null;
let previewSegments = [];
let previewIndex = 0;
let previewTotal = 0;
let previewSignature = "";
let previewPendingSourceTime = null;
let previewFinished = false;
let previewAdvancing = false;

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 2200);
}

function setExportStatus(message, state = "") {
  exportStatus.textContent = message;
  exportStatus.classList.toggle("hidden", !message);
  exportStatus.dataset.state = state;
}

function fmt(t) {
  if (!Number.isFinite(t)) return "00:00.000";
  const hours = Math.floor(t / 3600);
  const m = Math.floor(t / 60) % 60;
  const s = t % 60;
  return `${hours ? `${hours}:` : ""}${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
}

function previewPosition() {
  if (!previewSegments.length) return 0;
  if (previewFinished) return previewTotal;
  const before = previewSegments
    .slice(0, previewIndex)
    .reduce((sum, shot) => sum + Math.max(0, shot.clip_end - shot.clip_start), 0);
  const shot = previewSegments[previewIndex];
  const within = Math.max(0, Math.min(
    shot.clip_end - shot.clip_start,
    selectedVideo.currentTime - shot.clip_start,
  ));
  return Math.min(previewTotal, before + within);
}

function updatePreviewControls() {
  const position = previewPosition();
  previewSeek.max = String(previewTotal);
  previewSeek.value = String(position);
  previewSeek.disabled = previewSegments.length === 0;
  previewPlayBtn.disabled = previewSegments.length === 0;
  previewPlayBtn.textContent = selectedVideo.paused ? (previewFinished ? "Replay" : "Play") : "Pause";
  previewClock.textContent = `${fmt(position)} / ${fmt(previewTotal)}`;
  previewCount.textContent = `${previewSegments.length} ${previewSegments.length === 1 ? "shot" : "shots"}`;
}

function setPreviewSourceTime(sourceTime) {
  if (selectedVideo.readyState < HTMLMediaElement.HAVE_METADATA) {
    previewPendingSourceTime = sourceTime;
    return;
  }
  previewPendingSourceTime = null;
  selectedVideo.currentTime = sourceTime;
}

function seekPreviewCombined(requestedTime) {
  if (!previewSegments.length) return;
  const target = Math.max(0, Math.min(previewTotal, Number(requestedTime) || 0));
  let elapsed = 0;
  let index = previewSegments.length - 1;
  let sourceTime = previewSegments[index].clip_end;

  for (let i = 0; i < previewSegments.length; i += 1) {
    const shot = previewSegments[i];
    const shotDuration = Math.max(0, shot.clip_end - shot.clip_start);
    if (target < elapsed + shotDuration || i === previewSegments.length - 1) {
      index = i;
      sourceTime = shot.clip_start + Math.min(shotDuration, target - elapsed);
      break;
    }
    elapsed += shotDuration;
  }

  previewIndex = index;
  previewFinished = target >= previewTotal - 0.001;
  setPreviewSourceTime(sourceTime);
  updatePreviewControls();
}

function refreshPreviewPlaylist() {
  const selected = shots
    .filter((shot) => shot.included && shot.clip_end > shot.clip_start)
    .slice()
    .sort((a, b) => a.clip_start - b.clip_start);
  const signature = selected
    .map((shot) => `${shot.shot_id}:${shot.clip_start}:${shot.clip_end}`)
    .join("|");
  if (signature === previewSignature) {
    updatePreviewControls();
    return;
  }

  selectedVideo.pause();
  previewSignature = signature;
  previewSegments = selected;
  previewTotal = selected.reduce(
    (sum, shot) => sum + Math.max(0, shot.clip_end - shot.clip_start),
    0,
  );
  previewIndex = 0;
  previewFinished = false;
  if (selected.length) setPreviewSourceTime(selected[0].clip_start);
  updatePreviewControls();
}

async function togglePreviewPlayback() {
  if (!previewSegments.length) {
    showToast("Include at least one shot to preview");
    return;
  }
  if (!selectedVideo.paused) {
    selectedVideo.pause();
    return;
  }
  if (previewFinished) seekPreviewCombined(0);
  const shot = previewSegments[previewIndex];
  if (
    selectedVideo.currentTime < shot.clip_start - 0.02
    || selectedVideo.currentTime >= shot.clip_end - 0.01
  ) {
    setPreviewSourceTime(shot.clip_start);
  }
  try {
    await selectedVideo.play();
  } catch (_err) {
    showToast("Preview is still loading; try again");
  }
}

function advancePreview() {
  if (previewAdvancing || previewFinished) return;
  previewAdvancing = true;
  if (previewIndex >= previewSegments.length - 1) {
    previewFinished = true;
    selectedVideo.pause();
    previewAdvancing = false;
    updatePreviewControls();
    return;
  }
  previewIndex += 1;
  setPreviewSourceTime(previewSegments[previewIndex].clip_start);
  previewAdvancing = false;
  updatePreviewControls();
}

function monitorPreview() {
  if (!selectedVideo.paused && previewSegments.length && !selectedVideo.seeking) {
    const shot = previewSegments[previewIndex];
    if (selectedVideo.currentTime >= shot.clip_end - 0.02) advancePreview();
  }
  updatePreviewControls();
  window.requestAnimationFrame(monitorPreview);
}

selectedVideo.addEventListener("loadedmetadata", () => {
  if (previewPendingSourceTime !== null) {
    const target = previewPendingSourceTime;
    previewPendingSourceTime = null;
    selectedVideo.currentTime = target;
  }
});
selectedVideo.addEventListener("click", togglePreviewPlayback);
previewPlayBtn.addEventListener("click", togglePreviewPlayback);
previewSeek.addEventListener("input", () => seekPreviewCombined(previewSeek.value));
document.getElementById("preview-mute").addEventListener("click", (event) => {
  selectedVideo.muted = !selectedVideo.muted;
  event.currentTarget.textContent = selectedVideo.muted ? "Unmute" : "Mute";
});
document.getElementById("preview-fullscreen").addEventListener("click", () => {
  if (selectedVideo.requestFullscreen) selectedVideo.requestFullscreen();
});
monitorPreview();

function confClass(s) {
  if (!s.included) return "excluded";
  if (s.possible_replay) return "replay";
  if (s.manual_review_required) return "needs-review";
  return "";
}

function visibleShots() {
  return shots.filter((shot) => shotFilter.value === "included" ? shot.included
    : shotFilter.value === "excluded" ? !shot.included
    : shotFilter.value === "review" ? shot.manual_review_required : true);
}
shotFilter.addEventListener("change", render);

function render() {
  const focusedShot = document.activeElement?.closest(".shot-open")?.closest("[data-id]")?.dataset.id;
  const list = visibleShots();
  shotList.innerHTML = list.map((shot) => {
    const classes = [confClass(shot), shot.manual_review_required ? "needs-review" : "", shot.shot_id === activeId ? "active" : ""].filter(Boolean).join(" ");
    const outcome = {pot: "Pot", no_pot: "No points", foul: "Foul", points: "Points", unknown: "Unconfirmed"}[shot.pot_status];
    return `<li class="${classes}" data-id="${shot.shot_id}">
      <button class="shot-open" aria-label="Edit shot ${shot.shot_id}" ${shot.shot_id === activeId ? 'aria-current="true"' : ""}><span class="shot-number">${String(shot.shot_id).padStart(2, "0")}</span><span class="shot-copy"><strong>${UI.time(shot.clip_start)} → ${UI.time(shot.clip_end)}</strong><small>Strike ${fmt(shot.cue_strike)} · ${(shot.clip_end - shot.clip_start).toFixed(1)}s</small><span class="shot-badges">${outcome ? `<span class="shot-badge ${shot.pot_status}">${outcome}</span>` : ""}${shot.manual_review_required ? '<span class="shot-badge warning">Review</span>' : ""}${shot.possible_replay ? '<span class="shot-badge">Replay</span>' : ""}${!shot.included ? '<span class="shot-badge">Excluded</span>' : ""}</span></span></button>
      <label class="shot-include-label"><input type="checkbox" class="shot-include" data-shot-id="${shot.shot_id}" aria-label="Include shot ${shot.shot_id}" ${shot.included ? "checked" : ""} ${UI.active(meta) || exportBusy ? "disabled" : ""} /></label></li>`;
  }).join("") || '<li class="empty-state">No shots in this filter.</li>';

  timeline.innerHTML = shots
    .map((s) => {
      if (duration <= 0) return "";
      const left = (s.clip_start / duration) * 100;
      const width = Math.max(0.25, ((s.clip_end - s.clip_start) / duration) * 100);
      const strikeLeft = (s.cue_strike / duration) * 100;
      const cls = ["seg", confClass(s), s.shot_id === activeId ? "active" : ""]
        .filter(Boolean)
        .join(" ");
      return `<div class="${cls}" style="left:${left}%;width:${width}%" data-id="${s.shot_id}" title="Shot ${s.shot_id}"></div>
        <div class="mark ${confClass(s)}" style="left:${strikeLeft}%" title="Strike ${s.shot_id}"></div>`;
    })
    .join("");

  const included = shots.filter((s) => s.included);
  const edited = included.reduce((a, s) => a + Math.max(0, s.clip_end - s.clip_start), 0);
  stats.innerHTML = `<div><strong>${included.length} / ${shots.length}</strong><span>Shots included</span></div><div><strong>${UI.time(edited)}</strong><span>Selected footage</span></div><div><strong>${UI.time(duration)}</strong><span>Original match</span></div>`;
  document.getElementById("shot-total").textContent = shots.length;
  document.getElementById("shot-included-count").textContent = `${included.length} included`;
  syncJobState();

  const active = shots.find((s) => s.shot_id === activeId);
  document.getElementById("active-shot-label").textContent = active ? `Shot ${active.shot_id}` : "Shot controls";
  document.getElementById("toggle-include").textContent = active?.included ? "Exclude shot" : "Include shot";
  if (active) {
    document.getElementById("edit-start").value = active.clip_start.toFixed(3);
    document.getElementById("edit-strike").value = active.cue_strike.toFixed(3);
    document.getElementById("edit-end").value = active.clip_end.toFixed(3);
  }
  const endTrimState = window.snookerEndTrim.state(active);
  document.getElementById('shot-end-trim').value = endTrimState.seconds;
  document.getElementById('shot-end-trim-detail').textContent = endTrimState.description;
  refreshPreviewPlaylist();
  if (focusedShot) shotList.querySelector(`li[data-id="${focusedShot}"] .shot-open`)?.focus({preventScroll: true});
}

function syncJobState() {
  const busy = UI.active(meta) || exportBusy || editBusy || analysisRequestBusy || timelineBusy || timelineRefreshRequired || window.snookerCapCut.isPending(jobId);
  const hasSelection = shots.some((shot) => shot.included);
  exportCombinedBtn.disabled = exportClipsBtn.disabled = busy || !hasSelection;
  document.getElementById("video-title").textContent = UI.title(meta);
  document.getElementById("editor-subtitle").textContent = `${UI.mode(meta)} · ${UI.status(meta).label}`;
  document.getElementById("job-label").textContent = jobId;
  document.title = `${UI.title(meta)} — Shot editor`;
  document.getElementById("continue-analysis-btn").classList.toggle("hidden", !UI.actions(meta).resume);
  document.getElementById("continue-analysis-btn").disabled = busy;
  document.getElementById("continue-analysis-btn").textContent = meta.status === "paused" ? "Resume processing" : "Continue processing";
  document.getElementById("restart-analysis-btn").disabled = busy;
  document.getElementById("points-highlights-btn").disabled = busy || !UI.actions(meta, pointsCapabilityAvailable).points;
  document.getElementById("add-at-playhead").disabled = busy || !loadedTimelineMeta;
  openCapCutBtn.disabled = busy || !hasSelection;
  makeShortsBtn.disabled = busy || !hasSelection || !UI.finished(meta);
  updatePreviewControls();
  document.querySelectorAll(".shot-include").forEach((checkbox) => { checkbox.disabled = busy; });
  document.querySelectorAll(".boundary-edit input, .shot-tools button, .transport button").forEach((control) => { control.disabled = busy || !currentShot(); });
  document.querySelectorAll('.shot-end-trim input, .shot-end-trim button').forEach(control=>{control.disabled=busy||!currentShot();});
  document.querySelectorAll('.end-trim-panel input, .end-trim-panel button').forEach(control=>{control.disabled=busy||!hasSelection;});
  if (meta.download_url && !meta.export_needs_update && !busy) {
    combinedDownloadLink.href = meta.download_url;
    combinedDownloadLink.classList.remove("hidden");
  } else {
    combinedDownloadLink.classList.add("hidden");
    combinedDownloadLink.removeAttribute("href");
  }
}

async function readSavedShots(snapshot, {seek = false} = {}) {
  const epoch = metadataEpoch;
  timelineBusy = true;
  syncJobState();
  try {
    const response = await fetch(`/api/jobs/${jobId}/shots`);
    if (!response.ok) {
      if (!UI.active(snapshot)) setExportStatus("No saved shots are available for this video.", "error");
      return false;
    }
    const data = await response.json();
    if (epoch !== metadataEpoch) return false;
    shots = data.shots || [];
    duration = data.original_duration || 0;
    const videoUrl = `/api/jobs/${jobId}/video`;
    const mediaChanged = video.getAttribute("src") !== videoUrl;
    if (mediaChanged) {
      video.src = videoUrl;
      selectedVideo.src = videoUrl;
    }
    activeId = shots.some((shot) => shot.shot_id === activeId) ? activeId : shots[0]?.shot_id ?? null;
    if ((seek || mediaChanged) && currentShot()) seekTo(currentShot().clip_start);
    loadedTimelineMeta = {...snapshot};
    timelineRefreshRequired = false;
    render();
    return true;
  } finally {
    timelineBusy = false;
    syncJobState();
  }
}

function scheduleStatusCheck() {
  clearTimeout(statusTimer);
  statusTimer = null;
  if (statusRefreshQueued && !exportBusy && !editBusy && !analysisRequestBusy && !timelineBusy && !statusInFlight) {
    statusTimer = setTimeout(() => refreshJobStatus(), 0);
  } else if (UI.active(meta) || timelineRefreshRequired) statusTimer = setTimeout(() => refreshJobStatus(), 2000);
}

async function refreshJobStatus() {
  if (statusInFlight || exportBusy || editBusy || analysisRequestBusy || timelineBusy) {
    statusRefreshQueued = true;
    return;
  }
  statusRefreshQueued = false;
  statusInFlight = true;
  const epoch = metadataEpoch;
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) throw new Error("Could not read job status");
    const next = await response.json();
    if (epoch !== metadataEpoch || exportBusy || editBusy || analysisRequestBusy) return;
    const wasActive = UI.active(meta);
    meta = next;
    syncJobState();
    if (UI.active(meta)) {
      setExportStatus(`${UI.phase(meta)} · ${Math.round((meta.progress || 0) * 100)}%`, "working");
    } else if (UI.stopped(meta)) {
      setExportStatus("Processing stopped. Saved work is preserved; use Continue processing under More.", "error");
    } else if (meta.export_needs_update) {
      setExportStatus("Edits saved. Export video to update your highlights.", "ready");
    } else if (wasActive) {
      setExportStatus(meta.download_url ? "Your highlights are ready to download." : "Shots are ready to review.", "ready");
    }
    if (timelineRefreshRequired || UI.needsTimelineRefresh(meta, loadedTimelineMeta)) await readSavedShots(meta);
  } catch (error) {
    setExportStatus("Connection lost. Saved work is preserved; checking processing status again shortly.", "error");
  } finally {
    statusInFlight = false;
    scheduleStatusCheck();
  }
}

async function load() {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) { setExportStatus("This video could not be loaded. Return to Your videos and try again.", "error"); return; }
  meta = await response.json();
  if (meta.end_trim_settings) {
    for (const [name,key] of [['normal','seconds'],['long','long_seconds'],['threshold','long_threshold']])
      if (Number.isFinite(meta.end_trim_settings[key])) document.getElementById('end-trim-'+name).value=meta.end_trim_settings[key];
  }
  syncJobState();
  try {
    const capabilityResponse = await fetch("/api/shot-selections");
    if (capabilityResponse.ok) {
      const capabilities = await capabilityResponse.json();
      pointsCapabilityAvailable = capabilities.selections.some((item) => item.id === "pots_and_fouls" && item.available);
    }
  } catch (error) { /* Editing remains available while capabilities are offline. */ }
  if (UI.active(meta)) setExportStatus(`${UI.phase(meta)} · ${Math.round((meta.progress || 0) * 100)}%`, "working");
  else if (UI.stopped(meta)) setExportStatus("Processing stopped. Saved work is preserved; use Continue processing under More.", "error");
  else if (meta.export_needs_update) setExportStatus("Edits saved. Export video to update your highlights.", "ready");
  await readSavedShots(meta, {seek: loadedTimelineMeta === null});
  syncJobState();
  scheduleStatusCheck();
}

window.addEventListener("pagehide", () => { clearTimeout(statusTimer); });
window.addEventListener("pageshow", (event) => { if (event.persisted) refreshJobStatus(); });
window.addEventListener("snooker:work-changed", (event) => {
  if (event.detail.jobId === jobId) refreshJobStatus();
});

function seekTo(t) {
  if (!Number.isFinite(t)) return;
  const target = Math.max(0, t);
  if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
    pendingSeek = target;
    return;
  }
  pendingSeek = null;
  if (Math.abs(video.currentTime - target) > 0.01) video.currentTime = target;
}

video.addEventListener("loadedmetadata", () => {
  if (pendingSeek === null) return;
  const target = pendingSeek;
  pendingSeek = null;
  video.currentTime = target;
});

async function seekAndWait(t) {
  const target = Math.max(0, Number(t) || 0);
  if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
    pendingSeek = target;
    await new Promise((resolve) => video.addEventListener("loadedmetadata", resolve, { once: true }));
  }
  if (video.seeking) {
    await new Promise((resolve) => video.addEventListener("seeked", resolve, { once: true }));
  }
  if (Math.abs(video.currentTime - target) <= 0.01) return;
  await new Promise((resolve) => {
    video.addEventListener("seeked", resolve, { once: true });
    video.currentTime = target;
  });
}

function currentShot() {
  return shots.find((s) => s.shot_id === activeId);
}

shotList.addEventListener("click", (e) => {
  if (!e.target.closest(".shot-open")) return;
  const li = e.target.closest("li[data-id]");
  if (!li) return;
  activeId = Number(li.dataset.id);
  video.pause();
  playbackStopAt = null;
  const s = currentShot();
  if (s) seekTo(s.clip_start);
  setEditorView("source", true);
  render();
});

shotList.addEventListener("keydown", (event) => {
  const button = event.target.closest(".shot-open");
  if (!button || !["j", "k", "ArrowDown", "ArrowUp"].includes(event.key)) return;
  event.preventDefault();
  const list = visibleShots();
  const index = list.findIndex((shot) => shot.shot_id === Number(button.closest("[data-id]").dataset.id));
  const next = list[index + (["j", "ArrowDown"].includes(event.key) ? 1 : -1)];
  if (!next) return;
  activeId = next.shot_id;
  video.pause();
  playbackStopAt = null;
  seekTo(next.clip_start);
  setEditorView("source", true);
  render();
  shotList.querySelector(`li[data-id="${next.shot_id}"] .shot-open`)?.focus({preventScroll: true});
});

timeline.addEventListener("click", (e) => {
  const seg = e.target.closest(".seg[data-id]");
  if (!seg) return;
  activeId = Number(seg.dataset.id);
  video.pause();
  playbackStopAt = null;
  const s = currentShot();
  if (s) seekTo(s.clip_start);
  setEditorView("source", true);
  render();
});

document.querySelectorAll("[data-seek]").forEach((btn) => {
  btn.addEventListener("click", () => {
    video.currentTime = Math.max(0, video.currentTime + Number(btn.dataset.seek));
  });
});

document.getElementById("play-shot").addEventListener("click", async () => {
  const s = currentShot();
  if (!s) return;
  playbackStopAt = s.clip_end;
  try {
    await seekAndWait(s.clip_start);
    await video.play();
  } catch (_err) {
    playbackStopAt = null;
    showToast("Video is still loading; try again");
  }
});

video.addEventListener("timeupdate", () => {
  document.getElementById("clock").textContent = fmt(video.currentTime);
  if (playbackStopAt !== null && video.currentTime >= playbackStopAt) {
    video.pause();
    playbackStopAt = null;
  }
});

function markReviewChanged(message) {
  metadataEpoch += 1;
  meta.export_needs_update = true;
  meta.download_url = null;
  setExportStatus(`${message} Export video to update your highlights.`, "ready");
  syncJobState();
}

async function reviewRequest(path, options, message, {refreshShots = false, missingMessage = ''} = {}) {
  if (editBusy || exportBusy || analysisRequestBusy || timelineBusy || timelineRefreshRequired || UI.active(meta)) return null;
  editBusy = true;
  metadataEpoch += 1;
  syncJobState();
  let saved = false;
  try {
    const response = await fetch(path, options);
    if (!response.ok) {
      let detail = "Your change could not be saved. Try again.";
      try { detail = (await response.json()).detail || detail; } catch (error) { /* Keep a readable fallback. */ }
      if(response.status===404 && missingMessage) detail=missingMessage;
      throw new Error(detail);
    }
    const data = await response.json();
    saved = true;
    if(data.export_needs_update!==false) markReviewChanged(message);
    if (Array.isArray(data.shots)) shots=data.shots;
    else if (refreshShots) {
      timelineRefreshRequired = true;
      const stateResponse = await fetch(`/api/jobs/${jobId}`);
      if (stateResponse.ok) meta = await stateResponse.json();
      if (!await readSavedShots(meta)) throw new Error("Could not refresh saved shots");
    }
    else shots = shots.map((shot) => shot.shot_id === data.shot_id ? data : shot);
    showToast(message);
    render();
    return data;
  } catch (error) {
    showToast(saved ? "Change saved. Reconnecting to refresh your shot list…" : error.message || "The change could not be saved. Your saved shots are unchanged.");
    return null;
  } finally {
    editBusy = false;
    syncJobState();
    // A rejected edit may mean another tab started processing. Refresh once,
    // then follow an active worker without reloading either media element.
    refreshJobStatus();
  }
}

const jsonRequest = (body, method = "POST") => ({method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});

async function applyBounds() {
  const shot = currentShot();
  if (!shot) return;
  await reviewRequest(`/api/jobs/${jobId}/shots/${shot.shot_id}`, jsonRequest({
    clip_start: Number(document.getElementById("edit-start").value),
    cue_strike: Number(document.getElementById("edit-strike").value),
    clip_end: Number(document.getElementById("edit-end").value),
  }, "PATCH"), "Trim saved.");
}
document.getElementById("apply-bounds").addEventListener("click", applyBounds);

async function applyEndTrim(single=false,reset=false) {
  const shot=currentShot();
  if(single&&!shot)return;
  try {
    const body=window.snookerEndTrim.payload({normal:document.getElementById(single?'shot-end-trim':'end-trim-normal').value,
      long:document.getElementById('end-trim-long').value,threshold:document.getElementById('end-trim-threshold').value,
      ...(single?{shotId:shot.shot_id}:{}),reset});
    const data=await reviewRequest(`/api/jobs/${jobId}/trim-ends`,jsonRequest(body),reset?'Previous endings restored.':'End trim saved.',
      {missingMessage:'Restart the app server to enable saved end trimming.'});
    if(data) {
      const message=window.snookerEndTrim.message(data,reset);
      document.getElementById('end-trim-feedback').textContent=message;
      showToast(message);
      if(data.export_needs_update)setExportStatus(`${message} Export video to update your highlights.`,'ready');
    }
  }catch(error){showToast(error.message);}
}
document.getElementById('apply-shot-end-trim').addEventListener('click',()=>applyEndTrim(true));
document.getElementById('reset-shot-end-trim').addEventListener('click',()=>applyEndTrim(true,true));
document.getElementById('apply-all-end-trim').addEventListener('click',()=>applyEndTrim());
document.getElementById('reset-all-end-trim').addEventListener('click',()=>applyEndTrim(false,true));

document.getElementById("mark-replay").addEventListener("click", async () => {
  const shot = currentShot();
  if (!shot) return;
  await reviewRequest(`/api/jobs/${jobId}/shots/${shot.shot_id}`, jsonRequest({
    possible_replay: !shot.possible_replay, included: shot.possible_replay ? shot.included : false,
  }, "PATCH"), "Replay selection saved.");
});

async function setShotIncluded(shot, included) {
  const hadFocus = document.activeElement?.matches(`.shot-include[data-shot-id="${shot.shot_id}"]`);
  const updated = await reviewRequest(`/api/jobs/${jobId}/shots/${shot.shot_id}`, jsonRequest({included}, "PATCH"), "Selection saved.");
  if (hadFocus) (shotList.querySelector(`.shot-include[data-shot-id="${shot.shot_id}"]`) || shotFilter).focus({preventScroll: true});
  return updated;
}
document.getElementById("toggle-include").addEventListener("click", () => {
  const shot = currentShot();
  if (shot) setShotIncluded(shot, !shot.included);
});
shotList.addEventListener("change", async (event) => {
  const checkbox = event.target.closest(".shot-include");
  if (!checkbox) return;
  const shot = shots.find((item) => item.shot_id === Number(checkbox.dataset.shotId));
  const updated = await setShotIncluded(shot, checkbox.checked);
  if (!updated && checkbox.isConnected) checkbox.checked = shot.included;
});

document.getElementById("delete-shot").addEventListener("click", async () => {
  const shot = currentShot();
  if (!shot || !confirm(`Delete shot ${shot.shot_id}?`)) return;
  await reviewRequest(`/api/jobs/${jobId}/shots/${shot.shot_id}`, {method: "DELETE"}, "Shot deleted.", {refreshShots: true});
});

document.getElementById("split-shot").addEventListener("click", () => {
  const shot = currentShot();
  if (shot) reviewRequest(`/api/jobs/${jobId}/shots/${shot.shot_id}/split`, jsonRequest({at_time: video.currentTime}), "Shot split.", {refreshShots: true});
});

document.getElementById("merge-next").addEventListener("click", () => {
  const shot = currentShot();
  if (!shot) return;
  const index = shots.findIndex((item) => item.shot_id === shot.shot_id);
  const next = shots[index + 1];
  if (!next) { showToast("No next shot to merge"); return; }
  reviewRequest(`/api/jobs/${jobId}/shots/merge`, jsonRequest({shot_ids: [shot.shot_id, next.shot_id]}), "Shots merged.", {refreshShots: true});
});

document.getElementById("add-at-playhead").addEventListener("click", () => {
  setEditorView("source");
  const strike = pendingSeek ?? video.currentTime;
  reviewRequest(`/api/jobs/${jobId}/shots`, jsonRequest({cue_strike: strike, clip_start: Math.max(0, strike - 2), clip_end: strike + 4, shot_confidence: 1}), "Shot added.", {refreshShots: true});
});

openExportBtn.addEventListener("click", async () => {
  openExportBtn.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${jobId}/open-export-folder`, { method: "POST" });
    if (!res.ok) {
      const detail = await res.text();
      showToast("Could not open export folder: " + detail);
      setExportStatus("Could not open export folder", "error");
      return;
    }
    const data = await res.json();
    const count = Number.isFinite(data.clip_count) ? data.clip_count : 0;
    if (data.combined_exists) {
      const suffix = count === 1 ? "individual clip" : "individual clips";
      showToast("Opened export folder — combined video is ready");
      setExportStatus(
        `Combined video ready${count ? `; ${count} ${suffix} are in the clips subfolder` : ""}`,
        "ready",
      );
    } else {
      const suffix = count === 1 ? "clip" : "clips";
      showToast(`Opened export folder (${count} ${suffix} ready)`);
      setExportStatus(`${count} numbered ${suffix} ready in the clips subfolder`, "ready");
    }
  } catch (err) {
    showToast("Could not open export folder");
    setExportStatus("Could not open export folder", "error");
  } finally {
    openExportBtn.disabled = false;
  }
});

function downloadArtifact(url) {
  const link = document.createElement("a");
  link.href = url;
  link.download = "";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

async function runExport({ combined, clips, button }) {
  if (exportBusy || editBusy || analysisRequestBusy || timelineBusy || timelineRefreshRequired || UI.active(meta)) return;
  const originalLabel = button.textContent;
  exportBusy = true;
  metadataEpoch += 1;
  render();
  button.disabled = true;
  exportCombinedBtn.disabled = true;
  exportClipsBtn.disabled = true;
  combinedDownloadLink.classList.add("hidden");
  combinedDownloadLink.removeAttribute("href");
  button.textContent = "Exporting…";
  setExportStatus(
    combined
      ? "Rendering one continuous video from included shots..."
      : "Rendering numbered clips in chronological order...",
    "working",
  );
  showToast(combined ? "Creating combined video…" : "Exporting clips…");

  try {
    const res = await fetch(`/api/jobs/${jobId}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        output_name: "highlights.mp4",
        mode: "strict",
        accurate: true,
        export_clips: clips,
        export_joined: combined,
      }),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail);
    }

    const data = await res.json();
    if (data.status === "paused") {
      setExportStatus("Export paused. Use Resume in Current work to finish.", "ready");
      return;
    }
    const count = Number.isFinite(data.clip_count)
      ? data.clip_count
      : (data.clips || []).length;
    if (combined && data.download_url) {
      combinedDownloadLink.href = data.download_url;
      combinedDownloadLink.download = "highlights.mp4";
      combinedDownloadLink.classList.remove("hidden");
      downloadArtifact(data.download_url);
      showToast("Combined video ready — download started");
      setExportStatus("One combined video ready", "ready");
    } else {
      const suffix = count === 1 ? "clip" : "clips";
      showToast(`Export ready: ${count} ${suffix}`);
      setExportStatus(`${count} numbered ${suffix} ready — click Open export folder`, "ready");
    }
  } catch (err) {
    showToast("Export failed: " + err.message);
    setExportStatus("Export failed; check the analysis log for details", "error");
  } finally {
    exportBusy = false;
    button.textContent = originalLabel;
    try {
      const response = await fetch(`/api/jobs/${jobId}`);
      if (response.ok) meta = await response.json();
    } catch (error) { /* Keep the saved shot list while the connection recovers. */ }
    render();
    scheduleStatusCheck();
  }
}

exportCombinedBtn.addEventListener("click", () => runExport({
  combined: true,
  clips: false,
  button: exportCombinedBtn,
}));

exportClipsBtn.addEventListener("click", () => runExport({
  combined: false,
  clips: true,
  button: exportClipsBtn,
}));

document.getElementById("save-labels-btn").addEventListener("click", () => {
  window.open(`/api/jobs/${jobId}/download/training`, "_blank");
  // also corrections
  window.open(`/api/jobs/${jobId}/download/corrections`, "_blank");
  showToast("Labels download started");
});

async function runSavedAnalysis(action) {
  if (action === "restart" && !confirm(`Restart analysis for job ${jobId}? Saved checkpoints and your manual edits will be preserved.`)) return;
  if (analysisRequestBusy || editBusy || exportBusy || timelineBusy || timelineRefreshRequired || UI.active(meta)) return;
  analysisRequestBusy = true;
  metadataEpoch += 1;
  syncJobState();
  setExportStatus("Resuming processing from the last saved checkpoint…", "working");
  try {
    const response = await fetch(`/api/jobs/${jobId}/${action}`, {method: "POST"});
    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || "Could not start processing");
    }
    showToast(action === "continue" ? "Continuing saved work…" : "Analysis restarted");
  } catch (error) {
    setExportStatus(error.message || "Could not start processing", "error");
  } finally {
    analysisRequestBusy = false;
    await refreshJobStatus();
  }
}

document.getElementById("restart-analysis-btn").addEventListener("click", () => runSavedAnalysis("restart"));
document.getElementById("continue-analysis-btn").addEventListener("click", () => runSavedAnalysis("continue"));
document.getElementById("points-highlights-btn").addEventListener("click", async () => {
  await window.snookerPoints.open(jobId, {
    title: UI.title(meta),
    onUpdate: (state) => setExportStatus(state.message, state.state === "error" ? "error" : ""),
  });
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.matches("input, textarea, select, button, summary, a")) return;
  if (currentEditorView !== "source" && ["a", "r", "x", "t", "m", "delete", "i", "o", "s"].includes(e.key.toLowerCase())) {
    showToast("Select a shot or open Source & trim to edit it.");
    return;
  }
  const filtered = visibleShots();
  const idx = filtered.findIndex((s) => s.shot_id === activeId);
  if (e.key === "j" || e.key === "ArrowDown") {
    e.preventDefault();
    if (idx < filtered.length - 1) {
      activeId = filtered[idx + 1].shot_id;
      setEditorView("source", true);
      seekTo(currentShot().clip_start);
      render();
    }
  } else if (e.key === "k" || e.key === "ArrowUp") {
    e.preventDefault();
    if (idx > 0) {
      activeId = filtered[idx - 1].shot_id;
      setEditorView("source", true);
      seekTo(currentShot().clip_start);
      render();
    }
  } else if (e.key === "a" || e.key === "A") {
    applyBounds();
  } else if (e.key === "r" || e.key === "R") {
    document.getElementById("mark-replay").click();
  } else if (e.key === "x" || e.key === "X") {
    document.getElementById("toggle-include").click();
  } else if (e.key === "n" || e.key === "N") {
    document.getElementById("add-at-playhead").click();
  } else if (e.key === "t" || e.key === "T") {
    document.getElementById("split-shot").click();
  } else if (e.key === "m" || e.key === "M") {
    document.getElementById("merge-next").click();
  } else if (e.key === " " && !e.repeat) {
    e.preventDefault();
    if (currentEditorView === "source") {
      if (video.paused) video.play();
      else video.pause();
    } else {
      togglePreviewPlayback();
    }
  } else if (e.key === "Delete") {
    document.getElementById("delete-shot").click();
  } else if (e.key === "i") {
    document.getElementById("edit-start").value = video.currentTime.toFixed(3);
  } else if (e.key === "o") {
    document.getElementById("edit-end").value = video.currentTime.toFixed(3);
  } else if (e.key === "s") {
    document.getElementById("edit-strike").value = video.currentTime.toFixed(3);
  }
});

document.addEventListener("click", (event) => {
  document.querySelectorAll(".menu[open]").forEach((menu) => { if (!menu.contains(event.target)) menu.open = false; });
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") document.querySelectorAll(".menu[open]").forEach((menu) => { menu.open = false; menu.querySelector("summary").focus(); });
});
load().catch(() => {
  setExportStatus("Unable to load the editor. Check the app connection and reload.", "error");
  scheduleStatusCheck();
});
