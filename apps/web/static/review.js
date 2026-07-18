const jobId = window.SNOOKER_JOB_ID;
const video = document.getElementById("video");
const shotList = document.getElementById("shot-list");
const timeline = document.getElementById("timeline");
const stats = document.getElementById("stats");
const toast = document.getElementById("toast");
const exportBtn = document.getElementById("export-btn");
const openClipsBtn = document.getElementById("open-clips-btn");
const exportStatus = document.getElementById("export-status");

let shots = [];
let activeId = null;
let duration = 0;
let meta = {};
let pendingSeek = null;
let playbackStopAt = null;

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 2200);
}

function setExportStatus(message, state = "") {
  exportStatus.textContent = message;
  exportStatus.dataset.state = state;
}

function fmt(t) {
  if (!Number.isFinite(t)) return "00:00.000";
  const m = Math.floor(t / 60);
  const s = t % 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
}

function confClass(s) {
  if (!s.included) return "excluded";
  if (s.possible_replay) return "replay";
  if (s.manual_review_required) return "needs-review";
  return "";
}

function render() {
  shotList.innerHTML = shots
    .map((s) => {
      const cls = [confClass(s), s.shot_id === activeId ? "active" : ""].filter(Boolean).join(" ");
      return `<li class="${cls}" data-id="${s.shot_id}">
        <div class="title">Shot ${s.shot_id} · conf ${s.shot_confidence.toFixed(2)}</div>
        <div class="detail">
          ${fmt(s.clip_start)} → ${fmt(s.clip_end)} · strike ${fmt(s.cue_strike)}
          ${s.possible_replay ? " · REPLAY" : ""}
          ${s.manual_review_required ? " · REVIEW" : ""}
          ${!s.included ? " · EXCLUDED" : ""}
        </div>
      </li>`;
    })
    .join("");

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
  stats.innerHTML = `
    ${shots.length} shots · ${included.length} included ·
    source ${fmt(duration)} · edited ~${fmt(edited)} ·
    removed ~${fmt(Math.max(0, duration - edited))}
  `;

  const active = shots.find((s) => s.shot_id === activeId);
  if (active) {
    document.getElementById("edit-start").value = active.clip_start.toFixed(3);
    document.getElementById("edit-strike").value = active.cue_strike.toFixed(3);
    document.getElementById("edit-end").value = active.clip_end.toFixed(3);
  }
}

async function load() {
  const pr = await fetch(`/api/jobs/${jobId}`);
  meta = await pr.json();
  document.getElementById("job-label").textContent = jobId;
  if (meta.mode) document.getElementById("mode-select").value = meta.mode;

  const res = await fetch(`/api/jobs/${jobId}/shots`);
  if (!res.ok) {
    showToast("Analysis not ready yet");
    return;
  }
  const data = await res.json();
  shots = data.shots || [];
  duration = data.original_duration || 0;
  video.src = `/api/jobs/${jobId}/video`;
  if (shots.length) {
    activeId = shots[0].shot_id;
    seekTo(shots[0].clip_start);
  }
  render();
}

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
  const li = e.target.closest("li[data-id]");
  if (!li) return;
  activeId = Number(li.dataset.id);
  const s = currentShot();
  if (s) seekTo(s.clip_start);
  render();
});

timeline.addEventListener("click", (e) => {
  const seg = e.target.closest(".seg[data-id]");
  if (!seg) return;
  activeId = Number(seg.dataset.id);
  const s = currentShot();
  if (s) seekTo(s.clip_start);
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

async function applyBounds() {
  const s = currentShot();
  if (!s) return;
  const body = {
    clip_start: Number(document.getElementById("edit-start").value),
    cue_strike: Number(document.getElementById("edit-strike").value),
    clip_end: Number(document.getElementById("edit-end").value),
  };
  const res = await fetch(`/api/jobs/${jobId}/shots/${s.shot_id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    showToast("Update failed");
    return;
  }
  const updated = await res.json();
  shots = shots.map((x) => (x.shot_id === updated.shot_id ? updated : x));
  showToast("Boundaries saved");
  render();
}

document.getElementById("apply-bounds").addEventListener("click", applyBounds);

document.getElementById("mark-replay").addEventListener("click", async () => {
  const s = currentShot();
  if (!s) return;
  const res = await fetch(`/api/jobs/${jobId}/shots/${s.shot_id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ possible_replay: !s.possible_replay, included: s.possible_replay ? s.included : false }),
  });
  if (!res.ok) {
    showToast("Update failed");
    return;
  }
  const updated = await res.json();
  shots = shots.map((x) => (x.shot_id === updated.shot_id ? updated : x));
  render();
});

document.getElementById("toggle-include").addEventListener("click", async () => {
  const s = currentShot();
  if (!s) return;
  const res = await fetch(`/api/jobs/${jobId}/shots/${s.shot_id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ included: !s.included }),
  });
  if (!res.ok) {
    showToast("Update failed");
    return;
  }
  const updated = await res.json();
  shots = shots.map((x) => (x.shot_id === updated.shot_id ? updated : x));
  render();
});

document.getElementById("delete-shot").addEventListener("click", async () => {
  const s = currentShot();
  if (!s) return;
  if (!confirm(`Delete shot ${s.shot_id}?`)) return;
  await fetch(`/api/jobs/${jobId}/shots/${s.shot_id}`, { method: "DELETE" });
  shots = shots.filter((x) => x.shot_id !== s.shot_id);
  activeId = shots[0]?.shot_id ?? null;
  // reload to renumber
  await load();
  showToast("Shot deleted");
});

document.getElementById("split-shot").addEventListener("click", async () => {
  const s = currentShot();
  if (!s) return;
  const at = video.currentTime;
  const res = await fetch(`/api/jobs/${jobId}/shots/${s.shot_id}/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ at_time: at }),
  });
  if (!res.ok) {
    showToast("Split failed: " + (await res.text()));
    return;
  }
  await load();
  showToast("Shot split");
});

document.getElementById("merge-next").addEventListener("click", async () => {
  const s = currentShot();
  if (!s) return;
  const idx = shots.findIndex((x) => x.shot_id === s.shot_id);
  if (idx < 0 || idx >= shots.length - 1) {
    showToast("No next shot to merge");
    return;
  }
  const next = shots[idx + 1];
  const res = await fetch(`/api/jobs/${jobId}/shots/merge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ shot_ids: [s.shot_id, next.shot_id] }),
  });
  if (!res.ok) {
    showToast("Merge failed: " + (await res.text()));
    return;
  }
  await load();
  showToast("Shots merged");
});

document.getElementById("add-at-playhead").addEventListener("click", async () => {
  const t = video.currentTime;
  const body = {
    cue_strike: t,
    clip_start: Math.max(0, t - 2),
    clip_end: t + 4,
    shot_confidence: 1.0,
  };
  const res = await fetch(`/api/jobs/${jobId}/shots`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    showToast("Add failed");
    return;
  }
  await load();
  showToast("Shot added");
});

openClipsBtn.addEventListener("click", async () => {
  openClipsBtn.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${jobId}/open-clips-folder`, { method: "POST" });
    if (!res.ok) {
      const detail = await res.text();
      showToast("Could not open clips folder: " + detail);
      setExportStatus("Could not open clips folder", "error");
      return;
    }
    const data = await res.json();
    const count = Number.isFinite(data.clip_count) ? data.clip_count : 0;
    const suffix = count === 1 ? "clip" : "clips";
    showToast(`Opened clips folder in Windows Explorer (${count} ${suffix})`);
    setExportStatus(`${count} numbered ${suffix} ready for CapCut`, "ready");
  } catch (err) {
    showToast("Could not open clips folder");
    setExportStatus("Could not open clips folder", "error");
  } finally {
    openClipsBtn.disabled = false;
  }
});

exportBtn.addEventListener("click", async () => {
  exportBtn.disabled = true;
  exportBtn.textContent = "Exporting clips...";
  setExportStatus("Rendering numbered clips in chronological order...", "working");
  showToast("Exporting…");
  const res = await fetch(`/api/jobs/${jobId}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      output_name: "highlights.mp4",
      mode: document.getElementById("mode-select").value,
      accurate: true,
      export_clips: true,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    showToast("Export failed: " + detail);
    setExportStatus("Export failed; clips folder may still contain completed clips", "error");
    exportBtn.disabled = false;
    exportBtn.textContent = "Export clips";
    return;
  }
  const data = await res.json();
  const count = Number.isFinite(data.clip_count) ? data.clip_count : (data.clips || []).length;
  showToast(`Export ready: ${count} clips`);
  setExportStatus(`${count} numbered clips ready â€” click Open clips folder`, "ready");
  if (data.joined) window.open(`/api/jobs/${jobId}/download/highlights`, "_blank");
  console.log(data);
  exportBtn.disabled = false;
  exportBtn.textContent = "Export clips";
});

document.getElementById("save-labels-btn").addEventListener("click", () => {
  window.open(`/api/jobs/${jobId}/download/training`, "_blank");
  // also corrections
  window.open(`/api/jobs/${jobId}/download/corrections`, "_blank");
  showToast("Labels download started");
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, select")) return;
  const idx = shots.findIndex((s) => s.shot_id === activeId);
  if (e.key === "j" || e.key === "ArrowDown") {
    if (idx < shots.length - 1) {
      activeId = shots[idx + 1].shot_id;
      seekTo(currentShot().clip_start);
      render();
    }
  } else if (e.key === "k" || e.key === "ArrowUp") {
    if (idx > 0) {
      activeId = shots[idx - 1].shot_id;
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
    if (video.paused) video.play();
    else video.pause();
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

load();
