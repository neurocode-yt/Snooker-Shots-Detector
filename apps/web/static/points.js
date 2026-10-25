/* Pots + fouls creation, progress and completed-video handoff. */
(function (root) {
const message = value => (root.snookerUI || (typeof module !== "undefined" ? require('./ui.js') : null)).message(value);
function create(fetcher = (...args) => root.fetch(...args), wait = (ms) => new Promise((done) => setTimeout(done, ms))) {
  const requests = new Map();
  async function read(url, options) {
    const response = await fetcher(url, options);
    let data;
    try { data = await response.json(); }
    catch { throw new Error("The server returned an unreadable response. Please retry."); }
    if (!response.ok) throw new Error(data.detail || "Could not create pots + fouls highlights.");
    return data;
  }
  function start(jobId) {
    if (!requests.has(jobId)) {
      requests.set(jobId, read(`/api/jobs/${encodeURIComponent(jobId)}/points-highlights`, {method: "POST"})
        .finally(() => requests.delete(jobId)));
    }
    return requests.get(jobId);
  }
  async function run(jobId, update, isActive = () => true) {
    update({state: "starting", message: "Checking saved highlights and starting any work needed…"});
    try {
      const result = await start(jobId);
      if (!result.job_id) throw new Error("The server did not return a highlights job. Please retry.");
      const variantId = result.job_id;
      if (result.status === "ready") {
        update({state: "ready", jobId: variantId, alreadyMade: true,
          message: "Pots + fouls highlights already made."});
        return;
      }
      if (!["started", "running"].includes(result.status)) throw new Error("Could not start pots + fouls highlights. Please retry.");
      update({state: "processing", jobId: variantId, message: result.status === "running"
        ? "Pots + fouls highlights are already being created. Following their progress…"
        : "Creating pots + fouls highlights from saved shots…"});
      while (isActive()) {
        const job = await read(`/api/jobs/${encodeURIComponent(variantId)}/progress`);
        if (!isActive()) return;
        if (["failed", "interrupted", "cancelled"].includes(job.status) && !job.running) {
          throw new Error(job.error || job.message || "Processing stopped. Click Pots + fouls again to resume saved work.");
        }
        if (job.status === "completed" && job.download_url && !job.running) {
          update({state: "ready", jobId: variantId, alreadyMade: false,
            message: "Pots + fouls highlights are ready."});
          return;
        }
        if (["completed", "ready_for_review"].includes(job.status) && !job.running) {
          update({state: "finished", jobId: variantId,
            message: message(job.message) || "No confirmed scoring shots were selected automatically."});
          return;
        }
        const progress = Math.max(0, Math.min(100, Math.round(Number(job.progress || 0) * 100)));
        update({state: "processing", jobId: variantId, progress,
          message: job.queued ? "Pots + fouls highlights are queued. Processing starts when a slot is free."
            : message(job.message) || "Creating pots + fouls highlights…"});
        await wait(2500);
      }
    } catch (error) {
      update({state: "error", message: `Could not create pots + fouls highlights: ${error.message || error}`});
    }
  }
  return {start, run};
}

function createDialog(api = create()) {
  let dialog;
  let session = 0;
  let currentSource;
  let currentUpdate;
  let handoffPending = false;
  function init() {
    if (dialog) return;
    dialog = root.document.createElement("dialog");
    dialog.className = "points-dialog";
    dialog.setAttribute("aria-labelledby", "points-dialog-title");
    dialog.setAttribute("aria-describedby", "points-dialog-message");
    dialog.innerHTML = `<p class="eyebrow">POTS + FOULS</p>
      <h2 id="points-dialog-title">Create highlights</h2>
      <p class="muted points-dialog-video"></p>
      <p id="points-dialog-message" role="status" aria-live="polite"></p>
      <progress class="points-dialog-progress" max="100" aria-label="Pots + fouls processing"></progress>
      <p class="points-dialog-progress-label muted"></p>
      <p class="points-dialog-question hidden">Open in CapCut?</p>
      <p class="capcut-status hidden" role="status" aria-live="polite"></p>
      <div class="points-dialog-links hidden"><a class="points-download">Download video</a><a class="points-editor">Open editor</a></div>
      <div class="points-dialog-actions"><button class="points-close">Close</button><button class="points-open primary hidden">Open in CapCut</button></div>`;
    root.document.body.appendChild(dialog);
    dialog.querySelector(".points-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => { session++; });
  }
  function open(jobId, {title = "", onUpdate = () => {}} = {}) {
    init();
    if (handoffPending) { if (!dialog.open) dialog.showModal(); return; }
    if (dialog.open && currentSource === jobId) return;
    const token = ++session;
    currentSource = jobId;
    currentUpdate = null;
    const close = dialog.querySelector(".points-close");
    const capcut = dialog.querySelector(".points-open");
    const capcutStatus = dialog.querySelector(".capcut-status");
    capcutStatus.textContent = "";
    capcutStatus.classList.add("hidden");
    dialog.querySelector(".points-dialog-video").textContent = title;
    if (!dialog.open) dialog.showModal();
    const update = (state) => {
      if (session !== token) return;
      currentUpdate = state;
      onUpdate(state);
      const ready = state.state === "ready";
      const processing = ["starting", "processing"].includes(state.state);
      dialog.querySelector("h2").textContent = ready ? state.message
        : state.state === "error" ? "Highlights could not finish"
        : state.state === "finished" ? "No scoring video created" : "Creating pots + fouls highlights";
      dialog.querySelector("#points-dialog-message").textContent = ready
        ? "Your saved highlights are available below." : state.message;
      const progress = dialog.querySelector("progress");
      progress.classList.toggle("hidden", !processing);
      if (Number.isFinite(state.progress)) progress.value = state.progress; else progress.removeAttribute("value");
      dialog.querySelector(".points-dialog-progress-label").textContent = processing && Number.isFinite(state.progress)
        ? `${state.progress}% complete` : "";
      dialog.querySelector(".points-dialog-question").classList.toggle("hidden", !ready);
      dialog.querySelector(".points-dialog-links").classList.toggle("hidden", !ready && state.state !== "finished");
      dialog.querySelector(".points-download").classList.toggle("hidden", !ready);
      capcut.classList.toggle("hidden", !ready);
      close.textContent = ready ? "Not now" : processing ? "Keep working in background" : "Close";
      if (state.jobId) {
        dialog.querySelector(".points-download").href = `/api/jobs/${encodeURIComponent(state.jobId)}/download/highlights`;
        dialog.querySelector(".points-editor").href = `/review/${encodeURIComponent(state.jobId)}?editor=classic`;
      }
      if (ready) {
        capcut.focus();
      }
    };
    capcut.onclick = async () => {
      if (handoffPending || currentUpdate?.state !== "ready") return;
      handoffPending = true;
      try { await root.snookerCapCut.open(currentUpdate.jobId, capcut, capcutStatus); }
      finally { handoffPending = false; }
    };
    return api.run(jobId, update, () => session === token && dialog.open);
  }
  return {open};
}
if (root.document) root.snookerPoints = createDialog();
if (typeof module !== "undefined" && module.exports) module.exports = {create, createDialog};
})(typeof window !== "undefined" ? window : globalThis);
