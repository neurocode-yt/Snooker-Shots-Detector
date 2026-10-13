/* Shared presentation rules. Detection, selection and export stay on the server. */
(function (root) {
  const activeStates = new Set(["pending", "validating", "proxy", "analyzing", "detecting", "refining", "segmenting", "exporting"]);
  const stoppedStates = new Set(["failed", "interrupted", "cancelled"]);
  const finishedStates = new Set(["completed", "ready_for_review"]);
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
  function filename(job) {
    return String(job.source_path || job.job_id || "Untitled video").split(/[\\/]/).pop()
      .replace(/^\d{8}-\d{6}_[a-f0-9]{8}_/i, "");
  }
  function title(job) {
    return filename(job).replace(/\.(mp4|mov|mkv|avi|webm|m4v)$/i, "").replace(/_/g, " ");
  }
  function mode(job) {
    if (job.selection === "pots_and_fouls") return "Pots + fouls";
    if (job.selection === "pots_only") return "Pots only";
    if (job.algorithm === "dl_algo") return "DL Algo";
    return job.auto_export === false ? "Review & edit" : "All shots";
  }
  const active = (job) => Boolean(job.running) || activeStates.has(job.status);
  const stopped = (job) => stoppedStates.has(job.status);
  const finished = (job) => finishedStates.has(job.status);
  const needsAttention = (job) => stopped(job) || job.status === "ready_for_review"
    || Boolean(job.export_needs_update)
    || Number(job.pot_selection_summary?.unknown || 0) > 0;
  function status(job) {
    if (active(job)) return {label: job.status === "exporting" ? "Exporting" : "Processing", tone: "working"};
    if (stopped(job)) return {label: job.status === "interrupted" ? "Interrupted" : job.status === "cancelled" ? "Stopped" : "Needs attention", tone: "warning"};
    if (job.export_needs_update) return {label: "Edits saved", tone: "warning"};
    if (job.status === "ready_for_review") return {label: "Ready to review", tone: "warning"};
    return {label: job.download_url ? "Ready" : "Finished", tone: "ready"};
  }
  function phase(job) {
    const labels = {pending: "Queued", validating: "Checking video", proxy: "Preparing video", analyzing: "Finding shots", detecting: "Finding shots", refining: "Refining shot boundaries", segmenting: "Preparing highlights", exporting: "Creating your video"};
    return job.status === "segmenting" && job.selection && job.selection !== "all"
      ? "Checking point changes" : labels[job.status] || job.message || "";
  }
  function actions(job, pointsAvailable = false) {
    return {
      resume: !active(job) && stopped(job) && job.can_continue === true,
      restart: !active(job),
      editor: Boolean(job.has_saved_detection) || finished(job),
      points: finished(job) && !active(job) && job.algorithm !== "dl_algo"
        && job.selection !== "pots_and_fouls" && Boolean(job.can_create_points) && pointsAvailable,
    };
  }
  function matches(job, filter, query = "") {
    const text = `${title(job)} ${filename(job)} ${job.job_id || ""} ${mode(job)}`.toLowerCase();
    if (!text.includes(query.trim().toLowerCase())) return false;
    return filter === "processing" ? active(job) : filter === "ready" ? finished(job)
      : filter === "attention" ? needsAttention(job) : true;
  }
  function time(seconds) {
    const value = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = Math.floor(value / 3600);
    return `${hours ? `${hours}:` : ""}${String(Math.floor(value / 60) % 60).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
  }
  function needsTimelineRefresh(next, loaded) {
    if (active(next) || next.has_saved_detection === false) return false;
    if (!next.has_saved_detection && !finished(next)) return false;
    if (!loaded || active(loaded)) return true;
    return Boolean(next.timeline_revision && next.timeline_revision !== loaded.timeline_revision);
  }
  const api = Object.freeze({escape, filename, title, mode, active, stopped, finished, needsAttention, status, phase, actions, matches, time, needsTimelineRefresh});
  root.snookerUI = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(globalThis);
