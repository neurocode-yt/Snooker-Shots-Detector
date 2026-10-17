/* Shared local desktop handoff for the jobs page and shot editor. */
(function (root) {
function create(fetcher = (...args) => root.fetch(...args), now = () => Date.now()) {
  const pending = new Set();
  let capability = null;
  let expires = 0;
  let inFlight = null;
  async function load({force = false} = {}) {
    if (inFlight) return inFlight;
    if (force || !capability || now() >= expires) {
      inFlight = fetcher("/api/capcut")
        .then(async (response) => {
          if (!response.ok) throw new Error("Could not check CapCut Desktop");
          return response.json();
        })
        .catch((error) => ({ available: false, reason: error.message }))
        .then((data) => {
          capability = data;
          expires = now() + (data.available ? 30000 : 5000);
          return data;
        })
        .finally(() => { inFlight = null; });
      return inFlight;
    }
    return capability;
  }
  function isPending(jobId) {
    return pending.has(jobId);
  }
  async function open(jobId, button, status) {
    if (pending.has(jobId)) return;
    pending.add(jobId);
    const oldLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Preparing CapCut project\u2026";
    status.classList.remove("hidden");
    status.textContent = "Preparing editable shots\u2026";
    try {
      const readiness = await load({force: true});
      button.title = readiness.available
        ? "Create a new CapCut project with included shots as editable clips"
        : readiness.reason;
      if (!readiness.available) throw new Error(readiness.reason || "CapCut Desktop is unavailable on this PC.");
      const response = await fetcher(`/api/jobs/${encodeURIComponent(jobId)}/open-capcut`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not create CapCut project");
      status.textContent = root.snookerUI?.message(data.message) ?? data.message;
    } catch (error) {
      const detail = error.message || String(error);
      status.textContent = `Could not open in CapCut: ${root.snookerUI?.message(detail) ?? detail}`;
    } finally {
      pending.delete(jobId);
      button.textContent = oldLabel;
      // A failed discovery can recover without reloading the page. Keep the
      // action available for retry; the caller enforces job/selection guards.
      button.disabled = false;
    }
  }
  return { load, open, isPending };
}
root.snookerCapCut = create();
if (typeof module !== "undefined" && module.exports) module.exports = {create};
})(typeof window !== "undefined" ? window : globalThis);
