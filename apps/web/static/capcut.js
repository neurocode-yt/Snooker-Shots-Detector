/* Shared local desktop handoff for the jobs page and shot editor. */
(function (root) {
function create(fetcher = (...args) => root.fetch(...args), now = () => Date.now()) {
  const pending = new Set();
  let capability = null;
  let expires = 0;
  let inFlight = null;
  let revision = 0;
  async function load({force = false} = {}) {
    if (root.SnookerAndroid) return {available: root.SnookerAndroid.capcutAvailable(),
      reason: 'Import included shots as separate clips into CapCut on this phone'};
    if (inFlight) return inFlight;
    if (force || !capability || now() >= expires) {
      const startedRevision = revision;
      inFlight = fetcher("/api/capcut")
        .then(async (response) => {
          if (!response.ok) throw new Error("Could not check CapCut Desktop");
          return response.json();
        })
        .catch((error) => ({ available: false, reason: error.message }))
        .then((data) => {
          if (startedRevision === revision) {
            capability = data;
            expires = now() + (data.available ? 30000 : 5000);
          }
          return capability || data;
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
    button.textContent = root.SnookerAndroid ? "Preparing CapCut clips\u2026" : "Preparing CapCut project\u2026";
    status.classList.remove("hidden");
    status.textContent = "Preparing editable shots\u2026";
    try {
      if (root.SnookerAndroid) {
        const requestId = `capcut_${now()}_${Math.random().toString(36).slice(2)}`;
        await new Promise((resolve, reject) => {
          const update = (event) => {
            if (event.detail?.requestId !== requestId) return;
            status.textContent = event.detail.message;
            if (event.detail.done) {
              root.removeEventListener('snooker-capcut', update);
              if (event.detail.failed) reject(new Error(event.detail.message)); else resolve();
            }
          };
          root.addEventListener('snooker-capcut', update);
          try { root.SnookerAndroid.openCapCut(jobId, requestId); }
          catch (error) { root.removeEventListener('snooker-capcut', update); reject(error); }
        });
        return;
      }
      // The action performs fresh discovery itself. A cached/in-flight status
      // request must never veto an actual project-opening attempt.
      const response = await fetcher(`/api/jobs/${encodeURIComponent(jobId)}/open-capcut`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not create CapCut project");
      revision++;
      capability = {available:true};
      expires = now()+30000;
      button.title = "Open included shots as editable clips in CapCut";
      status.textContent = root.snookerUI?.message(data.message) ?? data.message;
    } catch (error) {
      revision++;
      capability = null;
      expires = 0;
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
