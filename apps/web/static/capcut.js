/* Shared local desktop handoff for the jobs page and shot editor. */
window.snookerCapCut = (() => {
  const pending = new Set();
  let capability = null;
  async function load() {
    if (!capability) {
      capability = fetch("/api/capcut")
        .then(async (response) => {
          if (!response.ok) throw new Error("Could not check CapCut Desktop");
          return response.json();
        })
        .catch((error) => ({ available: false, reason: error.message }));
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
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/open-capcut`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not create CapCut project");
      status.textContent = data.message;
    } catch (error) {
      status.textContent = `Could not open in CapCut: ${error.message || error}`;
    } finally {
      pending.delete(jobId);
      button.textContent = oldLabel;
      button.disabled = !(await load()).available;
    }
  }
  return { load, open, isPending };
})();
