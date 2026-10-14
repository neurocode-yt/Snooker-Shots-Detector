/* Submit each video independently. Retrying a failed submission reuses its upload. */
(function (root) {
  async function submit(entries, options, services) {
    for (const entry of entries) {
      if (entry.jobId) continue;
      try {
        entry.options ||= {...options};
        entry.error = "";
        if (!entry.sourcePath) {
          entry.state = "Uploading";
          services.changed(entry);
          entry.sourcePath = await services.upload(entry.file);
        }
        if (entry.ranges && !entry.cleaned) {
          entry.state = "Trimming";
          services.changed(entry);
          entry.sourcePath = await services.preprocess(entry.sourcePath, entry.ranges);
          entry.cleaned = true;
        }
        entry.state = "Adding to queue";
        services.changed(entry);
        entry.jobId = await services.start(entry.sourcePath, entry.options, entry);
        entry.state = "Submitted";
      } catch (error) {
        entry.error = error.message || String(error);
        entry.state = "Could not submit";
      }
      services.changed(entry);
    }
    return {submitted: entries.filter((entry) => entry.jobId).length,
      failed: entries.filter((entry) => !entry.jobId).length};
  }
  root.snookerBatch = Object.freeze({submit});
  if (typeof module !== "undefined" && module.exports) module.exports = root.snookerBatch;
})(globalThis);
