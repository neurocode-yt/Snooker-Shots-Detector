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
        (j) => `
      <div class="job-item">
        <div>
          <strong>${j.job_id}</strong>
          <div class="meta">${j.status || ""} · ${(j.progress * 100 || 0).toFixed(0)}% · ${j.message || ""}</div>
        </div>
        <a href="/review/${j.job_id}">Review</a>
      </div>`
      )
      .join("");
  } catch (e) {
    el.innerHTML = `<p class="meta">Could not load jobs: ${e}</p>`;
  }
}

document.getElementById("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  document.getElementById("file-name").textContent = f ? f.name : "Choose a snooker video…";
});

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("file");
  const file = fileInput.files[0];
  if (!file) return;
  const mode = document.getElementById("mode").value;
  const wrap = document.getElementById("progress-wrap");
  const fill = document.getElementById("progress-fill");
  const text = document.getElementById("progress-text");
  const btn = document.getElementById("start-btn");
  wrap.classList.remove("hidden");
  btn.disabled = true;
  text.textContent = "Uploading…";
  fill.style.width = "5%";

  try {
    const fd = new FormData();
    fd.append("file", file);
    const up = await fetch("/api/upload", { method: "POST", body: fd });
    if (!up.ok) throw new Error(await up.text());
    const { path } = await up.json();
    text.textContent = "Starting analysis…";
    fill.style.width = "10%";

    const start = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_path: path, mode, resume: true }),
    });
    if (!start.ok) throw new Error(await start.text());
    const { job_id } = await start.json();
    text.textContent = `Job ${job_id} running…`;

    const poll = setInterval(async () => {
      const pr = await fetch(`/api/jobs/${job_id}/progress`);
      const meta = await pr.json();
      const p = Math.max(10, (meta.progress || 0) * 100);
      fill.style.width = `${p}%`;
      text.textContent = `${meta.status}: ${meta.message || ""}`;
      if (["ready_for_review", "completed", "failed"].includes(meta.status)) {
        clearInterval(poll);
        btn.disabled = false;
        refreshJobs();
        if (meta.status === "failed") {
          text.textContent = `Failed: ${meta.error || meta.message}`;
        } else {
          text.textContent = "Done — opening review…";
          window.location.href = `/review/${job_id}`;
        }
      }
    }, 1500);
  } catch (err) {
    text.textContent = `Error: ${err.message || err}`;
    btn.disabled = false;
  }
});

refreshJobs();
setInterval(refreshJobs, 10000);
