/* Shared Make Shorts dialog for the library and editor. */
(function (root) {
  const numberFields = ['video_zoom', 'video_y', 'title_y', 'name_size', 'shot_size', 'line_spacing'];
  const colorFields = ['background', 'name_color', 'shot_color'];
  const busy = (batch) => ['queued', 'rendering'].includes(batch?.status);
  const resumable = (batch) => ['failed', 'interrupted', 'paused'].includes(batch?.status);
  function settingsFrom(values, overrides, shots) {
    const settings = {player_names: values.player_names.trim(), preset: values.preset, shot_names: {}};
    for (const field of numberFields) settings[field] = Number(values[field]);
    for (const field of colorFields) settings[field] = values[field];
    for (const shot of shots) {
      const name = (overrides[shot.shot_id] || '').trim();
      if (name) settings.shot_names[shot.shot_id] = name;
    }
    return settings;
  }
  function batchLabel(batch) {
    if (!batch) return 'Create Shorts';
    if (busy(batch)) return 'Rendering Shorts…';
    if (resumable(batch)) return 'Resume Shorts';
    return 'Create Shorts';
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {settingsFrom, batchLabel, busy};
  if (!root.document) return;
  const escape = (value) => root.snookerUI.escape(value);
  let dialog, jobId, setup, batch, selectedBatchId, pollTimer, previewTimer, previewController, previewUrl, session = 0, opener, submitting = false;
  const $ = (selector) => dialog.querySelector(selector);
  const field = (name) => $(`[name="${name}"]`);
  const base = () => `/api/jobs/${encodeURIComponent(jobId)}/shorts`;
  async function json(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not prepare Shorts. Check the settings and retry.');
    return data;
  }
  function mount() {
    if (dialog) return;
    dialog = document.createElement('dialog');
    dialog.className = 'shorts-dialog';
    dialog.setAttribute('aria-labelledby', 'shorts-heading');
    dialog.innerHTML = `<header class="shorts-header"><div><p class="eyebrow">ONE SHOT. ONE SHORT.</p><h2 id="shorts-heading">Make Shorts</h2><p class="muted" id="shorts-summary">Loading saved shots…</p></div><button type="button" class="shorts-close" aria-label="Close Make Shorts">✕</button></header>
      <div class="shorts-error notice hidden" role="alert"></div>
      <div class="shorts-grid"><form id="shorts-form"><fieldset id="shorts-settings" disabled>
      <label>Player or match names<input name="player_names" type="text" placeholder="Judd Trump vs Ronnie O’Sullivan" maxlength="160" required autocomplete="off" /></label>
      <p class="muted shorts-help">Used in the video title and filename. Original match audio is kept.</p>
      <label>Layout preset<select name="preset"></select></label>
      <details class="shorts-advanced"><summary>Canvas &amp; video</summary><div class="shorts-color-row"><label>Background<input name="background" type="color" /></label><label>Player text<input name="name_color" type="color" /></label><label>Shot text<input name="shot_color" type="color" /></label></div>
      <label>Video zoom <output data-short-value="video_zoom"></output><input name="video_zoom" type="range" min="0.5" max="2" step="0.05" /></label>
      <p class="muted shorts-help">1× keeps the full frame. Zooming in crops the edges.</p>
      <label>Video position <output data-short-value="video_y"></output><input name="video_y" type="range" min="0" max="1" step="0.01" /></label></details>
      <details class="shorts-advanced"><summary>Text size &amp; position</summary><div class="shorts-fields"><label>Player font size<input name="name_size" type="number" min="32" max="180" required /></label><label>Shot font size<input name="shot_size" type="number" min="24" max="140" required /></label><label>Text top (px)<input name="title_y" type="number" min="0" max="1650" required /></label><label>Line spacing<input name="line_spacing" type="number" min="0" max="80" required /></label></div></details>
      <details class="shorts-advanced"><summary>Player names per shot <span class="muted">Optional</span></summary><p class="muted shorts-help">Override the match name for shots from a particular player.</p><div id="shorts-name-overrides"></div></details>
      </fieldset><button type="submit" id="shorts-start" class="primary">Create Shorts</button><section class="shorts-results hidden" aria-label="Shorts export progress"><p id="shorts-batch-status" role="status" aria-live="polite"></p><progress id="shorts-progress" max="1" value="0" aria-label="Shorts rendered"></progress><p id="shorts-batch-error" class="shorts-help"></p><div class="shorts-download-actions"><button type="button" id="shorts-pause" class="hidden">Pause</button><a id="shorts-zip" class="button primary hidden" download>Download all ZIP</a><button type="button" id="shorts-folder">Open Shorts folder</button></div><ul id="shorts-files" aria-label="Individual Shorts"></ul></section></form>
      <section class="shorts-preview" aria-label="Shorts layout preview"><label>Preview shot<select name="preview_shot" aria-label="Preview shot"></select></label><div class="shorts-preview-frame"><img id="shorts-preview-image" alt="Vertical Short with player name and shot number" hidden /><p id="shorts-preview-status" class="muted" role="status">Preparing preview…</p></div><p class="muted shorts-help">1080 × 1920 · one MP4 per included live shot</p></section></div>`;
    document.body.appendChild(dialog);
    $('.shorts-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => {
      session++;
      clearTimeout(pollTimer); clearTimeout(previewTimer); previewController?.abort();
      if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
      $('#shorts-preview-image').hidden = true;
      opener?.focus();
    });
    field('preset').addEventListener('change', () => {
      const preset = setup.presets[field('preset').value];
      for (const name of [...numberFields, ...colorFields]) field(name).value = preset[name];
      updateLabels(); schedulePreview();
    });
    $('#shorts-form').addEventListener('input', () => { updateLabels(); schedulePreview(); });
    field('preview_shot').addEventListener('change', schedulePreview);
    $('#shorts-form').addEventListener('submit', start);
    $('#shorts-pause').addEventListener('click', async () => {
      if (!batch || batch.pause_requested) return;
      $('#shorts-pause').disabled = true;
      try {
        batch = await json(`${base()}/${batch.batch_id}/pause`, {method:'POST'});
        renderBatch(); poll(); root.snookerWork?.refresh();
      } catch(error) { showError(error.message); $('#shorts-pause').disabled=false; }
    });
    $('#shorts-folder').addEventListener('click', async () => {
      if (!batch) return;
      try { await json(`${base()}/${batch.batch_id}/open-folder`, {method:'POST'}); }
      catch (error) { showError(error.message); }
    });
  }
  function showError(message = '') {
    $('.shorts-error').textContent = message;
    $('.shorts-error').classList.toggle('hidden', !message);
  }
  function values() {
    const values = {}, overrides = {};
    for (const name of ['player_names', 'preset', ...numberFields, ...colorFields]) values[name] = field(name).value;
    dialog.querySelectorAll('[data-short-shot]').forEach((input) => { overrides[input.dataset.shortShot] = input.value; });
    return settingsFrom(values, overrides, setup.shots);
  }
  function updateLabels() {
    $('[data-short-value="video_zoom"]').textContent = `${Number(field('video_zoom').value).toFixed(2)}×`;
    $('[data-short-value="video_y"]').textContent = `${Math.round(Number(field('video_y').value) * 100)}%`;
  }
  function schedulePreview() {
    clearTimeout(previewTimer);
    previewController?.abort();
    previewTimer = setTimeout(renderPreview, 350);
  }
  async function renderPreview() {
    if (!setup || !dialog.open) return;
    const token = session;
    previewController = new AbortController();
    const settings = values();
    settings.player_names ||= 'Player name';
    $('#shorts-preview-status').textContent = 'Updating preview…';
    try {
      const response = await fetch(`${base()}/preview`, {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({settings, shot_id:Number(field('preview_shot').value), batch_id:selectedBatchId}), signal:previewController.signal});
      if (!response.ok) { const error = await response.json(); throw new Error(typeof error.detail === 'string' ? error.detail : 'Check the layout settings'); }
      const blob = await response.blob();
      if (token !== session) return;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(blob);
      $('#shorts-preview-image').src = previewUrl;
      $('#shorts-preview-image').hidden = false;
      $('#shorts-preview-status').textContent = '';
    } catch (error) {
      if (error.name !== 'AbortError' && token === session) $('#shorts-preview-status').textContent = error.message;
    }
  }
  function renderBatch() {
    $('.shorts-results').classList.toggle('hidden', !batch);
    $('#shorts-settings').disabled = submitting || busy(batch) || resumable(batch);
    $('#shorts-start').disabled = submitting || busy(batch);
    $('#shorts-start').textContent = submitting ? 'Starting…' : batchLabel(batch);
    if (!batch) return;
    $('#shorts-pause').classList.toggle('hidden', !busy(batch));
    $('#shorts-pause').disabled = Boolean(batch.pause_requested);
    $('#shorts-pause').textContent = batch.pause_requested ? 'Pausing…' : 'Pause';
    $('#shorts-batch-status').textContent = batch.message;
    $('#shorts-batch-error').textContent = batch.error || '';
    $('#shorts-progress').max = batch.total;
    $('#shorts-progress').value = batch.completed;
    $('#shorts-zip').classList.toggle('hidden', batch.status !== 'completed');
    $('#shorts-zip').href = `${base()}/${batch.batch_id}/zip`;
    $('#shorts-files').innerHTML = batch.clips.map((clip) => `<li><a download href="${base()}/${batch.batch_id}/clips/${clip.shot_id}">${escape(clip.title)} <span aria-hidden="true">↓</span></a><span class="muted">${clip.duration.toFixed(1)}s</span></li>`).join('');
  }
  function poll() {
    clearTimeout(pollTimer);
    if (!busy(batch) || !dialog.open) return;
    const token = session;
    pollTimer = setTimeout(async () => {
      try {
        const next = await json(`${base()}/${batch.batch_id}`);
        if (token !== session) return;
        batch = next; renderBatch(); poll();
      } catch (error) { if (token === session) { showError(`Progress check failed: ${error.message}`); poll(); } }
    }, 1500);
  }
  async function start(event) {
    event.preventDefault();
    if (submitting || busy(batch) || !$('#shorts-form').reportValidity()) return;
    const token = session;
    submitting = true; renderBatch(); showError();
    try {
      const next = resumable(batch)
        ? await json(`${base()}/${batch.batch_id}/resume`, {method:'POST'})
        : await json(base(), {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(values())});
      if (token !== session) return;
      batch = next; renderBatch(); poll(); root.snookerWork?.refresh();
    } catch (error) { if (token === session) showError(error.message); }
    finally { if (token === session) { submitting = false; renderBatch(); } }
  }
  async function open(id, button, batchId = null) {
    mount();
    if (dialog.open) return;
    const token = ++session;
    jobId = id; opener = button; setup = null; batch = null; submitting = false; selectedBatchId = batchId;
    $('#shorts-settings').disabled = true;
    $('#shorts-summary').textContent = 'Loading saved shots…';
    $('.shorts-results').classList.add('hidden');
    $('#shorts-preview-status').textContent = 'Preparing preview…';
    showError(); dialog.showModal();
    try {
      let data = await json(base() + (batchId ? `?batch_id=${encodeURIComponent(batchId)}` : ''));
      if (!batchId && resumable(data.batch)) {
        selectedBatchId = data.batch.batch_id;
        data = await json(`${base()}?batch_id=${encodeURIComponent(selectedBatchId)}`);
      }
      if (token !== session) return;
      setup = data; batch = setup.batch;
      const settings = batch?.settings || {...setup.defaults, player_names:''};
      field('preset').innerHTML = Object.entries(setup.presets).map(([key, preset]) => `<option value="${escape(key)}">${escape(preset.label)}</option>`).join('');
      for (const name of ['player_names','preset', ...numberFields, ...colorFields]) field(name).value = settings[name];
      field('preview_shot').innerHTML = setup.shots.map((shot) => `<option value="${shot.shot_id}">Shot ${String(shot.shot_id).padStart(2,'0')} · ${(shot.end-shot.start).toFixed(1)}s</option>`).join('');
      $('#shorts-name-overrides').innerHTML = setup.shots.map((shot) => `<label>Shot ${String(shot.shot_id).padStart(2,'0')} player name<input type="text" data-short-shot="${shot.shot_id}" maxlength="160" placeholder="Use match names" value="${escape(settings.shot_names?.[shot.shot_id] || '')}" /></label>`).join('');
      const review = setup.shots.filter((shot) => shot.needs_review).length;
      $('#shorts-summary').textContent = `${setup.shots.length} included live shots · individual vertical videos${review ? ` · ${review} flagged for review in the editor` : ''}${setup.selection_changed ? ' · Selection changed since the previous batch; create again to use your edits.' : ''}`;
      updateLabels(); renderBatch(); renderPreview(); poll();
    } catch (error) { if (token === session) showError(error.message); }
  }
  root.snookerShorts = {open};
})(typeof window !== 'undefined' ? window : globalThis);
