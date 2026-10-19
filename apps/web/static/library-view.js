/* One source card, with an independent action row for each saved version. */
(function (root) {
  const UI = root.snookerUI || (typeof module !== 'undefined' ? require('./ui.js') : null);
  const Library = root.snookerLibrary || (typeof module !== 'undefined' ? require('./library.js') : null);
  const escape = UI.escape;
  const actionKey = (key, selection) => `${key}|${selection}`;
  function date(job) {
    return job.created_at ? new Intl.DateTimeFormat('en-GB',{day:'numeric',month:'short',year:'numeric'}).format(new Date(job.created_at*1000)) : '';
  }
  function run(job, label, settings, historical=false) {
    const id = escape(job.job_id), state = Library.state(job), active = UI.active(job);
    const pending = settings.pending?.has(job.job_id);
    const actions = UI.actions(job), summary = job.pot_selection_summary || {};
    const oldFramePolicy = Number(summary.policy)>0 && Number(summary.policy)<6;
    const gaps = (Array.isArray(job.scoring_gaps_unresolved)?job.scoring_gaps_unresolved:[])
      .filter(gap=>Number.isFinite(gap.start)&&Number.isFinite(gap.end)&&gap.start>=0&&gap.end>gap.start);
    const count = Number.isFinite(summary.included) ? `${summary.included}${Number.isFinite(job.shots_detected)?` of ${job.shots_detected}`:''} shots selected`
      : active && !job.has_saved_detection && !job.shots_detected ? (job.reuse_saved_detection?'Reusing saved shots':'Detection in progress')
      : job.shots_detected != null ? `${job.shots_detected} shots detected` : 'Saved shots';
    const hasShots = Number(summary.included ?? job.shots_detected ?? 1)>0;
    const available = UI.finished(job) && !active && !pending && hasShots;
    const editor = `/review/${encodeURIComponent(job.job_id)}?editor=classic`;
    const editorLink = actions.editor ? `<a class="button secondary" href="${editor}" data-job-id="${id}">${active?'View shots':'Open editor'}</a>` : '';
    const controls = `${job.download_url && !active ? `<a class="button primary" href="${escape(job.download_url)}" data-job-id="${id}">Download MP4</a>` : ''}
      ${!job.download_url && UI.finished(job) && hasShots && !active ? `<button class="btn-export-version primary" data-job-id="${id}" ${pending?'disabled':''}>${pending?'Exporting…':'Export video'}</button>` : ''}
      ${actions.resume ? `<button class="btn-continue primary" data-job-id="${id}">${job.status==='paused'?'Resume':'Continue'}</button>` : ''}
      ${editorLink}
      ${UI.finished(job) && !active && job.selection && job.selection!=='all' && (oldFramePolicy || Number(summary.unknown||0)>0) ? `<button class="btn-recheck-scores secondary" data-job-id="${id}" ${pending?'disabled':''}>${oldFramePolicy?'Update frame shots':'Recheck scores'}</button>` : ''}
      ${UI.finished(job) ? `<button class="btn-capcut secondary" data-job-id="${id}" ${available?'':'disabled'} title="${escape(settings.capcut?.reason || 'Open this version as editable clips')}">Open in CapCut</button><button class="btn-shorts secondary" data-job-id="${id}" ${available?'':'disabled'}>Make Shorts</button>` : ''}
      <details class="menu job-menu"><summary class="button icon-button" data-job-id="${id}" aria-label="More options for ${escape(label)}${historical?' earlier run':''}">•••</summary><div class="menu-panel"><button class="btn-restart" data-job-id="${id}" ${actions.restart&&!pending?'':'disabled'}>Restart analysis</button><div class="menu-meta"><strong>Processing details</strong><span>${escape(UI.message(job.message))}</span>${job.error?`<span class="error-detail">${escape(UI.message(job.error))}</span>`:''}${Number.isFinite(job.checkpoint?.source_time)?`<span>Saved through ${UI.time(job.checkpoint.source_time)}${job.checkpoint.duration?` of ${UI.time(job.checkpoint.duration)}`:''}</span>`:''}<span class="job-id">${id}</span></div></div></details>`;
    return `<article class="job-item version-run" data-library-job="${id}" aria-label="${escape(label)}${historical?' earlier run':''}"><div class="job-copy"><div class="job-title-row"><h4>${escape(label)}${historical?' · earlier run':''}</h4><span class="status-badge ${state.tone}">${state.label}</span></div><p class="version-detail">${escape(count)}${historical?` · ${date(job)}`:''}${!active&&UI.finished(job)&&!job.download_url?' · Video export not ready':''}</p>
      ${active ? `<div class="job-progress"><span>${escape(UI.phase(job))}</span><span>${Math.round((job.progress||0)*100)}%</span><div role="progressbar" aria-label="${escape(label)} processing" aria-valuenow="${Math.round((job.progress||0)*100)}" aria-valuemin="0" aria-valuemax="100"><i style="width:${Math.round(Math.max(0,Math.min(1,job.progress||0))*100)}%"></i></div></div>`:''}
      ${UI.stopped(job)?`<p class="job-warning">${job.status==='paused'?'Paused':'Processing stopped'}. Saved work is preserved.</p>`:''}
      ${job.export_needs_update?'<p class="job-warning">Edits saved. Export video to update this version.</p>':''}
      ${Number(summary.frame_context_shots||0)>0?`<p class="version-detail">Includes opening and final shots from ${Number(summary.frames?.length||0)} frames${Number(summary.non_scoring_context_shots||0)>0?` · ${Number(summary.non_scoring_context_shots)} additional context shots`:''}.</p>`:''}
      ${Number(summary.unknown||0)>0?`<p class="version-detail">${summary.score_readings===0?'Scoreboard could not be read. Saved shots are preserved.':`${Number(summary.unknown)} shots have unconfirmed scores.`}</p>`:''}
      ${gaps.length?`<details class="version-history" data-history="gaps:${id}" ${settings.expanded?.has('gaps:'+job.job_id)?'open':''}><summary>Scoring details (${gaps.length})</summary><p class="version-detail">Points changed in these intervals without a confirmed visible contact.</p>${gaps.map(gap=>`<a class="button secondary" href="/api/jobs/${encodeURIComponent(job.job_id)}/video#t=${gap.start.toFixed(3)},${gap.end.toFixed(3)}" target="_blank" rel="noopener">Source ${UI.time(gap.start)}–${UI.time(gap.end)}</a>`).join(' ')}</details>`:''}
      ${UI.finished(job)&&!hasShots?'<p class="version-detail">No shots selected automatically.</p>':''}
      ${settings.messages?.has(job.job_id)?`<p class="job-feedback" role="status">${escape(settings.messages.get(job.job_id))}</p>`:''}
      ${settings.capcutMessages?.has(job.job_id)?`<p class="capcut-status job-feedback" role="status">${escape(settings.capcutMessages.get(job.job_id))}</p>`:''}
      </div><div class="job-actions">${controls}</div></article>`;
  }
  function family(group, settings={}) {
    const rows = group.versions.map(version=>{
      if (version.job) {
        const historyKey = actionKey(group.key,version.id);
        return `<section class="version-block" aria-label="${escape(version.label)} version">${run(version.job,version.label,settings)}${version.history.length?`<details class="version-history" data-history="${escape(historyKey)}" ${settings.expanded?.has(historyKey)?'open':''}><summary>${version.history.length} other ${escape(version.label)} ${version.history.length===1?'run':'runs'}</summary>${version.history.map(job=>run(job,version.label,settings,true)).join('')}</details>`:''}</section>`;
      }
      const key = actionKey(group.key,version.id), pending = settings.pending?.has(key);
      const supported = version.id==='all' || (version.id==='pots_and_fouls' ? settings.pointsAvailable : settings.potsAvailable);
      const canCreate = group.base && supported && !pending;
      return `<section class="version-empty" aria-label="${escape(version.label)} version"><div class="job-copy"><div class="job-title-row"><h4>${escape(version.label)}</h4><span class="status-badge empty">Not created</span></div><p class="version-detail">${pending?'Starting this version…':!group.base?'Available after shot detection finishes.':supported?'Create from saved shots.': 'This selection is currently unavailable.'}</p>${settings.messages?.has(key)?`<p class="job-feedback" role="status">${escape(settings.messages.get(key))}</p>`:''}</div><div class="job-actions"><button class="btn-create-version secondary" data-family="${escape(group.key)}" data-selection="${version.id}" ${canCreate?'':'disabled'}>${pending?'Starting…':`Create ${escape(version.label)}`}</button></div></section>`;
    }).join('');
    return `<article class="match-card"><header class="match-heading"><div><h3 title="${escape(UI.filename(group.titleJob))}">${escape(UI.title(group.titleJob))}</h3><p class="match-summary">${group.created} of ${Library.types.length} versions created${group.missing?` · ${group.missing} not created`:''} · ${date(group.jobs[0])}</p></div></header><div class="match-versions">${rows}</div></article>`;
  }
  function render(settings) {
    const el = root.document.getElementById('jobs');
    if (el.querySelector('[data-capcut-active], [data-continue-active], .menu[open]')) return;
    const all = Library.groups(settings.jobs);
    const query = root.document.getElementById('job-search').value;
    const groups = all.filter(group=>Library.matches(group,settings.filter,query));
    root.document.getElementById('jobs-count').textContent = all.length;
    root.document.querySelectorAll('[data-filter-count]').forEach(node=>{
      node.textContent = all.filter(group=>Library.matches(group,node.dataset.filterCount)).length;
    });
    root.document.querySelectorAll('[data-job-filter]').forEach(node=>node.setAttribute('aria-pressed',String(node.dataset.jobFilter===settings.filter)));
    const more = root.document.getElementById('show-more-jobs');
    more.classList.toggle('hidden',groups.length<=settings.visible);
    more.textContent = `Show ${Math.min(20,Math.max(0,groups.length-settings.visible))} more videos`;
    if (!groups.length) {
      el.innerHTML = `<div class="empty-state"><strong>${all.length?'No matching videos':'Your highlights will appear here'}</strong><p>${all.length?'Try another name or switch the filter.':'Choose a match above to create your first video.'}</p></div>`;
      return;
    }
    const focused = root.document.activeElement?.closest('[data-job-id], [data-family]');
    const focusId = focused?.dataset.jobId, focusFamily = focused?.dataset.family;
    const focusSelection = focused?.dataset.selection, focusClass = focused?.className;
    const expanded = new Set([...el.querySelectorAll('.version-history[open]')].map(node=>node.dataset.history));
    el.innerHTML = groups.slice(0,settings.visible).map(group=>family(group,{...settings,expanded})).join('');
    el.querySelectorAll('.menu').forEach(menu=>menu.addEventListener('toggle',()=>{if(!menu.open) settings.refresh();}));
    if (focused) [...el.querySelectorAll('[data-job-id], [data-family]')].find(node=>node.className===focusClass
      && (focusId ? node.dataset.jobId===focusId : node.dataset.family===focusFamily&&node.dataset.selection===focusSelection))?.focus({preventScroll:true});
  }
  const api = {render,family,run,actionKey};
  root.snookerLibraryView = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
