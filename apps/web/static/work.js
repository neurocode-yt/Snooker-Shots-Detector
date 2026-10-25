/* Persistent, shared activity panel; polling does not load analysis vectors. */
(function (root) {
  const active = item => !['paused','interrupted'].includes(item.status);
  const label = item => item.status === 'pausing' ? 'Pausing…' : item.status === 'paused' ? 'Paused' : item.status === 'interrupted' ? 'Interrupted' : item.queued || item.status === 'queued' || item.status === 'pending' ? 'Queued' : 'Working';
  function createVisibility(store) {
    return {
      visible: items => items.filter(item => !store.has(item.task_id)),
      hidden: items => items.filter(item => store.has(item.task_id)),
      dismiss: item => store.add(item.task_id),
      dismissAll: items => items.forEach(item => store.add(item.task_id)),
      restore: items => items.forEach(item => store.remove(item.task_id)),
    };
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {active,label,createVisibility};
  if (!root.document) return;
  const escape = root.snookerUI.escape;
  const panel = document.createElement('section');
  panel.className = 'work-panel';
  panel.hidden = true;
  panel.setAttribute('aria-labelledby','work-heading');
  panel.innerHTML = `<div class="work-heading-row"><div><p class="eyebrow">ACTIVITY</p><h2 id="work-heading">Current work <span id="work-count" class="count-label"></span></h2></div><div class="work-heading-actions"><span class="muted work-summary">Checking work…</span><button type="button" class="dismiss-close" data-work-action="dismiss-all" aria-label="Close Current work" title="Hide current tasks; work and saved progress are kept">×</button></div></div><p class="work-error notice hidden" role="status"></p><div class="work-items" aria-label="Current tasks"><p class="muted">Checking active and paused tasks…</p></div>`;
  document.querySelector('body > header').insertAdjacentElement('afterend',panel);
  const storageKey = 'snooker.dismissed-work.v1';
  const store = root.snookerDismiss.createStore(root.snookerDismiss.storage(), storageKey);
  const visibility = createVisibility(store);
  const restore = document.createElement('button');
  restore.type = 'button'; restore.className = 'secondary work-restore'; restore.hidden = true;
  root.snookerDismiss.controls.append(restore);
  let items = [], pending = new Set();
  const polling = root.snookerPolling.create(loadWork, {isActive: () => items.some(active)});
  function render() {
    const shown = visibility.visible(items), hidden = visibility.hidden(items);
    const error = panel.querySelector('.work-error');
    panel.hidden = !shown.length && (error.classList.contains('hidden') || root.snookerDismiss.isDismissed(error));
    restore.hidden = !hidden.length;
    restore.textContent = `Show hidden work (${hidden.length})`;
    panel.querySelector('#work-count').textContent = shown.length || '';
    panel.querySelector('.work-summary').textContent = shown.length ? `${shown.filter(active).length} active / queued · ${shown.filter(item=>!active(item)).length} paused / interrupted` : 'Nothing is running';
    const focused = document.activeElement?.closest('[data-task]');
    const focusId = focused?.dataset.task, action = focused?.dataset.workAction;
    panel.querySelector('.work-items').innerHTML = shown.length ? shown.map(item=>{
      const progress = Math.round(Math.max(0,Math.min(1,item.progress||0))*100);
      const name = root.snookerUI.title(item);
      const detail = item.kind === 'capcut' ? `${item.completed} / ${item.total} clips saved` : item.kind === 'shorts' ? `${item.completed} / ${item.total} Shorts ready` : root.snookerUI.phase(item);
      const view = item.kind === 'shorts' ? `<button data-work-action="view" data-task="${escape(item.task_id)}">View Shorts</button>` : item.has_saved_detection ? `<a class="button" href="/review/${encodeURIComponent(item.job_id)}?editor=classic">View shots</a>` : '<a class="button" href="/#videos">Video library</a>';
      return `<article class="work-item"><div class="work-copy"><div class="work-title"><strong>${escape(name)}</strong><div class="work-title-actions"><span class="status-badge ${active(item)?'working':'warning'}">${label(item)}</span><button type="button" class="dismiss-close" data-work-action="dismiss" data-task="${escape(item.task_id)}" aria-label="Hide ${escape(name)} from Current work" title="Hide task; work and saved progress are kept">×</button></div></div><p class="muted">${item.kind === 'capcut' ? 'CapCut preparation' : item.kind === 'shorts' ? 'Shorts export' : item.status === 'exporting' ? 'Video export' : 'Match processing'} · ${escape(detail)} · ${progress}%</p><progress max="100" value="${progress}" aria-label="${escape(name)} progress"></progress><p class="muted work-message">${escape(item.status === 'pausing' ? 'Pause requested. Saving progress and stopping current work…' : root.snookerUI.message(item.message))}</p></div><div class="work-actions">${item.can_pause ? `<button data-work-action="pause" data-task="${escape(item.task_id)}" ${pending.has(item.task_id)?'disabled':''}>Pause</button>` : ''}${item.can_resume ? `<button class="primary" data-work-action="resume" data-task="${escape(item.task_id)}" ${pending.has(item.task_id)?'disabled':''}>Resume</button>` : ''}${view}</div></article>`;
    }).join('') : '<p class="work-empty muted">Active and queued matches, exports, and Shorts appear here. Paused work stays here until you resume it.</p>';
    if (focusId) [...panel.querySelectorAll('[data-task]')].find(node=>node.dataset.task===focusId&&node.dataset.workAction===action)?.focus({preventScroll:true});
  }
  async function loadWork() {
    try {
      const response = await fetch('/api/work');
      if (!response.ok) throw new Error(response.status === 404 ? 'Restart the server to enable Current work and Pause.' : 'Could not check current work. Retrying…');
      items = (await response.json()).items;
      panel.querySelector('.work-error').classList.add('hidden'); render();
      return true;
    } catch(error) {
      panel.querySelector('.work-error').textContent = error.message;
      panel.querySelector('.work-error').classList.remove('hidden');
      render();
      return false;
    }
  }
  panel.addEventListener('click',async event=>{
    const button = event.target.closest('[data-work-action]');
    if (!button) return;
    if (button.dataset.workAction === 'dismiss-all') { visibility.dismissAll(items); root.snookerDismiss.dismiss(panel.querySelector('.work-error')); render(); if (!restore.hidden) restore.focus(); return; }
    const item = items.find(item=>item.task_id===button.dataset.task);
    if (!item) return;
    if (button.dataset.workAction === 'dismiss') { visibility.dismiss(item); render(); (panel.hidden ? restore : panel.querySelector('[data-work-action="dismiss-all"]')).focus(); return; }
    if (button.dataset.workAction === 'view') { root.snookerShorts.open(item.job_id,button,item.batch_id); return; }
    if (pending.has(item.task_id)) return;
    pending.add(item.task_id); render();
    try {
      const response = await fetch(button.dataset.workAction === 'pause' ? item.pause_url : item.resume_url,{method:'POST'});
      if (!response.ok) { const error=await response.json(); throw new Error(typeof error.detail==='string'?error.detail:'Could not change this task'); }
      await polling.refresh();
      root.dispatchEvent(new CustomEvent('snooker:work-changed', {detail: {jobId: item.job_id}}));
    } catch(error) {
      panel.querySelector('.work-error').textContent=error.message;
      panel.querySelector('.work-error').classList.remove('hidden');
    } finally { pending.delete(item.task_id); render(); }
  });
  root.snookerWork = {refresh: polling.refresh};
  restore.addEventListener('click', () => { visibility.restore(items); render(); panel.querySelector('[data-work-action="dismiss-all"]').focus(); });
  root.addEventListener('storage', event => { if (event.key === storageKey) { store.reload(); render(); } });
  root.addEventListener('snooker:notices-changed', render);
  polling.start();
})(typeof window !== 'undefined' ? window : globalThis);
