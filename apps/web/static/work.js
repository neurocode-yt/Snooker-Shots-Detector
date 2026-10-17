/* Persistent, shared activity panel; polling does not load analysis vectors. */
(function (root) {
  const active = item => !['paused','interrupted'].includes(item.status);
  const label = item => item.status === 'pausing' ? 'Pausing…' : item.status === 'paused' ? 'Paused' : item.status === 'interrupted' ? 'Interrupted' : item.queued || item.status === 'queued' || item.status === 'pending' ? 'Queued' : 'Working';
  if (typeof module !== 'undefined' && module.exports) module.exports = {active,label};
  if (!root.document) return;
  const escape = root.snookerUI.escape;
  const panel = document.createElement('section');
  panel.className = 'work-panel';
  panel.setAttribute('aria-labelledby','work-heading');
  panel.innerHTML = `<div class="work-heading-row"><div><p class="eyebrow">ACTIVITY</p><h2 id="work-heading">Current work <span id="work-count" class="count-label"></span></h2></div><span class="muted work-summary">Checking work…</span></div><p class="work-error notice hidden" role="status"></p><div class="work-items" aria-label="Current tasks"><p class="muted">Checking active and paused tasks…</p></div>`;
  document.querySelector('body > header').insertAdjacentElement('afterend',panel);
  let items = [], pending = new Set();
  const polling = root.snookerPolling.create(loadWork, {isActive: () => items.some(active)});
  function render() {
    panel.querySelector('#work-count').textContent = items.length || '';
    panel.querySelector('.work-summary').textContent = items.length ? `${items.filter(active).length} active / queued · ${items.filter(item=>!active(item)).length} paused / interrupted` : 'Nothing is running';
    const focused = document.activeElement?.closest('[data-task]');
    const focusId = focused?.dataset.task, action = focused?.dataset.workAction;
    panel.querySelector('.work-items').innerHTML = items.length ? items.map(item=>{
      const progress = Math.round(Math.max(0,Math.min(1,item.progress||0))*100);
      const name = root.snookerUI.title(item);
      const detail = item.kind === 'shorts' ? `${item.completed} / ${item.total} Shorts ready` : root.snookerUI.phase(item);
      const view = item.kind === 'shorts' ? `<button data-work-action="view" data-task="${escape(item.task_id)}">View Shorts</button>` : item.has_saved_detection ? `<a class="button" href="/review/${encodeURIComponent(item.job_id)}?editor=classic">View shots</a>` : '<a class="button" href="/#videos">Video library</a>';
      return `<article class="work-item"><div class="work-copy"><div class="work-title"><strong>${escape(name)}</strong><span class="status-badge ${active(item)?'working':'warning'}">${label(item)}</span></div><p class="muted">${item.kind === 'shorts' ? 'Shorts export' : item.status === 'exporting' ? 'Video export' : 'Match processing'} · ${escape(detail)} · ${progress}%</p><progress max="100" value="${progress}" aria-label="${escape(name)} progress"></progress><p class="muted work-message">${escape(item.status === 'pausing' ? 'Pause requested. Saving progress and stopping current work…' : root.snookerUI.message(item.message))}</p></div><div class="work-actions">${item.can_pause ? `<button data-work-action="pause" data-task="${escape(item.task_id)}" ${pending.has(item.task_id)?'disabled':''}>Pause</button>` : ''}${item.can_resume ? `<button class="primary" data-work-action="resume" data-task="${escape(item.task_id)}" ${pending.has(item.task_id)?'disabled':''}>Resume</button>` : ''}${view}</div></article>`;
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
      return false;
    }
  }
  panel.addEventListener('click',async event=>{
    const button = event.target.closest('[data-work-action]');
    if (!button) return;
    const item = items.find(item=>item.task_id===button.dataset.task);
    if (!item) return;
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
  polling.start();
})(typeof window !== 'undefined' ? window : globalThis);
