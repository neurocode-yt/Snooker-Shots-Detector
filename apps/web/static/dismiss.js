/* Closing app messages only hides UI; it never cancels or deletes saved work. */
(function (root) {
  const NOTICE_KEY = 'snooker.dismissed-notices.v1';
  function createStore(storage, key) {
    let values = new Set();
    function reload() {
      try {
        const saved = JSON.parse(storage?.getItem(key) || '[]');
        if (Array.isArray(saved)) values = new Set(saved.filter(v => typeof v === 'string').slice(-500));
      } catch (_) { /* Keep this tab usable if browser storage is unavailable. */ }
    }
    function save() {
      values = new Set([...values].slice(-500));
      try { storage?.setItem(key, JSON.stringify([...values])); } catch (_) { /* Session fallback. */ }
    }
    reload();
    return {
      has: key => values.has(key),
      add(key) { values.add(key); save(); },
      remove(key) { values.delete(key); save(); },
      clear() { values.clear(); save(); },
      reload,
    };
  }
  function storage() { try { return root.localStorage; } catch (_) { return null; } }
  const api = {createStore, storage};
  root.snookerDismiss = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (!root.document) return;

  const doc = root.document;
  const dismissed = createStore(storage(), NOTICE_KEY);
  const selector = '.notice, .job-feedback, .job-warning, .capcut-status, .toast';
  const controls = doc.createElement('div');
  controls.className = 'dismissed-tools';
  controls.setAttribute('aria-label', 'Hidden messages and work');
  const restore = doc.createElement('button');
  restore.type = 'button'; restore.className = 'secondary notice-restore'; restore.hidden = true;
  restore.textContent = 'Show hidden notices';
  controls.append(restore);
  doc.querySelector('body > header').after(controls);
  api.controls = controls;
  function text(node) {
    return [...node.childNodes].filter(n => !n.matches?.('[data-dismiss-notice]'))
      .map(n => n.textContent).join('').trim();
  }
  function noticeKey(node) {
    const owner = node.closest('[data-library-job], [data-family]');
    const identity = node.id || `${owner?.dataset.libraryJob || owner?.dataset.family || ''}:${[...node.classList].filter(c => !['hidden', 'is-dismissed', 'dismissible'].includes(c)).sort().join('.')}`;
    return `${root.location.pathname}:${identity}:${text(node)}`;
  }
  function closeButton(label, attribute) {
    const button = doc.createElement('button');
    button.type = 'button'; button.className = 'dismiss-close';
    button.setAttribute('aria-label', label); button.title = label;
    button.setAttribute(attribute, '');
    button.innerHTML = '<span aria-hidden="true">×</span>';
    return button;
  }
  function changed() { root.dispatchEvent(new root.CustomEvent('snooker:notices-changed')); }
  api.isDismissed = node => dismissed.has(noticeKey(node));
  api.dismiss = node => {
    if (!text(node)) return;
    dismissed.add(noticeKey(node));
    node.classList.add('is-dismissed'); schedule(); changed();
  };
  function scan() {
    observer.disconnect();
    let hidden = 0;
    for (const node of doc.querySelectorAll(selector)) {
      if (!text(node)) {
        node.classList.remove('is-dismissed', 'dismissible');
        node.querySelector('[data-dismiss-notice]')?.remove();
        continue;
      }
      node.classList.add('dismissible');
      if (!node.querySelector('[data-dismiss-notice]')) node.append(closeButton('Dismiss message', 'data-dismiss-notice'));
      const closed = dismissed.has(noticeKey(node));
      node.classList.toggle('is-dismissed', closed);
      if (closed && !node.hidden && !node.classList.contains('hidden')) hidden++;
    }
    restore.hidden = !hidden;
    for (const dialog of doc.querySelectorAll('dialog')) {
      if (!dialog.querySelector('[data-dismiss-dialog], .shorts-close, .source-close')) {
        dialog.classList.add('dismissible-dialog');
        dialog.append(closeButton('Close popup', 'data-dismiss-dialog'));
      }
    }
    for (const menu of doc.querySelectorAll('.menu-panel')) {
      if (!menu.querySelector('[data-dismiss-menu]')) menu.prepend(closeButton('Close menu', 'data-dismiss-menu'));
    }
    observer.observe(doc.body, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['class', 'hidden', 'open']});
  }
  let queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    Promise.resolve().then(() => { queued = false; scan(); });
  }
  const observer = new root.MutationObserver(schedule);
  doc.addEventListener('click', event => {
    const notice = event.target.closest('[data-dismiss-notice]');
    const popup = event.target.closest('[data-dismiss-dialog]');
    const menu = event.target.closest('[data-dismiss-menu]');
    if (notice) {
      event.preventDefault(); event.stopPropagation();
      api.dismiss(notice.parentElement);
    } else if (popup) {
      event.preventDefault(); popup.closest('dialog').close();
    } else if (menu) {
      event.preventDefault(); menu.closest('details').open = false;
      menu.closest('details').querySelector('summary').focus();
    }
  });
  restore.addEventListener('click', () => { dismissed.clear(); scan(); changed(); });
  root.addEventListener('storage', event => { if (event.key === NOTICE_KEY) { dismissed.reload(); scan(); changed(); } });
  scan();
})(typeof window !== 'undefined' ? window : globalThis);
