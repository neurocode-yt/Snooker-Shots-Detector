/* Status reads stay fast during work and quiet while the page is idle/hidden. */
(function (root) {
  function create(task, {
    isActive = () => false, activeMs = 2000, idleMs = 30000, retryMs = 10000,
    document = root.document, events = root,
    setTimer = root.setTimeout.bind(root), clearTimer = root.clearTimeout.bind(root),
  } = {}) {
    let started = false, timer = null, inFlight = null;
    function clear() {
      if (timer !== null) clearTimer(timer);
      timer = null;
    }
    function run() {
      clear();
      if (!started) return Promise.resolve();
      if (inFlight) return inFlight;
      let success = false;
      inFlight = Promise.resolve().then(task).then(result => {
        success = result !== false;
      }, () => { /* The task displays its own connection error. Retry below. */ }).finally(() => {
        inFlight = null;
        if (started && !document?.hidden) {
          timer = setTimer(run, success ? (isActive() ? activeMs : idleMs) : retryMs);
        }
      });
      return inFlight;
    }
    function refresh() {
      // An action can finish while an earlier status read is in flight. Read
      // again afterwards so its stale response cannot leave the UI idle for 30s.
      return inFlight ? inFlight.then(run) : run();
    }
    function wake() {
      if (document?.hidden) clear();
      else run();
    }
    function start() {
      if (started) return inFlight || Promise.resolve();
      started = true;
      document?.addEventListener('visibilitychange', wake);
      events.addEventListener?.('focus', wake);
      return document?.hidden ? Promise.resolve() : run();
    }
    function stop() {
      started = false;
      clear();
      document?.removeEventListener('visibilitychange', wake);
      events.removeEventListener?.('focus', wake);
    }
    return {start, refresh, stop};
  }
  root.snookerPolling = {create};
  if (typeof module !== 'undefined' && module.exports) module.exports = {create};
})(typeof window !== 'undefined' ? window : globalThis);
