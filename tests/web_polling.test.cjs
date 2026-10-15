const {test} = require('node:test');
const assert = require('node:assert/strict');
const {create} = require('../apps/web/static/polling.js');

const settle = () => new Promise(setImmediate);
function surface() {
  const listeners = new Map();
  return {
    hidden: false,
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(handler);
    },
    removeEventListener(type, handler) { listeners.get(type)?.delete(handler); },
    emit(type) { for (const handler of listeners.get(type) || []) handler(); },
  };
}
function harness(task, options = {}) {
  let now = 0, sequence = 0;
  const timers = new Map(), document = surface(), events = surface();
  const polling = create(task, {
    document, events,
    setTimer(fn, delay) { const id = ++sequence; timers.set(id, {fn, at: now + delay}); return id; },
    clearTimer(id) { timers.delete(id); },
    ...options,
  });
  return {
    ...polling, document, events, timers,
    async advance(ms) {
      const end = now + ms;
      for (;;) {
        const next = [...timers.entries()].sort((a, b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > end) break;
        now = next[1].at; timers.delete(next[0]);
        await next[1].fn();
      }
      now = end;
    },
  };
}

test('idle panels read once on opening, then once every 30 seconds', async () => {
  let calls = 0;
  const poll = harness(() => { calls++; });
  await poll.start();
  await poll.start();
  assert.equal(calls, 1);
  await poll.advance(29999);
  assert.equal(calls, 1);
  await poll.advance(1);
  assert.equal(calls, 2);
  assert.equal(poll.timers.size, 1);
});

test('local actions refresh immediately; active work uses fast updates then goes idle', async () => {
  let calls = 0, active = false;
  const poll = harness(() => { calls++; }, {isActive: () => active, activeMs: 2500});
  await poll.start();
  active = true;
  await poll.refresh();
  assert.equal(calls, 2);
  await poll.advance(2499);
  assert.equal(calls, 2);
  active = false;
  await poll.advance(1);
  assert.equal(calls, 3);
  await poll.advance(29999);
  assert.equal(calls, 3);
  await poll.advance(1);
  assert.equal(calls, 4);
});

test('hidden tabs skip initial and automatic reads; focus/visibility refresh without duplicates', async () => {
  let calls = 0;
  const poll = harness(() => { calls++; });
  poll.document.hidden = true;
  await poll.start();
  await poll.advance(60000);
  assert.equal(calls, 0);
  poll.document.hidden = false;
  poll.document.emit('visibilitychange'); poll.events.emit('focus');
  await settle();
  assert.equal(calls, 1);
  poll.document.hidden = true;
  poll.document.emit('visibilitychange');
  assert.equal(poll.timers.size, 0);
  await poll.advance(60000);
  assert.equal(calls, 1);
  poll.document.hidden = false;
  poll.document.emit('visibilitychange');
  await settle();
  assert.equal(calls, 2);
});

test('slow requests never overlap or reschedule after the tab becomes hidden', async () => {
  let calls = 0, finish;
  const poll = harness(() => { calls++; return new Promise(resolve => { finish = resolve; }); });
  const first = poll.start();
  await settle();
  assert.equal(poll.start(), first);
  poll.events.emit('focus');
  await poll.advance(60000);
  assert.equal(calls, 1);
  poll.document.hidden = true; poll.document.emit('visibilitychange');
  finish(); await first;
  assert.equal(poll.timers.size, 0);
});

test('an action during a pending read queues one fresh read without overlapping requests', async () => {
  let calls = 0, finish;
  const poll = harness(() => { calls++; return new Promise(resolve => { finish = resolve; }); });
  const first = poll.start(); await settle();
  const action = poll.refresh(), anotherAction = poll.refresh();
  assert.equal(calls, 1);
  finish(); await first; await settle();
  assert.equal(calls, 2);
  finish(); await Promise.all([action, anotherAction]);
  assert.equal(calls, 2);
  assert.equal(poll.timers.size, 1);
});

test('connection failures retry after 10 seconds and return to idle after recovery', async () => {
  let calls = 0;
  const poll = harness(() => {
    calls++;
    if (calls === 1) throw new Error('offline');
    return calls !== 2;
  });
  await poll.start();
  await poll.advance(9999);
  assert.equal(calls, 1);
  await poll.advance(1);
  assert.equal(calls, 2);
  await poll.advance(10000);
  assert.equal(calls, 3);
  await poll.advance(29999);
  assert.equal(calls, 3);
  await poll.advance(1);
  assert.equal(calls, 4);
});

test('stopping removes wake listeners and an in-flight read cannot restart polling', async () => {
  let calls = 0, finish;
  const poll = harness(() => { calls++; return new Promise(resolve => { finish = resolve; }); });
  const first = poll.start(); await settle();
  poll.stop(); finish(); await first;
  poll.events.emit('focus'); poll.document.emit('visibilitychange');
  await poll.advance(60000);
  assert.equal(calls, 1);
  assert.equal(poll.timers.size, 0);
});
