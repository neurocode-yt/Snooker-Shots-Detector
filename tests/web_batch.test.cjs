const {test} = require('node:test');
const assert = require('node:assert/strict');
const batch = require('../apps/web/static/batch.js');
const ui = require('../apps/web/static/ui.js');

test('failed submission does not stop other videos; retry reuses upload and skips submitted jobs', async () => {
  const entries = ['a', 'b', 'c'].map(file => ({file}));
  const uploads = [], starts = [];
  let rejectB = true;
  const services = {
    upload: async file => {uploads.push(file); return `/${file}.mp4`;},
    start: async (path, options) => {assert.equal(options.algorithm, 'classic'); starts.push(path); if (path === '/b.mp4' && rejectB) throw Error('queue unavailable'); return path;},
    changed: () => {},
  };
  assert.deepEqual(await batch.submit(entries, {algorithm: 'classic'}, services), {submitted: 2, failed: 1});
  assert.equal(entries[1].error, 'queue unavailable');
  rejectB = false;
  assert.deepEqual(await batch.submit(entries, {algorithm: 'changed'}, services), {submitted: 3, failed: 0});
  assert.deepEqual(uploads, ['a', 'b', 'c']);
  assert.deepEqual(starts, ['/a.mp4', '/b.mp4', '/c.mp4', '/b.mp4']);
});

test('upload failure is isolated and source trimming is not repeated after a queue failure', async () => {
  const entries = [{file: 'invalid'}, {file: 'trimmed', ranges: [{start: 2, end: 8}]}];
  let trims = 0, attempt = 0;
  const services = {
    upload: async file => {if (file === 'invalid') throw Error('file too large'); return '/source.mp4';},
    preprocess: async () => {trims++; return '/cleaned.mp4';},
    start: async path => {assert.equal(path, '/cleaned.mp4'); if (!attempt++) throw Error('temporary failure'); return 'job';},
    changed: () => {},
  };
  await batch.submit(entries, {}, services);
  assert.equal(entries[0].error, 'file too large');
  assert.deepEqual(await batch.submit(entries, {}, services), {submitted: 1, failed: 1});
  assert.equal(trims, 1);
});

test('queued matches show waiting position and block restart/continue without losing checkpoints', () => {
  const job = {status: 'pending', running: true, queued: true, queue_position: 3, has_saved_detection: true};
  assert.equal(ui.status(job).label, 'Queued');
  assert.equal(ui.phase(job), 'Waiting · queue position 3');
  assert.equal(ui.actions(job).restart, false);
  assert.equal(ui.actions(job).editor, true);
});
