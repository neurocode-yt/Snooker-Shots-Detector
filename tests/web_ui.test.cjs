const {test} = require('node:test');
const assert = require('node:assert/strict');
const ui = require('../apps/web/static/ui.js');

test('finished videos offer conversion without a misleading Continue action', () => {
  const job = {status: 'completed', algorithm: 'classic', can_continue: true, can_create_points: true};
  assert.equal(ui.actions(job, true).resume, false);
  assert.equal(ui.actions(job, true).points, true);
  assert.equal(ui.actions({...job, selection: 'pots_and_fouls'}, true).points, false);
  assert.equal(ui.actions(job, false).points, false);
});
test('a live worker blocks destructive and duplicate actions even with stale status', () => {
  const job = {status: 'completed', running: true, can_create_points: true, can_continue: true};
  assert.equal(ui.actions(job, true).restart, false);
  assert.equal(ui.actions(job, true).points, false);
  assert.equal(ui.actions(job, true).resume, false);
  assert.equal(ui.status(job).label, 'Processing');
});
test('only stopped jobs with a resumable checkpoint offer Continue', () => {
  for (const status of ['failed', 'interrupted', 'cancelled']) {
    assert.equal(ui.actions({status, can_continue: true}).resume, true);
    assert.equal(ui.actions({status, can_continue: false}).resume, false);
  }
  assert.equal(ui.actions({status: 'refining', can_continue: true}).resume, false);
});
test('library search reaches old videos and attention includes unresolved exported shots', () => {
  const job = {source_path: 'G:\\matches\\20261012-174918_dff74362_David_Lilley_vs_Liu.mp4', job_id: 'older-job', status: 'completed', pot_selection_summary: {unknown: 22}};
  assert.equal(ui.title(job), 'David Lilley vs Liu');
  assert.equal(ui.matches(job, 'ready', 'LIU'), true);
  assert.equal(ui.matches(job, 'ready', 'David Lilley'), true);
  assert.equal(ui.matches(job, 'attention'), true);
  assert.equal(ui.matches(job, 'processing'), false);
  assert.equal(ui.matches(job, 'all', 'older-job'), true);
});
test('filenames and errors are escaped for both text and menu attributes', () => {
  assert.equal(ui.escape('" onfocus="alert(1)<script>&'), '&quot; onfocus=&quot;alert(1)&lt;script&gt;&amp;');
  assert.equal(ui.filename({source_path: '/matches/original.name.mp4'}), 'original.name.mp4');
});
test('long-match clocks and mode labels remain clear', () => {
  assert.equal(ui.time(5990.016), '1:39:50');
  assert.equal(ui.time(-4), '00:00');
  assert.equal(ui.mode({selection: 'pots_only'}), 'Pots only');
  assert.equal(ui.mode({selection: 'pots_and_fouls'}), 'Pots + fouls');
  assert.equal(ui.mode({auto_export: false}), 'Review & edit');
});
test('saved edits require a fresh export and remain easy to find', () => {
  const job = {status: 'completed', export_needs_update: true, can_continue: true};
  assert.equal(ui.status(job).label, 'Edits saved');
  assert.equal(ui.matches(job, 'attention'), true);
  assert.equal(ui.actions(job).resume, false);
  assert.equal(ui.actions(job).editor, true);
});
test('processing progress does not reload media; completion loads the saved timeline', () => {
  const loaded = {status: 'refining', has_saved_detection: true, timeline_revision: 'old'};
  assert.equal(ui.needsTimelineRefresh({...loaded, timeline_revision: 'new'}, loaded), false);
  const completed = {status: 'completed', has_saved_detection: true, timeline_revision: 'new'};
  assert.equal(ui.needsTimelineRefresh(completed, loaded), true);
  assert.equal(ui.needsTimelineRefresh(completed, completed), false);
  assert.equal(ui.needsTimelineRefresh(completed, null), true);
});
test('legacy processing jobs still update once and unsaved timelines are not requested', () => {
  const loaded = {status: 'refining', has_saved_detection: true};
  const completed = {status: 'completed', has_saved_detection: true};
  assert.equal(ui.needsTimelineRefresh(completed, loaded), true);
  assert.equal(ui.needsTimelineRefresh(completed, completed), false);
  assert.equal(ui.needsTimelineRefresh({status: 'interrupted', has_saved_detection: false}, null), false);
});
