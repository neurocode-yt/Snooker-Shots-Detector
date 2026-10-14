const {test} = require('node:test');
const assert = require('node:assert/strict');
const {create} = require('../apps/web/static/points.js');
const response = (data, ok = true) => ({ok, json: async () => data});

test('completed highlights report already made and keep the variant ID for CapCut', async () => {
  const calls = [], states = [];
  const api = create(async (url, options) => {
    calls.push([url, options.method]);
    return response({job_id:'saved-points', status:'ready'});
  });
  await api.run('original', state => states.push(state));
  assert.deepEqual(calls, [['/api/jobs/original/points-highlights', 'POST']]);
  assert.equal(states.at(-1).state, 'ready');
  assert.equal(states.at(-1).alreadyMade, true);
  assert.equal(states.at(-1).jobId, 'saved-points');
  assert.match(states.at(-1).message, /Pots \+ fouls highlights already made/);
});

test('new highlights follow queue, score pass and export before offering CapCut', async () => {
  const calls = [], states = [], delays = [];
  const replies = [
    {job_id:'points', status:'started'},
    {status:'pending', queued:true, progress:0},
    {status:'segmenting', running:true, progress:.8, message:'Checking pot scores at 100s'},
    {status:'exporting', running:true, progress:.92, message:'Creating finished video automatically'},
    {status:'completed', download_url:'/download', progress:1},
  ];
  const api = create(async url => {calls.push(url); return response(replies.shift());}, async ms => delays.push(ms));
  await api.run('original', state => states.push(state));
  assert.equal(calls.length, 5);
  assert.ok(calls.slice(1).every(url => url === '/api/jobs/points/progress'));
  assert.match(states[2].message, /queued/);
  assert.equal(states[3].progress, 80);
  assert.equal(states[4].progress, 92);
  assert.equal(states.at(-1).state, 'ready');
  assert.equal(states.at(-1).alreadyMade, false);
  assert.deepEqual(delays, [2500,2500,2500]);
});

test('an existing running version is followed instead of submitted again', async () => {
  const states = [], calls = [];
  const api = create(async (url, options) => {
    calls.push(options?.method || 'GET');
    return response(options ? {job_id:'in-flight',status:'running'}
      : {status:'completed',download_url:'/download'});
  });
  await api.run('original', state => states.push(state));
  assert.match(states[1].message, /already being created/);
  assert.equal(states.at(-1).jobId, 'in-flight');
  assert.deepEqual(calls, ['POST','GET']);
});

test('rapid duplicate submissions share one request and a failed request can be retried', async () => {
  let release, posts = 0;
  const gate = new Promise(done => {release = done;});
  const api = create(async () => { posts++; await gate; return response({detail:'Source moved'},false); });
  const first = api.start('original');
  const second = api.start('original');
  assert.equal(first, second);
  release();
  await assert.rejects(first, /Source moved/);
  await assert.rejects(api.start('original'), /Source moved/);
  assert.equal(posts, 2);
});

test('server rejection and unreadable responses are visible failures', async () => {
  for (const reply of [response({detail:'Source video changed'},false), {ok:false,json:async () => {throw new Error('HTML');}}]) {
    const states = [];
    await create(async () => reply).run('original', state => states.push(state));
    assert.equal(states.at(-1).state, 'error');
    assert.match(states.at(-1).message, /Source video changed|unreadable response/);
  }
});

test('interrupted score checks show a reason instead of waiting forever', async () => {
  const states = [];
  const api = create(async (url, options) => response(options ? {job_id:'points',status:'started'}
    : {status:'interrupted',error:'Worker stopped; saved work can resume'}));
  await api.run('original', state => states.push(state));
  assert.equal(states.at(-1).state, 'error');
  assert.match(states.at(-1).message, /Worker stopped/);
});

test('no scoring shots or inconclusive scores finish with an editor option', async () => {
  for (const status of ['completed','ready_for_review']) {
    const states = [];
    const api = create(async (url, options) => response(options ? {job_id:'points',status:'started'}
      : {status,message:'No confirmed scoring shots; 3 need review'}));
    await api.run('original', state => states.push(state));
    assert.equal(states.at(-1).state, 'review');
    assert.equal(states.at(-1).jobId, 'points');
    assert.match(states.at(-1).message, /3 need review/);
  }
});

test('closing progress stops browser polling and never cancels the server job', async () => {
  let active = true;
  const calls = [];
  const api = create(async (url, options) => {
    calls.push(url);
    return response(options ? {job_id:'points',status:'started'} : {status:'segmenting',progress:.8});
  }, async () => {active = false;});
  await api.run('original', () => {}, () => active);
  assert.deepEqual(calls, ['/api/jobs/original/points-highlights','/api/jobs/points/progress']);
});

test('completed metadata with a live worker waits for release before CapCut', async () => {
  const states = [];
  let checks = 0;
  const api = create(async (url, options) => response(options ? {job_id:'points',status:'started'}
    : {status:'completed',download_url:'/download',running:++checks === 1}), async () => {});
  await api.run('original', state => states.push(state));
  assert.equal(checks, 2);
  assert.equal(states.at(-1).state, 'ready');
});
