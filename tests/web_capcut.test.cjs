const {test} = require('node:test');
const assert = require('node:assert/strict');
const {create} = require('../apps/web/static/capcut.js');
const response = (data, ok=true) => ({ok, json: async () => data});
const elements = () => ({button: {textContent:'Open in CapCut',disabled:false}, status:{textContent:'',classList:{remove:()=>{}}}});

test('unavailable capability expires and can recover on a fresh click', async () => {
  let time=0, calls=0;
  const api = create(async () => response(++calls === 1 ? {available:false,reason:'Not found'} : {available:true}), () => time);
  assert.equal((await api.load()).available,false);
  assert.equal((await api.load()).available,false);
  assert.equal(calls,1);
  time=5001;
  assert.equal((await api.load()).available,true);
  assert.equal(calls,2);
});

test('opening reports an actual server failure and permits retry', async () => {
  const api=create(async () => response({detail:'Installation drive unavailable'},false));
  const {button,status}=elements();
  await api.open('job',button,status);
  assert.match(status.textContent,/Installation drive unavailable/);
  assert.equal(button.disabled,false);
  assert.equal(button.textContent,'Open in CapCut');
  assert.equal(api.isPending('job'),false);
});

test('an old failed check cannot stop a real CapCut handoff', async () => {
  let checks=0, posts=0;
  const api=create(async (url) => url==='/api/capcut'
    ? response({available:++checks>1,reason:'Earlier failure'})
    : (posts++,response({message:'3 editable shots ready'})));
  await api.load();
  const {button,status}=elements();
  await api.open('job',button,status);
  assert.equal(posts,1);
  assert.equal(checks,1);
  assert.equal(status.textContent,'3 editable shots ready');
  assert.match(button.title,/editable clips/);
});

test('an in-flight negative precheck cannot veto a successful opening', async () => {
  let resolve;
  const waiting=new Promise(done=>{resolve=done;});
  let posts=0;
  const api=create(async url=>url==='/api/capcut' ? (await waiting,response({available:false}))
    : (posts++,response({message:'176 editable shots ready'})));
  const oldCheck=api.load();
  const {button,status}=elements();
  await api.open('job',button,status);
  assert.equal(posts,1);
  assert.equal(status.textContent,'176 editable shots ready');
  resolve();assert.equal((await oldCheck).available,true);
  assert.equal((await api.load()).available,true);
  assert.equal(button.disabled,false);
});

test('one in-flight project blocks duplicate clicks and restores the button on failure', async () => {
  let resolve, posts=0;
  const wait=new Promise(done => {resolve=done;});
  const api=create(async (url) => url==='/api/capcut' ? response({available:true})
    : (posts++,await wait,response({detail:'Source moved'},false)));
  const {button,status}=elements();
  const first=api.open('job',button,status);
  await api.load();
  assert.equal(api.isPending('job'),true);
  assert.equal(button.disabled,true);
  await api.open('job',button,status);
  resolve(); await first;
  assert.equal(posts,1);
  assert.match(status.textContent,/Source moved/);
  assert.equal(button.disabled,false);
});

test('Android handoff uses the native importer and displays per-clip progress', async () => {
  const events = new Map();
  global.addEventListener = (name, callback) => events.set(name, callback);
  global.removeEventListener = name => events.delete(name);
  let request, posts = 0;
  global.SnookerAndroid = {
    capcutAvailable: () => true,
    openCapCut: (job, id) => { request = {job, id}; },
  };
  try {
    const api = create(async () => { posts++; throw Error('Desktop route must not run'); });
    assert.equal((await api.load()).available, true);
    const {button, status} = elements();
    const opened = api.open('phone-job', button, status);
    assert.equal(request.job, 'phone-job');
    const update = events.get('snooker-capcut');
    update({detail:{requestId:request.id,message:'CapCut: 1/2 clips saved',done:false}});
    assert.equal(status.textContent, 'CapCut: 1/2 clips saved');
    assert.equal(api.isPending('phone-job'), true);
    update({detail:{requestId:request.id,message:'Sent 2 separate clips to CapCut',done:true}});
    await opened;
    assert.equal(posts, 0);
    assert.equal(button.disabled, false);
    assert.equal(events.size, 0);
  } finally {
    delete global.SnookerAndroid; delete global.addEventListener; delete global.removeEventListener;
  }
});
