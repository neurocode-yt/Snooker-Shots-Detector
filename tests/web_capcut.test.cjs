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

test('clicking unavailable CapCut explains the reason and permits retry', async () => {
  const api=create(async () => response({available:false,reason:'Install CapCut on this PC'}));
  const {button,status}=elements();
  await api.open('job',button,status);
  assert.match(status.textContent,/Install CapCut/);
  assert.equal(button.disabled,false);
  assert.equal(button.textContent,'Open in CapCut');
  assert.equal(api.isPending('job'),false);
});

test('an old failed check cannot stop a newly available CapCut handoff', async () => {
  let checks=0, posts=0;
  const api=create(async (url) => url==='/api/capcut'
    ? response({available:++checks>1,reason:'Earlier failure'})
    : (posts++,response({message:'3 editable shots ready'})));
  await api.load();
  const {button,status}=elements();
  await api.open('job',button,status);
  assert.equal(posts,1);
  assert.equal(checks,2);
  assert.equal(status.textContent,'3 editable shots ready');
  assert.match(button.title,/editable clips/);
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
