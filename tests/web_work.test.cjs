const {test}=require('node:test');
const assert=require('node:assert/strict');
const {active,label,createVisibility}=require('../apps/web/static/work.js');
const {createStore}=require('../apps/web/static/dismiss.js');
const UI=require('../apps/web/static/ui.js');
const shorts=require('../apps/web/static/shorts.js');

test('paused work stays visible and offers resume without being labelled failed',()=>{
  const job={status:'paused',can_continue:true};
  assert.equal(UI.status(job).label,'Paused');
  assert.equal(UI.active(job),false);
  assert.equal(UI.actions(job).resume,true);
  assert.equal(active(job),false);
  assert.equal(label(job),'Paused');
  assert.equal(shorts.batchLabel(job),'Resume Shorts');
});

test('pause requests remain active until acknowledged by the worker',()=>{
  assert.equal(UI.status({status:'analyzing',pause_requested:true,running:true}).label,'Pausing…');
  assert.equal(UI.actions({status:'paused',running:true,can_continue:true}).resume,false);
  assert.equal(active({status:'pausing'}),true);
  assert.equal(label({status:'queued'}),'Queued');
});

function browserStorage() {
  const saved=new Map();
  return {getItem:key=>saved.get(key)||null,setItem:(key,value)=>saved.set(key,value)};
}

test('dismissed work stays hidden through polling and page reload without changing saved task state',()=>{
  const storage=browserStorage();
  const interrupted=Object.freeze({task_id:'match:old',status:'interrupted',progress:.92,can_resume:true});
  const running=Object.freeze({task_id:'export:new',status:'exporting',progress:.3});
  const visibility=createVisibility(createStore(storage,'work'));
  visibility.dismiss(interrupted);
  assert.deepEqual(visibility.visible([interrupted,running]),[running]);
  const refreshed={...interrupted,progress:.95,message:'Progress updated'};
  const afterReload=createVisibility(createStore(storage,'work'));
  assert.deepEqual(afterReload.visible([refreshed,running]),[running]);
  assert.equal(interrupted.status,'interrupted');
  assert.equal(interrupted.can_resume,true);
  afterReload.restore([refreshed]);
  assert.deepEqual(afterReload.visible([refreshed,running]),[refreshed,running]);
});

test('closing all work hides only existing task IDs and lets new work appear',()=>{
  const storage=browserStorage(), visibility=createVisibility(createStore(storage,'work'));
  const tasks=[{task_id:'match:one'},{task_id:'shorts:two'}];
  visibility.dismissAll(tasks);
  const later={task_id:'capcut:three'};
  assert.deepEqual(visibility.hidden([...tasks,later]),tasks);
  assert.deepEqual(visibility.visible([...tasks,later]),[later]);
  visibility.restore(tasks);
  assert.deepEqual(createVisibility(createStore(storage,'work')).visible(tasks),tasks);
});

test('hidden work synchronizes with another page and tolerates unavailable browser storage',()=>{
  const storage=browserStorage(), first=createStore(storage,'work'), second=createStore(storage,'work');
  first.add('task'); second.reload(); assert.equal(second.has('task'),true);
  second.remove('task'); first.reload(); assert.equal(first.has('task'),false);
  const blocked={getItem(){throw new Error('Blocked');},setItem(){throw new Error('Full');}};
  const fallback=createVisibility(createStore(blocked,'work')), item={task_id:'task'};
  fallback.dismiss(item); assert.deepEqual(fallback.visible([item]),[]);
  fallback.restore([item]); assert.deepEqual(fallback.visible([item]),[item]);
});

test('invalid saved preferences do not break work and message preferences stay separate',()=>{
  const storage=browserStorage(); storage.setItem('work','broken JSON');
  const work=createStore(storage,'work'), notices=createStore(storage,'notices');
  work.add('match:one'); notices.add('message'); notices.clear();
  assert.equal(work.has('match:one'),true);
  assert.equal(notices.has('message'),false);
  storage.setItem('work',JSON.stringify([null,42,'safe'])); work.reload();
  assert.equal(work.has('safe'),true); assert.equal(work.has(null),false);
});
