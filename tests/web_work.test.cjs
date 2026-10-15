const {test}=require('node:test');
const assert=require('node:assert/strict');
const {active,label}=require('../apps/web/static/work.js');
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
