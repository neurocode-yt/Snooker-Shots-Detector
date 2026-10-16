const {test}=require('node:test');const assert=require('node:assert/strict');
const trim=require('../apps/web/static/end-trim.js');
test('bulk settings retain the old separate normal and long shot offsets',()=>{
  assert.deepEqual(trim.payload({normal:'2',long:'3',threshold:'7'}),{seconds:2,long_seconds:3,long_threshold:7});
  assert.deepEqual(trim.payload({normal:'5',shotId:2}),{seconds:5,shot_ids:[2]});
  assert.deepEqual(trim.payload({shotId:2,reset:true}),{reset:true,shot_ids:[2]});
});
test('empty invalid and nonfinite amounts cannot silently become zero trims',()=>{
  for(const value of ['',-1,'NaN',Infinity,3601])assert.throws(()=>trim.payload({normal:value,shotId:2}));
  assert.equal(trim.payload({normal:0,shotId:2}).seconds,0);
});
test('effective trim and minimum playback limits remain visible',()=>{
  const shot={clip_start:8,clip_end:12.48,evidence:{end_trim:{active:true,seconds:10,applied_seconds:2.52}}};
  assert.match(trim.state(shot).description,/2.5s removed.*minimum playback protected/);
  assert.equal(trim.state(shot).seconds,10);
  assert.match(trim.state({...shot,evidence:{end_trim:{active:false}}}).description,/Previous ending restored/);
  assert.match(trim.message({changed_shots:2,limited_shots:1}),/2 shot endings changed.*1 kept extra/);
});
