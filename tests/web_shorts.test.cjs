const {test} = require('node:test');
const assert = require('node:assert/strict');
const {settingsFrom,batchLabel,busy} = require('../apps/web/static/shorts.js');

test('batch settings retain only included shot overrides and contain no generation fields',()=>{
  const values={player_names:'  Judd Trump  ',preset:'classic',video_zoom:'1.25',video_y:'.5',title_y:'120',name_size:'92',shot_size:'72',line_spacing:'16',background:'#000000',name_color:'#FFD700',shot_color:'#FFFFFF'};
  const settings=settingsFrom(values,{1:' Mark Allen ',2:'Excluded',3:' '},[{shot_id:1},{shot_id:3}]);
  assert.equal(settings.player_names,'Judd Trump');
  assert.deepEqual(settings.shot_names,{1:'Mark Allen'});
  assert.equal(settings.video_zoom,1.25);
  assert.equal(settings.title_y,120);
  assert.equal(Object.hasOwn(settings,'voiceover'),false);
  assert.equal(Object.hasOwn(settings,'description'),false);
});

test('active batches block duplicates and stopped batches offer resume',()=>{
  assert.equal(batchLabel(null),'Create Shorts');
  assert.equal(batchLabel({status:'failed'}),'Resume Shorts');
  assert.equal(batchLabel({status:'interrupted'}),'Resume Shorts');
  assert.equal(busy({status:'queued'}),true);
  assert.equal(busy({status:'rendering'}),true);
  assert.equal(busy({status:'completed'}),false);
});
