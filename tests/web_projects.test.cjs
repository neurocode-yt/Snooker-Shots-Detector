const {test}=require('node:test');
const assert=require('node:assert/strict');
const UI=require('../apps/web/static/ui.js');
const Library=require('../apps/web/static/library.js');
const View=require('../apps/web/static/library-view.js');
const Projects=require('../apps/web/static/library-projects.js');
const job=(id,extra={})=>({job_id:id,source_path:`/matches/${id}.mp4`,status:'completed',...extra});

test('custom names are displayed and searchable without losing source search',()=>{
  const named=job('one',{display_name:'Final <match> & friends'});
  assert.equal(UI.title(named),'Final <match> & friends');
  assert.equal(UI.matches(named,'all','final <match>'),true);
  assert.equal(UI.matches(named,'all','one.mp4'),true);
  const [group]=Library.groups([named]);
  const html=View.family(group,{selected:new Set([group.key])});
  assert.match(html,/Final &lt;match&gt; &amp; friends/);
  assert.match(html,/data-project-select=.*checked/);
  assert.match(html,/btn-project-name/);
  assert.doesNotMatch(html,/<match>/);
});

test('selection keeps one project per source and covers matching filters',()=>{
  const jobs=[job('one'),job('version',{source_path:'/matches/one.mp4',selection:'pots_only'}),job('two')];
  const [group]=Library.groups([jobs[0]]);
  const data=Projects.selection(jobs,new Set([group.key]),'all','one');
  assert.equal(data.matching.length,1);
  assert.equal(data.chosen.length,1);
  assert.equal(data.chosen[0].jobs.length,2);
});

test('active projects cannot be selected and resource confirmation is explicit',()=>{
  const [group]=Library.groups([job('busy',{status:'exporting',running:true})]);
  assert.match(View.family(group),/data-project-select=.*disabled/);
  assert.match(Projects.summary({job_ids:['one','two'],capcut:[{}],uploads:['/video.mp4','/video.mp4.source.json']}),/2 saved versions.*1 linked CapCut project and 1 uploaded video/);
});

test('modern 32-digit upload ids are removed from default names',()=>{
  assert.equal(UI.title(job('one',{source_path:'20261024-010137_8f7d344adcb34544ac9fdc3e09ecb4af_Thepchaiya_vs_Hill.mp4'})),'Thepchaiya vs Hill');
});

test('server project identity prevents grouping distinct source revisions together',()=>{
  const jobs=[job('a',{project_id:'first'}),job('b',{source_path:'/matches/a.mp4',project_id:'second'})];
  assert.equal(Library.groups(jobs).length,2);
});
