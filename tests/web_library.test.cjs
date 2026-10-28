const {test} = require('node:test');
const assert = require('node:assert/strict');
const library = require('../apps/web/static/library.js');
const view = require('../apps/web/static/library-view.js');

function job(id, selection='all', extra={}) {
  return {job_id:id, source_path:'G:\\matches\\Player_A_vs_Player_B.mp4',
    source_fingerprint:{path:'G:\\matches\\Player_A_vs_Player_B.mp4',size:100,mtime_ns:200},
    selection,status:'completed',created_at:100,shots_detected:20,has_saved_detection:true,
    can_create_variants:true,download_url:`/api/jobs/${id}/download/highlights`,...extra};
}
const options = {pointsAvailable:true,potsAvailable:true};

test('one match shows independent saved versions and explicit missing versions',()=>{
  const [group] = library.groups([job('all'),job('points','pots_and_fouls')]);
  assert.equal(group.jobs.length,2);
  assert.equal(group.created,2);
  assert.equal(group.missing,1);
  assert.equal(group.base.job_id,'all');
  assert.deepEqual(group.versions.map(v=>v.job?.job_id),['all','points',undefined]);
  const html = view.family(group,options);
  for (const id of ['all','points']) {
    const row = html.match(new RegExp(`<article[^>]*data-library-job="${id}"[\\s\\S]*?</article>`))[0];
    assert.match(row,new RegExp(`href="/api/jobs/${id}/download/highlights"`));
    assert.match(row,new RegExp(`class="btn-capcut secondary" data-job-id="${id}"`));
    assert.match(row,new RegExp(`class="btn-shorts secondary" data-job-id="${id}"`));
    assert.match(row,new RegExp(`href="/review/${id}\\?editor=classic"`));
  }
  assert.match(html,/Pots only<\/h4><span class="status-badge empty">Not created/);
  assert.match(html,/data-selection="pots_only" >Create Pots only/);
  assert.match(html,/2 of 3 versions created/);
});

test('legacy jobs join a sole known revision but never bridge replaced files',()=>{
  const legacy=job('old','all',{source_fingerprint:undefined,selection:undefined});
  assert.equal(library.groups([legacy,job('new','pots_only')]).length,1);
  const replacement=job('replacement','pots_only',{
    source_fingerprint:{path:legacy.source_path,size:100,mtime_ns:201},
  });
  assert.equal(library.groups([legacy,job('new','pots_only'),replacement]).length,3);
  assert.equal(library.groups([legacy,{...legacy,job_id:'older'}]).length,1);
});

test('source identity keeps separate folders and algorithms apart',()=>{
  const original=job('one');
  assert.equal(library.groups([original,job('two','pots_only',{
    source_fingerprint:undefined,source_path:'G:/other/Player_A_vs_Player_B.mp4',
  })]).length,2);
  assert.equal(library.groups([original,job('two','pots_only',{source_fingerprint:{
    ...original.source_fingerprint,path:'g:/MATCHES/player_a_vs_player_b.mp4',
  }})]).length,1);
  assert.equal(library.groups([original,job('dl','all',{algorithm:'dl_algo'})]).length,2);
  assert.notEqual(library.sourcePath({source_path:'/Matches/a.mp4'}),library.sourcePath({source_path:'/matches/a.mp4'}));
  assert.equal(library.groups([{job_id:'one'},{job_id:'two'}]).length,2);
});

test('active runs stay visible while previous successful exports remain accessible',()=>{
  const [group]=library.groups([job('ready'),job('busy','all',{
    status:'exporting',running:true,download_url:undefined,created_at:101,
  }),job('failed','all',{status:'failed',download_url:undefined,created_at:102})]);
  assert.equal(group.versions[0].job.job_id,'busy');
  assert.deepEqual(group.versions[0].history.map(j=>j.job_id),['ready','failed']);
  const html=view.family(group,options);
  assert.match(html,/2 other All shots runs/);
  assert.match(html,/href="\/api\/jobs\/ready\/download\/highlights"/);
  assert.match(html,/All shots · earlier run/);
});

test('filters count matches and search every saved version',()=>{
  const [group]=library.groups([job('original'),job('filtered','pots_and_fouls',{
    status:'paused',download_url:undefined,
  })]);
  assert.equal(library.matches(group,'missing'),true);
  assert.equal(library.matches(group,'attention','filtered'),true);
  assert.equal(library.matches(group,'ready','PLAYER A'),true);
  assert.equal(library.matches(group,'processing'),false);
  assert.equal(library.matches(group,'missing','unrelated'),false);
  const [complete]=library.groups(['all','pots_only','pots_and_fouls'].map(s=>job(s,s)));
  assert.equal(library.matches(complete,'missing'),false);
});

test('detected shots and finished video exports have distinct status and actions',()=>{
  const detected=job('review','pots_only',{status:'ready_for_review',download_url:undefined});
  assert.deepEqual(library.state(detected),{label:'Shots ready',tone:'ready'});
  const html=view.run(detected,'Pots only',{});
  assert.match(html,/btn-export-version primary" data-job-id="review" >Export video/);
  assert.match(html,/btn-capcut secondary" data-job-id="review"\s+title=/);
  assert.doesNotMatch(html,/Download MP4/);
  assert.equal(library.state(job('ready')).label,'Ready');
  assert.equal(library.state({...detected,export_needs_update:true}).label,'Edits saved');
  assert.match(view.run({...detected,pot_selection_summary:{included:3},shots_detected:undefined},'Pots only',{}),/3 shots selected/);
});

test('unfinished and empty selections never offer CapCut or Shorts as ready actions',()=>{
  const empty=job('empty','pots_only',{pot_selection_summary:{included:0},download_url:undefined});
  const html=view.run(empty,'Pots only',{});
  assert.match(html,/No shots selected/);
  assert.match(html,/btn-capcut secondary" data-job-id="empty" disabled/);
  assert.doesNotMatch(html,/btn-export-version/);
  const active=view.run(job('busy','all',{status:'analyzing',running:true,has_saved_detection:false,shots_detected:0,download_url:undefined}),'All shots',{});
  assert.match(active,/Detection in progress/);
  assert.doesNotMatch(active,/btn-capcut|Download MP4/);
});

test('pending creation prevents duplicate clicks and keeps escaped errors visible',()=>{
  const [group]=library.groups([job('base')]);
  const key=view.actionKey(group.key,'pots_only');
  const html=view.family(group,{...options,pending:new Map([[key,true]]),messages:new Map([[key,'<retry> "later"']])});
  assert.match(html,/data-selection="pots_only" disabled>Starting…/);
  assert.match(html,/&lt;retry&gt; &quot;later&quot;/);
  assert.match(view.family(group,{...options,potsAvailable:false}),/data-selection="pots_only" disabled>Create Pots only/);
  const [unfinished]=library.groups([job('busy','all',{status:'analyzing',running:true})]);
  assert.match(view.family(unfinished,options),/Available after shot detection finishes/);
});

test('untrusted filenames and messages cannot become markup',()=>{
  const [group]=library.groups([job('safe','all',{source_path:'G:/matches/<script>alert(1)</script>.mp4',error:'<img onerror="bad">',message:'<unsafe>'})]);
  const html=view.family(group,options);
  assert.doesNotMatch(html,/<script>|<img |<unsafe>/);
  assert.match(html,/&lt;img onerror=&quot;bad&quot;&gt;/);
});

test('unreadable scoreboard is distinguished from zero scoring shots and can be rechecked',()=>{
  const failed=job('scores','pots_and_fouls',{download_url:undefined,
    pot_selection_summary:{included:0,unknown:20,score_readings:0}});
  const html=view.run(failed,'Pots + fouls',{});
  assert.equal(library.state(failed).label,'Scores unavailable');
  assert.match(html,/Scoreboard could not be read/);
  assert.match(html,/btn-recheck-scores secondary" data-job-id="scores" >Recheck scores/);
  assert.match(view.run(failed,'Pots + fouls',{pending:new Map([['scores',true]])}),/data-job-id="scores" disabled>Recheck scores/);
});

test('scoring details retain bounded evidence without a manual-work prompt',()=>{
  const gap=job('gap','all',{scoring_gaps_unresolved:[{start:120,end:132},
    {start:-1,end:5},{start:3,end:2},{start:'<bad>',end:6}]});
  const html=view.run(gap,'All shots',{expanded:new Set(['gaps:gap'])});
  assert.match(html,/Scoring details \(1\)/);
  assert.match(html,/data-history="gaps:gap" open/);
  assert.match(html,/\/api\/jobs\/gap\/video#t=120.000,132.000/);
  assert.match(html,/data-source-preview data-start="120.000" data-end="132.000"/);
  assert.doesNotMatch(html,/target="_blank"/);
  assert.doesNotMatch(html,/<bad>|#t=-1|#t=3.000/);
  const [group]=library.groups([gap]);
  assert.equal(library.matches(group,'attention'),false);
});

test('saved uncertain scoring results stay usable without review labels or demands',()=>{
  const saved=job('saved','pots_and_fouls',{
    message:'Pots + fouls highlights ready: 18 kept from 20 shots; 2 need review',
    pot_selection_summary:{included:18,unknown:2,score_readings:100},
    scoring_gaps_unresolved:[{start:120,end:132}],
  });
  const html=view.run(saved,'Pots + fouls',{});
  assert.match(html,/2 shots have unconfirmed scores/);
  assert.match(html,/Scoring details \(1\)/);
  assert.match(html,/Download MP4/);
  assert.match(html,/Open in CapCut/);
  assert.doesNotMatch(html.replace(/<[^>]*>/g,''),/\breview\b|need a score check|check these source/i);
  assert.equal(saved.pot_selection_summary.unknown,2);
  assert.equal(saved.scoring_gaps_unresolved.length,1);
});

test('frame context is visible and older scoring exports can be updated',()=>{
  const saved=job('bookends','pots_and_fouls',{pot_selection_summary:{policy:6,included:176,
    frames:Array(6).fill({}),frame_context_shots:12,non_scoring_context_shots:11}});
  const html=view.run(saved,'Pots + fouls',{});
  assert.match(html,/Includes opening and final shots from 6 frames · 11 additional context shots/);
  assert.doesNotMatch(html,/Update frame shots/);
  const old=view.run({...saved,pot_selection_summary:{policy:5,included:165,unknown:0}},'Pots + fouls',{});
  assert.match(old,/btn-recheck-scores.*Update frame shots/);
  const prior=view.run({...saved,pot_selection_summary:{policy:7,included:176,unknown:0}},'Pots + fouls',{});
  assert.match(prior,/btn-recheck-scores.*Recheck scores/);
});
