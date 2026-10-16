"""Completed highlights can change end offsets without another detector pass."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.jobs.lease import JobLease
from snooker_ai.jobs.store import JobStore
from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.rendering.exporter import Exporter, ExportResult
from snooker_ai.types import AnalysisResult, FrameFeatures, ShotRecord, ShotSelection, ShotUpdate, VideoMetadata


def setup(config,tmp_path):
    config._data['paths'].update(jobs_dir=str(tmp_path/'jobs'),uploads_dir=str(tmp_path/'uploads'))
    source=tmp_path/'source.mp4';source.write_bytes(b'source')
    store=JobStore(config);job=store.create(source,selection='pots_and_fouls')
    shots=[ShotRecord(shot_id=1,cue_strike=10,clip_start=8,clip_end=13,
        ball_motion_end=15,manual_review_required=True,pot_status='pot',
        evidence={'stop_confirmed':True,'end_before_ball_stop_seconds':2.,'usable_source_end_timestamp':18.,
                  'minimum_clip_seconds':4.,'minimum_clip_transition_padding_seconds':.48,
                  'pots_only_base_included':True,'pot_confirmation':{'points':1}}),
        ShotRecord(shot_id=2,cue_strike=30,clip_start=28,clip_end=47.4,
        ball_motion_end=55,manual_review_required=True,pot_status='foul',
        evidence={'stop_confirmed':False,'usable_source_end_timestamp':47.4,'pots_only_base_included':True}),
        ShotRecord(shot_id=3,cue_strike=60,clip_start=58,clip_end=70,included=False,pot_status='no_pot')]
    result=AnalysisResult(job_id=job,source_path=str(source),metadata=VideoMetadata(path=str(source),duration=100,fps=30,width=1280,height=720),
        original_duration=100,shots=shots,features=[FrameFeatures(t=99.5,motion_raw=.123)],
        selection=ShotSelection.POTS_AND_FOULS,pot_selection_summary={'pot':1,'foul':1,'included':2})
    store.save_analysis(result);store.update_progress(job,1,'completed','Ready',shots_detected=3)
    path=store.job_dir(job)/'export/highlights.mp4';path.parent.mkdir();path.write_bytes(b'old video')
    return store,job


def test_edit_offsets_are_absolute_reversible_and_preserve_detection(config,tmp_path):
    store,job=setup(config,tmp_path)
    original=store.load_analysis(job)
    store.trim_ends(job,seconds=3.,shot_ids=[2])
    first=store.load_analysis(job).shots[1]
    assert first.clip_end==pytest.approx(44.4)
    assert store.trim_ends(job,seconds=3.,shot_ids=[2])['changed_shots']==0
    store.trim_ends(job,seconds=5.,shot_ids=[2]);assert store.load_analysis(job).shots[1].clip_end==pytest.approx(42.4)
    store.trim_ends(job,seconds=0.,shot_ids=[2],reset=True)
    assert store.load_analysis(job).shots[1].clip_end==47.4
    store.trim_ends(job,seconds=3.,shot_ids=[2]);restored=store.load_analysis(job)
    assert restored.shots[1].clip_end==first.clip_end
    assert restored.shots[1].ball_motion_end==original.shots[1].ball_motion_end
    assert restored.shots[1].physical_stop_timestamp==55
    assert restored.shots[1].manual_review_required
    assert restored.shots[1].pot_status.value=='foul'
    assert restored.features==original.features
    assert restored.shots[0]==original.shots[0] and restored.shots[2]==original.shots[2]
    Exporter(config)._validate_strict_boundaries([restored.shots[1]],source_duration=100,source_fps=30)
    fresh=[s.model_copy(deep=True) for s in original.shots]
    assert Analyzer(config,store.job_dir(job))._preserve_user_edits(fresh,job)[1].clip_end==first.clip_end


def test_existing_automatic_trim_can_be_changed_without_double_cutting(config,tmp_path):
    store,job=setup(config,tmp_path)
    store.trim_ends(job,seconds=1.,shot_ids=[1]);assert store.load_analysis(job).shots[0].clip_end==14
    store.trim_ends(job,seconds=0.,shot_ids=[1]);assert store.load_analysis(job).shots[0].clip_end==15
    store.trim_ends(job,seconds=0.,shot_ids=[1],reset=True);assert store.load_analysis(job).shots[0].clip_end==13
    store.trim_ends(job,seconds=1.,shot_ids=[1]);assert store.load_analysis(job).shots[0].clip_end==14


def test_legacy_motion_end_and_measured_physical_stop_remain_distinct(config,tmp_path):
    store,job=setup(config,tmp_path)
    r=store.load_analysis(job);shot=r.shots[1]
    shot.ball_motion_end=54.5;shot.physical_stop_timestamp=55.
    shot.last_ball_motion_timestamp=54.4;shot.stop_confirmation_timestamp=55.5
    store.save_analysis(r)
    for reset in (False,True):
        store.trim_ends(job,seconds=3.,shot_ids=[2],reset=reset)
        saved=store.load_analysis(job).shots[1]
        assert (saved.ball_motion_end,saved.physical_stop_timestamp,
                saved.last_ball_motion_timestamp,saved.stop_confirmation_timestamp)==(54.5,55.,54.4,55.5)


def test_old_short_long_policy_applies_to_included_shots_only(config,tmp_path):
    store,job=setup(config,tmp_path)
    summary=store.trim_ends(job,seconds=1.,long_seconds=5.,long_threshold=7.)
    shots=store.load_analysis(job).shots
    assert [s.clip_end for s in shots]==pytest.approx([14,42.4,70])
    assert summary['changed_shots']==2 and summary['affected_shots']==2
    assert summary['settings']==store.get_meta(job)['end_trim_settings']
    assert store.load_analysis(job).pot_selection_summary['included']==2


def test_minimum_playback_and_impact_survive_excessive_trim(config,tmp_path):
    store,job=setup(config,tmp_path)
    summary=store.trim_ends(job,seconds=300.)
    shots=store.load_analysis(job).shots
    assert shots[0].clip_end==pytest.approx(12.48)
    assert shots[1].clip_end==pytest.approx(32.48)
    assert summary['limited_shots']==2
    assert all(s.clip_end>=s.cue_strike+2 for s in shots)


def test_trim_reference_cannot_extend_past_source_or_next_shot(config,tmp_path):
    store,job=setup(config,tmp_path)
    r=store.load_analysis(job);r.shots[0].physical_stop_timestamp=r.shots[0].ball_motion_end=40
    r.shots[0].evidence['usable_source_end_timestamp']=100
    store.save_analysis(r);store.trim_ends(job,seconds=0.,shot_ids=[1])
    assert store.load_analysis(job).shots[0].clip_end==28  # next shot lead-in


def test_manual_boundary_split_and_merge_invalidate_old_trim_reference(config,tmp_path):
    store,job=setup(config,tmp_path)
    store.trim_ends(job,seconds=3.,shot_ids=[2])
    store.update_shot(job,2,ShotUpdate(clip_end=40))
    assert 'end_trim' not in store.load_analysis(job).shots[1].evidence
    store.trim_ends(job,seconds=1.,shot_ids=[2]);assert store.load_analysis(job).shots[1].clip_end==39
    split=store.split_shot(job,2,35)
    assert all('end_trim' not in s.evidence for s in split if 28<=s.clip_start<40)
    store.trim_ends(job,seconds=1.,shot_ids=[2])
    merged=store.merge_shots(job,[2,3]);assert 'end_trim' not in merged.evidence


def test_saving_unchanged_rounded_boundaries_does_not_make_trim_cumulative(config,tmp_path):
    store,job=setup(config,tmp_path)
    store.trim_ends(job,seconds=3.,shot_ids=[2])
    store.update_shot(job,2,ShotUpdate(clip_start=28.,cue_strike=30.,clip_end=44.4004))
    store.trim_ends(job,seconds=5.,shot_ids=[2])
    assert store.load_analysis(job).shots[1].clip_end==pytest.approx(42.4)


def test_api_uses_saved_shots_preserves_old_export_and_reexports_new_ends(config,tmp_path,monkeypatch):
    store,job=setup(config,tmp_path)
    monkeypatch.setattr('apps.api.main.create_analyzer',lambda *a,**k:pytest.fail('Detector reran'))
    def export(self,result,directory,request):
        assert result.shots[1].clip_end==pytest.approx(44.4)
        p=Path(directory)/'highlights.mp4';p.write_bytes(b'updated trim');return ExportResult(joined_path=p)
    monkeypatch.setattr('apps.api.main.Exporter.export',export)
    with TestClient(create_app(config)) as client:
        response=client.post(f'/api/jobs/{job}/trim-ends',json={'seconds':3,'shot_ids':[2]})
        assert response.status_code==200 and response.json()['changed_shots']==1
        assert response.json()['shots'][1]['clip_end_timestamp']==pytest.approx(44.4)
        assert client.get(f'/api/jobs/{job}/download/highlights').status_code==409
        assert (store.job_dir(job)/'export/highlights.mp4').read_bytes()==b'old video'
        assert client.post(f'/api/jobs/{job}/export',json={'export_clips':False}).status_code==200
        assert client.get(f'/api/jobs/{job}/download/highlights').content==b'updated trim'


def test_capcut_draft_uses_the_updated_source_end(config,tmp_path):
    from snooker_ai.rendering.capcut import CapCutInstallation,create_capcut_project
    store,job=setup(config,tmp_path)
    store.trim_ends(job,seconds=3.,shot_ids=[2])
    executable=tmp_path/'CapCut.exe';executable.touch()
    install=CapCutInstallation(executable,tmp_path/'drafts','9.5.0.4050')
    project=create_capcut_project(store.load_selection_analysis(job),config,install)
    content=json.loads((Path(project['project_path'])/'draft_content.json').read_text())
    segment=next(s for s in content['tracks'][0]['segments'] if s['desc']=='Shot 2')
    assert segment['source_timerange']=={'start':28_000_000,'duration':16_400_000}


@pytest.mark.parametrize('body',[{'seconds':-1},{'seconds':3601},{'long_seconds':-1},
    {'long_threshold':-1},{'shot_ids':[]},{'seconds':'NaN'}])
def test_invalid_settings_are_rejected_before_saving(config,tmp_path,body):
    store,job=setup(config,tmp_path);before=(store.job_dir(job)/'analysis.json').read_bytes()
    with TestClient(create_app(config)) as client:
        assert client.post(f'/api/jobs/{job}/trim-ends',json=body).status_code==422
    assert (store.job_dir(job)/'analysis.json').read_bytes()==before


def test_unknown_ids_and_active_worker_cannot_partially_trim(config,tmp_path):
    store,job=setup(config,tmp_path);before=(store.job_dir(job)/'analysis.json').read_bytes()
    with TestClient(create_app(config)) as client:
        assert client.post(f'/api/jobs/{job}/trim-ends',json={'shot_ids':[1,99]}).status_code==404
        with JobLease(store.job_dir(job)):
            assert client.post(f'/api/jobs/{job}/trim-ends',json={'seconds':3}).status_code==409
    assert (store.job_dir(job)/'analysis.json').read_bytes()==before
