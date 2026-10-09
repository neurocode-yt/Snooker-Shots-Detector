from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.dl_holdout import checked_features, validate_holdout


def test_holdout_rejects_model_selection_leak_in_any_ensemble_member():
    training = {'member_reports': [
        {'split_groups': {'train': ['match_a'], 'validation': []}},
        {'split_groups': {'train': ['match_b'], 'validation': ['match_c']}},
    ]}
    manifest = {'independent_holdout_available': True, 'videos': [
        {'id': 'unseen', 'group_id': 'match_c', 'split': 'holdout'}]}
    with pytest.raises(ValueError, match='overlaps'):
        validate_holdout(manifest, training)
    manifest['videos'][0]['group_id'] = 'match_d'
    assert validate_holdout(manifest, training) == {'match_a','match_b','match_c'}


def test_missing_group_provenance_cannot_be_reported_as_independent():
    manifest = {'independent_holdout_available': True, 'videos': [
        {'id': 'unseen', 'group_id': 'match_c', 'split': 'holdout'}]}
    with pytest.raises(ValueError, match='provenance'):
        validate_holdout(manifest, {})


@pytest.mark.parametrize('name', ['handling','replay','replays','event','keep','end'])
def test_holdout_rejects_annotations_that_evaluation_would_silently_ignore(name):
    manifest={'independent_holdout_available':True,'videos':[
        {'id':'unseen','group_id':'match_b','split':'holdout',name:[{'start':1.,'end':2.}]}]}
    with pytest.raises(ValueError,match='evaluated labels schema'):
        validate_holdout(manifest,{'split_groups':{'train':['match_a']}})


def make_archive(tmp_path, intervals=None, times=None):
    source = tmp_path/'source.mp4'
    source.write_bytes(b'source identity fixture')
    stat = source.stat()
    spec = {'sample_fps': 8., 'dimension': 2}
    identity = {'path':str(source.resolve()), 'size':stat.st_size, 'mtime_ns':stat.st_mtime_ns,
                'intervals':intervals or [[0., 2.]]}
    key = hashlib.sha256(json.dumps({'source':identity, 'spec':spec}, sort_keys=True).encode()).hexdigest()
    times = np.asarray(times if times is not None else np.arange(0,2,.125))
    np.savez(tmp_path/'features.npz', timestamps=times, features=np.zeros((len(times),2)),
             feature_spec=json.dumps(spec), source_identity=json.dumps(identity), fingerprint=key)
    return {'id':'unseen', 'duration':2., 'source_path':'source.mp4', 'features_path':'features.npz',
            'source_fingerprint':{'size_bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns}}


def test_whole_source_test_rejects_partial_source_feature_cache(tmp_path):
    video = make_archive(tmp_path, intervals=[[0.,1.]])
    with pytest.raises(ValueError, match='coverage mismatch'):
        checked_features(video, tmp_path)


@pytest.mark.parametrize('times', [np.arange(0,1,.125),
                                   np.r_[np.arange(0,.5,.125), np.arange(1,2,.125)]])
def test_incomplete_or_gapped_features_cannot_pass_a_whole_source_test(tmp_path, times):
    video = make_archive(tmp_path, times=times)
    with pytest.raises(ValueError, match='Incomplete'):
        checked_features(video, tmp_path)


def test_same_length_wrong_source_cannot_reuse_frozen_labels(tmp_path):
    video = make_archive(tmp_path)
    checked_features(video, tmp_path)
    another = tmp_path/'another.mp4'
    another.write_bytes((tmp_path/'source.mp4').read_bytes())
    video['source_path'] = 'another.mp4'
    with pytest.raises(ValueError, match='identity'):
        checked_features(video, tmp_path)
