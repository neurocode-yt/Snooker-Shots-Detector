import hashlib
import json

import pytest

from tools.classic_calibrate import candidate_overrides, fit, validate_dataset


def case(identifier, group, split, features_path='unused.json', digest='pinned'):
    return {'id': identifier, 'source_group': group, 'split': split, 'source_reviewed': True,
            'features_path': features_path, 'features_sha256': digest,
            'window': {'start': 0., 'end': 10.}, 'contacts': []}


def test_underlying_match_cannot_leak_into_selection_or_test():
    for split in ['selection', 'test']:
        manifest = {'cases': [case('one', 'same_match', 'development'), case('two', 'same_match', split)]}
        with pytest.raises(ValueError, match='multiple splits'):
            validate_dataset(manifest)


def test_predictions_and_unpinned_features_cannot_be_ground_truth():
    for key, value in [('source_reviewed', False), ('features_sha256', '')]:
        sample = case('one', 'match', 'development')
        sample[key] = value
        with pytest.raises(ValueError):
            validate_dataset({'cases': [sample]})


def test_fitting_does_not_read_reserved_test_features(config, tmp_path):
    path = tmp_path/'empty.json'
    content = json.dumps({'features': [{'t': i/25, 'observation_fps': 25.} for i in range(251)]}).encode()
    path.write_bytes(content)
    manifest = {'cases': [case('dev', 'match_one', 'development', 'empty.json', hashlib.sha256(content).hexdigest()),
                          case('test', 'match_two', 'test', 'DOES_NOT_EXIST.json')]}
    report = fit(manifest, {}, config, tmp_path)
    assert report['test_data_consulted'] is False
    assert report['selected_overrides'] == {}


def test_calibration_rejects_nan_and_unrelated_settings(config):
    for grid in [{'strike_fusion.min_confidence': [float('nan')]}, {'device': [0.]},
                 {'strike_fusion.not_a_runtime_setting': [.5]}]:
        with pytest.raises(ValueError):
            candidate_overrides(grid, config)


def test_missing_observations_cannot_receive_false_positive_free_credit(config, tmp_path):
    path = tmp_path/'empty.json'
    content = json.dumps({'features': []}).encode()
    path.write_bytes(content)
    manifest = {'cases': [case('dev', 'match', 'development', 'empty.json', hashlib.sha256(content).hexdigest())]}
    with pytest.raises(ValueError, match='Incomplete native-frame'):
        fit(manifest, {}, config, tmp_path)


def test_invalid_tolerance_cannot_change_calibration_objective():
    for value in [-1., float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='tolerance'):
            validate_dataset({'contact_tolerance_seconds': value,
                              'cases': [case('dev', 'match', 'development')]})


def test_adjudicated_labels_cannot_be_reported_as_reserved_match_truth():
    sample = case('one', 'match', 'development')
    sample['label_predictions_consulted'] = True
    validate_dataset({'cases': [sample]})
    for split in ['selection', 'test']:
        sample['split'] = split
        with pytest.raises(ValueError, match='adjudication'):
            validate_dataset({'cases': [sample]})
