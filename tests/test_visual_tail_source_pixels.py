"""Real low-angle cushions must keep calibrated referee-entry boundaries."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.scene_detection.table_context import view_geometry
from snooker_ai.segmentation.visual_tail import _TailPixels, _confirmed_entries
from snooker_ai.types import FrameFeatures


FIXTURES = Path(__file__).parent / 'fixtures/classic_hard/tail-blue'
SOURCE = json.loads((FIXTURES / 'source.json').read_text())


def source_frame(sample):
    frame = cv2.imread(str(FIXTURES / sample['file']))
    assert frame is not None
    return frame


def test_source_shallow_cushion_recovers_measured_geometry_at_native_resolution(config):
    sample = SOURCE['samples'][11]
    pixels = _TailPixels(config)
    frame = source_frame(sample)
    coarse = cv2.resize(frame, (640, 360))
    assert not view_geometry(pixels.localizer.detect(coarse), coarse.shape).full_table
    row = pixels.observe(frame, sample['t'], FrameFeatures.model_validate(sample['context']))
    assert row.full_table
    assert pixels.geometry.homography is not None
    polygon = pixels._polygon(pixels.geometry, coarse.shape[:2])
    assert .2 < np.mean(polygon > 0) < .8


def test_source_referee_boundary_survives_corner_occlusion(config):
    pixels = _TailPixels(config)
    rows = [pixels.observe(source_frame(sample), sample['t'],
                          FrameFeatures.model_validate(sample['context']))
            for sample in SOURCE['samples']]
    entries = _confirmed_entries(rows, SOURCE['strike_timestamp'], .5)
    assert len(entries) == 1
    # Contour rasterization differs slightly between CPU and OpenCL. Both
    # onsets precede the referee hand and remain within four native frames.
    assert 95.44 <= entries[0].entry_timestamp <= SOURCE['expected_entry_timestamp'] + 1e-6
    assert entries[0].confirmation_timestamp >= entries[0].entry_timestamp + .16


@pytest.mark.parametrize('change', ['camera_cut', 'camera_pan', 'other_view', 'cloth_shift'])
def test_source_geometry_cannot_be_revalidated_after_view_disruption(config, change):
    pixels = _TailPixels(config)
    sample = SOURCE['samples'][0]
    frame = source_frame(sample)
    context = FrameFeatures.model_validate(sample['context'])
    assert pixels.observe(frame, sample['t'], context).full_table
    context = context.model_copy(update={'t': sample['t'] + 1.12})
    if change == 'camera_cut':
        context.scene_cut_score = 1.
    elif change == 'camera_pan':
        context.camera_motion_magnitude = 4.
    elif change == 'other_view':
        context.view_type = 'ball_closeup'
        # Fitting this low-resolution frame fails independently of the label.
        frame = cv2.resize(source_frame(SOURCE['samples'][8]), (640, 360))
    else:
        frame = np.roll(cv2.resize(frame, (640, 360)), 150, axis=1)
    # Continuous sampling is essential: resetting by a time hole would hide
    # whether the view and pixel vetoes actually reject stale calibration.
    pixels.last_t = context.t - .08
    assert not pixels.observe(frame, context.t, context).full_table
