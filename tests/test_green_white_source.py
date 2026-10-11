"""A measured green-cast white beside a warm bridge must retain its identity."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from snooker_ai.object_detection.detector import ObjectDetector


FIXTURES = Path(__file__).parent / 'fixtures/classic_hard'


@pytest.mark.parametrize('name,x,y', [('address', 603, 206), ('roll', 555, 232)])
def test_original_itv_source_keeps_white_instead_of_bridge_fingertip(config, name, x, y):
    # Source [273.72, 273.96] in Trump vs Un-Nooh, 2019. Native images resized
    # to analysis resolution; source hashes and contact bounds live in the
    # classic_library_more_20261010_adjudicated benchmark.
    frame = cv2.imread(str(FIXTURES / f'green-white-{name}.png'))
    assert frame is not None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    found = ObjectDetector(config).detect(frame, mask, partial_view=True)
    cues = [d for d in found if d.label == 'cue_ball']
    assert len(cues) == 1
    assert cues[0].cx == pytest.approx(x, abs=4)
    assert cues[0].cy == pytest.approx(y, abs=4)
    assert 38 <= cues[0].diameter <= 55
    assert cues[0].cue_sphere_supported


def test_cooler_outline_still_needs_a_large_neutral_highlight(config):
    frame = np.full((240, 400, 3), (35, 130, 35), np.uint8)
    colour = cv2.cvtColor(np.uint8([[[32, 120, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    cv2.circle(frame, (200, 130), 20, tuple(int(c) for c in colour), -1)
    assert not ObjectDetector(config)._warm_cue_spheres(
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), np.full((240, 400), 255, np.uint8),
        green_cast_only=True,
    )
