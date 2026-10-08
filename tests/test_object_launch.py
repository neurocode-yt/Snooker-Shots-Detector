"""Colour departures require independent quiet and travelling observations."""

import numpy as np
import pytest

from snooker_ai.tracking.tracker import BallTracker, Track


@pytest.mark.parametrize("motion,expected", [("launch",1), ("still",0), ("continuous",0), ("jump",0)])
def test_colour_launch_rejects_stillness_pan_and_identity_jump(motion, expected):
    positions = []
    for t in np.arange(0.,1.21,.04):
        x = 100.
        if motion == "launch":
            x += 300*max(0.,t-1.)
        elif motion == "continuous":
            x += 300*t
        elif motion == "jump" and t >= 1.:
            x += 180.
        positions.append((float(t),x,100.))
    tracker = BallTracker()
    tracker.tracks = [Track(track_id=1,label="object_ball",positions=positions,hits=len(positions),
                            diameter=80.,shape_confidence=.70,visible=True)]
    assert tracker.object_ball_launch_count(1.2) == expected
