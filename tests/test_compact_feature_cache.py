"""Internal checkpoint compaction preserves defaults and replay ownership."""

import json

import pytest

from snooker_ai.pipeline.analyzer import Analyzer
from snooker_ai.types import CameraViewType, FrameFeatures


@pytest.mark.parametrize("kind", ["coarse", "dense"])
def test_compact_cache_round_trip_keeps_all_observations(config, tmp_path, kind):
    analyzer = Analyzer(config, tmp_path)
    features = [
        FrameFeatures(t=1),
        FrameFeatures(t=2, view_type=CameraViewType.REPLAY, broadcast_replay=False),
        FrameFeatures(t=3, broadcast_replay=True, replay_stinger_annotated=False),
        FrameFeatures(t=4, broadcast_replay=True, replay_stinger_annotated=True),
        FrameFeatures(t=5, observation_fps=50, camera_scene_id=17,
                      table_full_view=True, cue_ball_quality=True,
                      cue_ball_x=111, cue_ball_y=22,
                      camera_frame_transform=[1, 0, 2, 0, 1, 4],
                      ball_layout_signature=[.2, .5, .7]),
    ]
    if kind == "coarse":
        analyzer._save_coarse_cache("signature", features, [], [])
        restored = analyzer._load_coarse_cache("signature")[0]
        path = analyzer._coarse_cache_path()
    else:
        analyzer._save_dense_window("signature", 0, 1, 6, features)
        restored = analyzer._load_dense_window("signature", 0, 1, 6)
        path = analyzer._dense_cache_path(0)
    assert [f.model_dump() for f in restored] == [f.model_dump() for f in features]
    rows = json.loads(path.read_text())["features"]
    assert all("replay_stinger_annotated" in row for row in rows)
    assert "cue_ball_detected" not in rows[0]
    assert restored[1].broadcast_replay is False
    assert restored[2].broadcast_replay is True


def test_coarse_cache_still_migrates_legacy_replay_ownership(config, tmp_path):
    analyzer = Analyzer(config, tmp_path)
    features = [FrameFeatures(t=1, view_type=CameraViewType.REPLAY)]
    analyzer._save_coarse_cache("signature", features, [], [])
    path = analyzer._coarse_cache_path()
    payload = json.loads(path.read_text())
    del payload["features"][0]["replay_stinger_annotated"]
    path.write_text(json.dumps(payload))
    restored = analyzer._load_coarse_cache("signature")[0]
    assert restored[0].broadcast_replay is True
