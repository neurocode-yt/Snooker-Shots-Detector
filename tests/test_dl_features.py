from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from snooker_ai.dl.features import NeuralFeatures, merge_intervals, source_frames


class _Frame:
    def __init__(self, pts, index, time_base=Fraction(1, 1000)):
        self.pts = pts
        self.time_base = time_base
        self.image = np.full((2, 3, 3), index % 251, np.uint8)

    def to_ndarray(self, *, format):
        assert format == "rgb24"
        return self.image.copy()


def _fake_video(monkeypatch, pts, *, origin=0):
    """A keyframe-seeking decoder with explicit native source PTS."""
    frames = [_Frame(t, i) for i, t in enumerate(pts)]
    stream = SimpleNamespace(start_time=origin, time_base=Fraction(1, 1000))
    opens = []

    class Container:
        def __init__(self):
            self.streams = SimpleNamespace(video=[stream])
            self.cursor = 0
            self.seeks = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def seek(self, offset, *, stream, backward):
            assert backward is True
            self.seeks.append(offset)
            preceding = [i for i, frame in enumerate(frames)
                         if frame.pts is not None and frame.pts <= offset]
            self.cursor = (preceding[-1] // 10) * 10 if preceding else 0

        def decode(self, stream):
            yield from frames[self.cursor:]

    def open_video(path):
        container = Container()
        opens.append(container)
        return container

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_video))
    return opens


def _times(frames):
    return np.asarray([time for time, _ in frames])


def test_cfr_sampling_keeps_original_pts_and_subtracts_stream_origin(monkeypatch):
    _fake_video(monkeypatch, [7000 + i * 40 for i in range(31)], origin=7000)
    frames = list(source_frames(Path("original.mp4"), [(0, 1)], sample_fps=8))
    np.testing.assert_allclose(_times(frames), [0, .16, .28, .4, .52, .64, .76, .88, 1])
    assert frames[1][1][0, 0, 0] == 4  # actual .16 frame, not an invented .125 frame


@pytest.mark.parametrize("resume_index", range(9))
def test_vfr_resume_preserves_uninterrupted_sampling_grid(monkeypatch, resume_index):
    pts = [0, 40, 80, 132, 200, 272, 388, 404, 524, 648, 712, 836, 928, 1072]
    _fake_video(monkeypatch, pts)
    whole = list(source_frames(Path("original.mp4"), [(0, 1.2)], 8))
    expected = [0, .132, .272, .388, .524, .648, .836, .928, 1.072]
    np.testing.assert_allclose(_times(whole), expected)
    resumed = list(source_frames(Path("original.mp4"), [(0, 1.2)], 8,
                                 resume_after=whole[resume_index][0]))
    np.testing.assert_allclose(_times(resumed), _times(whole[resume_index + 1:]))
    for (_, actual_image), (_, expected_image) in zip(resumed, whole[resume_index + 1:]):
        np.testing.assert_array_equal(actual_image, expected_image)


@pytest.mark.parametrize("resume_after", [.4, .96, 1.28, 1.4, 1.64, 2.0])
def test_reviewed_interval_gaps_and_resume_do_not_decode_unknown_samples(monkeypatch,
                                                                       resume_after):
    _fake_video(monkeypatch, [i * 40 for i in range(51)])
    intervals = [(0, .49), (1.25, 1.76)]
    whole = _times(source_frames(Path("original.mp4"), intervals, 8))
    np.testing.assert_allclose(whole, [0, .16, .28, .4, 1.28, 1.4, 1.52, 1.64, 1.76])
    resumed = _times(source_frames(Path("original.mp4"), intervals, 8, resume_after))
    np.testing.assert_allclose(resumed, whole[whole > resume_after])


def test_missing_native_pts_is_rejected(monkeypatch):
    _fake_video(monkeypatch, [0, None, 80])
    with pytest.raises(RuntimeError, match="presentation timestamp"):
        list(source_frames(Path("original.mp4"), [(0, 1)], 8))


@pytest.mark.parametrize("fps", [0, -1, 60.1, float("nan")])
def test_invalid_sampling_cadence_is_rejected_before_decode(monkeypatch, fps):
    opened = _fake_video(monkeypatch, [0, 40])
    with pytest.raises(ValueError, match="sampling FPS"):
        list(source_frames(Path("original.mp4"), [(0, 1)], fps))
    assert opened == []


def test_context_merging_bounds_gaps_to_source_duration():
    intervals = [{"start": 90, "end": 95}, {"start": 4, "end": 8},
                 {"start": 14, "end": 27}, {"start": 70, "end": 72}]
    assert merge_intervals(intervals, 100, context_seconds=5) == [(0, 32), (65, 77), (85, 100)]


def test_pyav_vfr_file_sampling_and_keyframe_resume_use_decoded_pts(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "native-vfr.mp4"
    pts = [0, 40, 80, 132, 200, 272, 388, 404, 524, 648, 712, 836, 928, 1072]
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=25)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.max_b_frames = 0
        for index, timestamp in enumerate(pts):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 10, np.uint8),
                                              format="rgb24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        decoded = [float(frame.pts * frame.time_base) - origin
                   for frame in container.decode(stream)]
    whole = list(source_frames(path, [(0, 1.2)], 8))
    assert set(_times(whole)) <= set(decoded)
    np.testing.assert_allclose(_times(whole), [0, .132, .272, .388, .524, .648, .836, .928, 1.072])
    for resume_index in [0, 3, 6, 8]:
        resumed = list(source_frames(path, [(0, 1.2)], 8, whole[resume_index][0]))
        np.testing.assert_allclose(_times(resumed), _times(whole[resume_index + 1:]))
        for (_, actual_image), (_, expected_image) in zip(resumed, whole[resume_index + 1:]):
            np.testing.assert_array_equal(actual_image, expected_image)


class _StateTensor:
    """Only the tensor state serialization contract; no downloaded neural model."""
    def __init__(self, data):
        self.data = np.asarray(data, np.float32)

    def to(self, device):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class _StatefulEncoder(NeuralFeatures):
    def __init__(self):
        self.torch = SimpleNamespace(from_numpy=_StateTensor)
        self.settings = {"batch_frames": 4}
        self.spec = {"sample_fps": 8, "dimension": 4, "flow_size": [2, 3],
                     "source_time": "video_pts_zero_origin"}
        self.previous = None
        self.device = "cpu"
        self.encoded_samples = 0

    def encode_batch(self, frames):
        vectors = []
        for timestamp, image in frames:
            delta, previous_pixel = 0., 0.
            if self.previous is not None:
                previous_time, previous_image = self.previous
                if 0 < timestamp - previous_time <= 3 / self.spec["sample_fps"]:
                    delta = timestamp - previous_time
                    previous_pixel = previous_image.data.flat[0]
            vectors.append([timestamp, image.flat[0], delta, previous_pixel])
            self.previous = (timestamp, _StateTensor(image.transpose(2, 0, 1)))
            self.encoded_samples += 1
        return np.asarray(vectors, np.float32)


def _interrupted_checkpoint(monkeypatch, tmp_path):
    _fake_video(monkeypatch, [i * 40 for i in range(1001)])
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source identity")
    output = tmp_path / "interrupted.npz"
    encoder = _StatefulEncoder()

    def interrupt(progress, message):
        if encoder.encoded_samples >= 256:
            raise RuntimeError("simulated interrupted worker")

    with pytest.raises(RuntimeError, match="simulated interrupted"):
        encoder.extract(source, output, 40, progress=interrupt)
    assert not output.exists()
    chunks = list(tmp_path.glob("interrupted.chunks/*/chunk_*.npz"))
    assert len(chunks) == 1
    return source, output, chunks[0]


def test_interrupted_extraction_resumes_flow_state_without_losing_or_repeating_samples(monkeypatch,
                                                                                     tmp_path):
    source, output, checkpoint = _interrupted_checkpoint(monkeypatch, tmp_path)
    whole_encoder = _StatefulEncoder()
    whole_times, whole_features = whole_encoder.extract(source, tmp_path / "whole.npz", 40)
    resumed_encoder = _StatefulEncoder()
    messages = []
    resumed_times, resumed_features = resumed_encoder.extract(
        source, output, 40, progress=lambda fraction, message: messages.append(message))
    np.testing.assert_array_equal(resumed_times, whole_times)
    np.testing.assert_array_equal(resumed_features, whole_features)
    assert resumed_encoder.encoded_samples == len(whole_times) - 256
    assert messages[0] == "Resumed 256 learned observations"
    assert checkpoint.exists()
    cached_encoder = _StatefulEncoder()
    cached_times, cached_features = cached_encoder.extract(source, output, 40)
    assert cached_encoder.encoded_samples == 0
    np.testing.assert_array_equal(cached_times, whole_times)
    np.testing.assert_array_equal(cached_features, whole_features)


def test_partial_checkpoint_from_interrupted_atomic_write_is_ignored(monkeypatch, tmp_path):
    source, output, checkpoint = _interrupted_checkpoint(monkeypatch, tmp_path)
    (checkpoint.parent / "chunk_00001.partial.npz").write_bytes(b"incomplete zip")
    resumed_times, resumed_features = _StatefulEncoder().extract(source, output, 40)
    assert len(resumed_times) == len(resumed_features) == 321


def test_missing_committed_chunk_cannot_silently_drop_prior_observations(monkeypatch, tmp_path):
    source, output, checkpoint = _interrupted_checkpoint(monkeypatch, tmp_path)
    checkpoint.rename(checkpoint.with_name("chunk_00001.npz"))
    with pytest.raises(RuntimeError, match="missing a committed chunk"):
        _StatefulEncoder().extract(source, output, 40)


def test_motion_pair_state_is_reset_at_a_reviewed_source_gap(monkeypatch, tmp_path):
    _fake_video(monkeypatch, [i * 40 for i in range(151)])
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    timestamps, features = _StatefulEncoder().extract(
        source, tmp_path / "features.npz", 6, intervals=[(0, 1), (5, 6)])
    gap_index = np.flatnonzero(np.diff(timestamps) > 1)[0] + 1
    assert timestamps[gap_index] == 5
    assert features[gap_index, 2:].tolist() == [0, 0]
    assert features[gap_index + 1, 2] > 0


@pytest.mark.parametrize("corruption", ["nan_time", "duplicate_time", "wrong_dimension",
                                        "nan_motion_state"])
def test_corrupt_checkpoint_is_rejected_before_encoding(monkeypatch, tmp_path, corruption):
    source, output, checkpoint = _interrupted_checkpoint(monkeypatch, tmp_path)
    with np.load(checkpoint, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    if corruption == "nan_time":
        payload["timestamps"][1] = np.nan
    elif corruption == "duplicate_time":
        payload["timestamps"][1] = payload["timestamps"][0]
    elif corruption == "wrong_dimension":
        payload["features"] = payload["features"][:, :3]
    else:
        payload["previous_image"][0, 0, 0] = np.nan
    np.savez_compressed(checkpoint, **payload)
    resumed_encoder = _StatefulEncoder()
    with pytest.raises((ValueError, RuntimeError)):
        resumed_encoder.extract(source, output, 40)
    assert resumed_encoder.encoded_samples == 0


def test_matching_final_cache_with_invalid_features_is_rejected(monkeypatch, tmp_path):
    _fake_video(monkeypatch, [i * 40 for i in range(26)])
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "features.npz"
    _StatefulEncoder().extract(source, output, 1)
    with np.load(output, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["features"][0, 0] = np.nan
    np.savez_compressed(output, **payload)
    with pytest.raises((ValueError, RuntimeError)):
        _StatefulEncoder().extract(source, output, 1)


def test_source_identity_and_preprocessing_change_do_not_reuse_cached_observations(monkeypatch,
                                                                                 tmp_path):
    _fake_video(monkeypatch, [i * 40 for i in range(26)])
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "features.npz"
    _StatefulEncoder().extract(source, output, 1)
    source.write_bytes(b"different source identity")
    modified_source_encoder = _StatefulEncoder()
    modified_source_encoder.extract(source, output, 1)
    assert modified_source_encoder.encoded_samples == 9
    modified_spec_encoder = _StatefulEncoder()
    modified_spec_encoder.spec["rgb_revision"] = "different frozen backbone"
    modified_spec_encoder.extract(source, output, 1)
    assert modified_spec_encoder.encoded_samples == 9


def test_new_job_reuses_exact_full_source_neural_archive(monkeypatch, tmp_path):
    _fake_video(monkeypatch, [i * 40 for i in range(26)])
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source identity')
    cache = tmp_path / 'training_features'
    original = cache / 'reviewed_source.npz'
    expected_times, expected_values = _StatefulEncoder().extract(source, original, 1)
    reader = _StatefulEncoder()
    reader.settings['feature_cache'] = str(cache)
    output = tmp_path / 'new_job' / 'dl_features.npz'
    times, values = reader.extract(source, output, 1)
    assert reader.encoded_samples == 0
    assert output.read_bytes() == original.read_bytes()
    np.testing.assert_array_equal(times, expected_times)
    np.testing.assert_array_equal(values, expected_values)


@pytest.mark.parametrize('change', ['coverage', 'weights', 'source'])
def test_shared_neural_cache_requires_exact_coverage_weights_and_source(monkeypatch, tmp_path, change):
    _fake_video(monkeypatch, [i * 40 for i in range(26)])
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source identity')
    cache = tmp_path / 'training_features'
    _StatefulEncoder().extract(source, cache / 'reviewed_source.npz', 1,
                              intervals=[(0, .5)] if change == 'coverage' else None)
    reader = _StatefulEncoder()
    reader.settings['feature_cache'] = str(cache)
    if change == 'weights':
        reader.spec['rgb_revision'] = 'other model weights'
    if change == 'source':
        source.write_bytes(b'changed source identity')
    reader.extract(source, tmp_path / 'new_job' / 'dl_features.npz', 1)
    assert reader.encoded_samples == 9


def test_weak_clip_outside_contact_review_still_gets_feature_coverage():
    from snooker_ai.dl.features import annotation_intervals, merge_intervals
    video = {'reviewed_intervals': [], 'clips': [{'start': 903.56, 'end': 908.84}],
             'labels': [{'start': 875.8, 'end': 881.5, 'head': 'handling'}]}
    intervals = merge_intervals(annotation_intervals(video), 1230)
    assert any(start <= 903.56 and end >= 908.84 for start, end in intervals)
