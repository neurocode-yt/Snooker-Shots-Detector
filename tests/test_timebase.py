from snooker_ai.utils.timebase import TimeMapper, clamp, format_timestamp


def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0


def test_format_timestamp():
    assert format_timestamp(65.5).startswith("01:05")


def test_time_mapper():
    m = TimeMapper(source_duration=100.0, proxy_duration=100.0, analysis_fps=10)
    assert m.to_source(50) == 50
    assert m.time_to_frame(1.0) == 10
