from snooker_ai.config import load_config, deep_merge
from snooker_ai.types import EditMode


def test_default_config_loads():
    cfg = load_config()
    assert cfg.get("proxy.max_width") == 960
    modes = cfg.get("modes")
    assert "strict" in modes
    assert set(modes) == {"strict"}


def test_mode_settings():
    cfg = load_config()
    m = cfg.mode_settings(EditMode.STRICT)
    assert m["pre_roll"] == 2.0
    assert m["min_seconds_after_strike"] == 4.0
    assert m["max_seconds_after_strike"] == 60.0


def test_deep_merge():
    a = {"x": 1, "nested": {"a": 1, "b": 2}}
    b = {"nested": {"b": 3, "c": 4}}
    m = deep_merge(a, b)
    assert m["x"] == 1
    assert m["nested"]["a"] == 1
    assert m["nested"]["b"] == 3
    assert m["nested"]["c"] == 4


def test_edit_mode_coerces_legacy_names_to_strict():
    # Strict is the only mode; every legacy name and stored value must load.
    assert EditMode.from_string("strict") == EditMode.STRICT
    assert EditMode.from_string("shots_only") == EditMode.STRICT
    assert EditMode.from_string("action-only") == EditMode.STRICT
    assert EditMode.from_string("highlights") == EditMode.STRICT
    assert EditMode.from_string("full") == EditMode.STRICT
    assert EditMode("natural") == EditMode.STRICT
    assert EditMode("full_sequence") == EditMode.STRICT
