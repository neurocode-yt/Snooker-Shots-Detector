from snooker_ai.config import load_config, deep_merge
from snooker_ai.types import EditMode


def test_default_config_loads():
    cfg = load_config()
    assert cfg.get("proxy.max_width") == 960
    assert "action_only" in cfg.get("modes")


def test_mode_settings():
    cfg = load_config()
    m = cfg.mode_settings(EditMode.NATURAL)
    assert m["pre_roll"] >= 1.0
    assert "post_roll" in m


def test_deep_merge():
    a = {"x": 1, "nested": {"a": 1, "b": 2}}
    b = {"nested": {"b": 3, "c": 4}}
    m = deep_merge(a, b)
    assert m["x"] == 1
    assert m["nested"]["a"] == 1
    assert m["nested"]["b"] == 3
    assert m["nested"]["c"] == 4


def test_edit_mode_aliases():
    assert EditMode.from_string("action-only") == EditMode.ACTION_ONLY
    assert EditMode.from_string("highlights") == EditMode.NATURAL
    assert EditMode.from_string("full") == EditMode.FULL_SEQUENCE
    assert EditMode.from_string("strict") == EditMode.STRICT
    assert EditMode.from_string("shots_only") == EditMode.STRICT
