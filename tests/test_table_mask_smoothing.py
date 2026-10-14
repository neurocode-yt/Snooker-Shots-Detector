"""Temporal cloth masks preserve voting, ties, and saved state."""
import numpy as np
import pytest

from snooker_ai.table_detection.localizer import TableLocalizer


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 8, 300])
def test_mask_history_matches_existing_mean_rule_without_mutation(length):
    rng = np.random.default_rng(length)
    history = [(rng.random((32, 64)) > .4).astype(np.uint8) * 255 for _ in range(length)]
    copies = [mask.copy() for mask in history]
    expected = (np.mean(np.stack(history).astype(np.float32), axis=0) > 127).astype(np.uint8) * 255
    assert np.array_equal(TableLocalizer._smooth_masks(history), expected)
    assert all(np.array_equal(mask, saved) for mask, saved in zip(history, copies))


def test_even_history_preserves_visible_cloth_on_ties():
    masks = [np.array([[0, 255, 255]], dtype=np.uint8), np.array([[0, 0, 255]], dtype=np.uint8)]
    assert TableLocalizer._smooth_masks(masks).tolist() == [[0, 255, 255]]


def test_changed_dimensions_are_rejected_instead_of_broadcasting_mask_history():
    with pytest.raises(ValueError, match="same shape"):
        TableLocalizer._smooth_masks([np.zeros((4, 8), np.uint8), np.zeros((1, 8), np.uint8)])
