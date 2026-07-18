from snooker_ai.evaluation.metrics import evaluate_predictions, match_events


def test_match_events():
    pred = [1.0, 5.0, 9.0]
    gt = [1.1, 5.2, 12.0]
    matches, um_p, um_g = match_events(pred, gt, tolerance=0.5)
    assert len(matches) == 2
    assert len(um_p) == 1
    assert len(um_g) == 1


def test_evaluate_predictions():
    pred = [
        {"cue_strike": 1.0, "ball_motion_end": 3.0},
        {"cue_strike": 10.0, "ball_motion_end": 12.0},
    ]
    gt = [
        {"cue_strike": 1.05, "ball_motion_end": 3.2},
        {"cue_strike": 10.1, "ball_motion_end": 12.5},
        {"cue_strike": 20.0, "ball_motion_end": 22.0},
    ]
    m = evaluate_predictions(pred, gt, strike_tol=0.5)
    assert m["matched"] == 2
    assert m["ground_truth"] == 3
    assert 0 < m["recall"] < 1
    assert m["precision"] == 1.0
