"""Compare independent detector runs without hiding omissions behind equal counts."""
from __future__ import annotations

import math

from scipy.optimize import linear_sum_assignment


BOUNDARIES = ("cue_strike", "clip_start", "clip_end")


def compare_shots(reference, candidate, *, tolerance=.001, match_window=1.0):
    """Maximise one-to-one contact matches, then minimise their timing error.

    The wider matching window identifies corresponding strokes for diagnosis;
    passing still requires every boundary to meet the strict tolerance.
    Neither run is treated as human-labelled ground truth.
    """
    if not 0 <= tolerance <= match_window or not math.isfinite(match_window):
        raise ValueError("Require finite 0 <= tolerance <= match_window")
    for shots in (reference, candidate):
        for shot in shots:
            if any(not math.isfinite(float(shot[key])) for key in BOUNDARIES):
                raise ValueError("Detection boundaries must be finite")
    n, m = len(reference), len(candidate)
    # A skipped stroke costs more than all possible matched errors together.
    # Dummy columns allow each desktop stroke to remain unmatched.
    penalty = (n + m + 1) * (match_window + 1)
    costs = [[abs(float(left["cue_strike"]) - float(right["cue_strike"]))
              if abs(float(left["cue_strike"]) - float(right["cue_strike"])) <= match_window
              else 3 * penalty for right in candidate] + [penalty] * n
             for left in reference]
    matched, used = [], set()
    if n:
        rows, columns = linear_sum_assignment(costs)
        for i, j in zip(rows.tolist(), columns.tolist()):
            if j >= m or costs[i][j] > match_window:
                continue
            used.add(j)
            differences = {key: float(candidate[j][key]) - float(reference[i][key]) for key in BOUNDARIES}
            matched.append({"reference_index": i, "candidate_index": j,
                            "reference": reference[i], "candidate": candidate[j],
                            "delta_seconds": differences,
                            "within_tolerance": all(abs(value) <= tolerance for value in differences.values())})
    paired = {item["reference_index"] for item in matched}
    missing = [shot for i, shot in enumerate(reference) if i not in paired]
    extra = [shot for j, shot in enumerate(candidate) if j not in used]
    maximum = {key: max((abs(item["delta_seconds"][key]) for item in matched), default=0.) for key in BOUNDARIES}
    return {"passed": not missing and not extra and all(item["within_tolerance"] for item in matched),
            "reference_count": n, "candidate_count": m, "matched_count": len(matched),
            "tolerance_seconds": tolerance, "match_window_seconds": match_window,
            "missing": missing, "extra": extra, "maximum_error_seconds": maximum,
            "matches": matched}
