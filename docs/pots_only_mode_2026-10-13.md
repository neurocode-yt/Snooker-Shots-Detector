# Potted shots only

The optional `selection: "pots_only"` workflow uses the original detector and
then filters its shots with score evidence. Default automatic/classic jobs use
`selection: "all"` and do not run this pass. The DL interface remains hidden.

The reader supports paired points boxes and the older Eurosport layout with
points outside the centre frame counter. It uses classical digit templates,
shape/holes, confidence margins, repeated readings and the pre-shot turn marker.
Scoring increments for the striker confirm pots; opponent foul points, replays,
unchanged scores and ambiguous/reset/corrected scores are handled separately.
Unknown outcomes are excluded from automatic export and flagged for review.

The selector waits for score updates, bounds attribution at the next strike,
and does not claim an early update belongs to a new shot when the previous
outcome is unresolved. Partial score windows and complete windows are saved
atomically for Continue. Explicit review selections are preserved.

Changing selection creates a separate job and seeds it from a compatible saved
analysis. Original analysis, corrections, and exports are not overwritten.
The existing editor and clip boundaries stay available. A server capability
check prevents an old backend from silently treating the new selection as all
shots. Restart the server after updating to enable the new interface option.

## Validation

The full suite passed with 1,262 tests and 16 skips before the final sampling
guard. Targeted coverage includes actual broadcast pixels, multi-digit scores,
all normal ball values, opponent foul points, delayed updates, frame resets,
replays, explicit review overrides, separate-job preservation, default-workflow
isolation, score-checkpoint continuation and slower sampling.

The actual 99m50s Lilley–Liu source was processed through all 207 detected shots:
104 classified as pots, 78 as no pot, 3 as fouls, and 22 as unknown. The score
pass took 195.11 seconds on this machine. Fifteen-minute sections of Holt–Burns
(29 shots) and Zhao–Trump (36 shots) were also checked. These are classification
counts, not independently adjudicated precision/recall measurements.

A browser integration fixture exercised the workflow picker, creation of a
separate version, rendering/download, and unchanged original-analysis hash.
The fixture kept one scoring shot and excluded a foul and an unresolved shot.

This mode filters the detector's existing shots. It cannot recover a missed
strike, and unsupported/unreadable overlays require review. It does not claim
100% accuracy on arbitrary broadcasts.
