# Score-based highlights

The optional `selection: "pots_only"` workflow uses the original detector and
then filters its shots with score evidence. Default automatic/classic jobs use
`selection: "all"` and do not run this pass. The DL interface remains hidden.

The reader supports paired points boxes and the older Eurosport layout with
points outside the centre frame counter. It uses classical digit templates,
shape/holes, confidence margins, repeated readings and the pre-shot turn marker.
Scoring increments for the striker confirm pots; opponent foul points, replays,
unchanged scores and ambiguous/reset/corrected scores are handled separately.
Unknown outcomes are excluded from automatic export and flagged for review.

`selection: "pots_and_fouls"` keeps pots, confirmed opponent foul points and
confirmed point increases whose pot/foul attribution is uncertain. These last
outcomes are labelled `points`, without claiming a pot or foul. Corrected scores,
resets, unreadable overlays and updates which cannot be assigned to a stroke
are still excluded and flagged. The original pots-only mode continues to exclude
fouls. Both use the classical reader and the existing detector.

The selector waits for score updates, bounds attribution at the next strike,
and does not claim an early update belongs to a new shot when the previous
outcome is unresolved. Partial score windows and complete windows are saved
atomically for Continue. Explicit review selections are preserved.

Changing selection creates a separate job and seeds it from a compatible saved
analysis. Original analysis, corrections, and exports are not overwritten.
The existing editor and clip boundaries stay available. A server capability
check prevents an old backend from silently treating the new selection as all
shots. Restart the server after updating to enable the new interface option.

The one-click **Pots + fouls highlights** action in Recent jobs and the editor
calls `POST /api/jobs/{job_id}/points-highlights`. It loads the compact timeline,
skips detector creation and reruns only score confirmation and selected-clip
export. Saved score windows are copied and validated against source, settings,
templates and ranges before reuse. Changes to detector settings do not cause
this action to rescan an already completed match. Source changes are rejected
instead of pairing old detections with a different video. A parent edit produces
a new version; repeat clicks share the existing running or completed version.
Failed/interrupted variants and Continue reuse the lightweight saved timeline.
Explicit review inclusion/exclusion choices are retained during conversion.

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

The pots + fouls follow-up passed the full suite with 1,281 tests and 16 skips.
Coverage includes penalties of 4–7 points, stable increases without a turn
marker, corrected foul points, legacy jobs, source replacement, compact-timeline
reuse despite changed detector settings, repeat clicks, review choices and
export retries. Its browser fixture rendered an 11.73-second H.264 export with
one pot and one foul, excluded the unresolved shot and reused the finished
version on a second click. Original analysis, timeline, metadata and export
hashes remained unchanged by conversion.

With previously validated score checkpoints, the full 99m50s Lilley–Liu timeline
loaded in 0.066 seconds and score confirmation took 1.075 seconds without opening
a video decoder: 104 pots and 3 fouls were selected from 207 detected shots, with
22 unresolved outcomes flagged. This measures cached selection time; first-time
score reading and selected-clip encoding still take time. The counts are not
independent accuracy measurements.

This mode filters the detector's existing shots. It cannot recover a missed
strike, and unsupported/unreadable overlays require review. It does not claim
100% accuracy on arbitrary broadcasts.
