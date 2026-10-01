# Referee handling at the end of a shot

The previous full-match demo retained referee collection after its first real
shot. Rejecting separate preparation candidates did not fix a real shot whose
stop estimate already extended into that activity.

## Changes

- Reject circle proposals belonging to large connected foreground or long cue
  shapes. Connectivity includes the surrounding image outside the table mask;
  clipping the mask first separated glove tips from arms and made them look
  like balls. Compact red clusters remain eligible.
- Fit camera compensation to a tighter background consensus so moving people
  do not introduce false velocity into otherwise stationary ball histories.
- Retain coherent last-visible speed for occluded balls. A resting ball's
  subpixel detector jitter no longer becomes sustained movement when covered.
  A genuinely moving ball remains unresolved during occlusion.
- Count repeated movement throughout a stillness-confirmation window, including
  intermittent detections. Requiring adjacent moving samples incorrectly
  accepted a slow roll as stillness. One isolated tracking spike is still ignored.
- Invalidate older analysis/features with cache version 11. Existing exports
  are not rewritten; restart the server and reanalyze to apply the changes.

The strike-minus-two / stop-minus-two editing policy and mix renderer are unchanged.

## Evidence

- All 183 automated tests pass, including new foreground, red-cluster, occlusion,
  camera-motion, and intermittent-roll regressions. Ruff and whitespace checks pass.
- A fresh run on the supplied match's 2190–2310-second excerpt retains one
  genuine shot, detects its stop at local 17.233 seconds, and ends its clip at
  15.233 seconds. The previous demo ended at 26.167 seconds. This removes
  approximately 10.93 seconds from its tail, including referee collection.
- Visual inspection covers the source around the stop and the referee's first
  collection, not just the detected strike. The returned-red preparation
  interval still rejects the later rearrangement candidates.
- Fresh analysis of the 2480–2600-second excerpt retains all four genuine
  strikes and excludes hand placement. The intermittent-roll correction keeps
  the second shot through local 72.267 seconds instead of the initial repair's
  premature 71.267-second cut. Source endpoints were visually inspected.
- The earlier Mark Allen sample still retains five shots after a fresh run.
- The rebuilt five-shot demo is 39.72 seconds, 1920 x 1080 at 25 fps with audio.
  All four 0.24-second mix transitions were visually inspected, including the
  first transition away from the problem shot. The output skips 318.4 seconds
  between frames and fully decodes with FFmpeg without errors. Rendering took
  8.53 seconds; this measures export only, not analysis.

Local evidence is ignored by Git: `data/editor_audit/full_match/` contains
`referee_final_run.log`, `referee_previous_sample.log`, `referee_tail.jpg`,
`referee_demo_mixes.jpg`, and the updated
`frame_break_demo_referee_fixed/highlights.mp4` with export metadata.

This is a targeted regression check, not an independent accuracy benchmark or
validation of every shot in the complete match. Difficult occlusions, very slow
motion, and unusual views can still affect the estimated boundaries.
