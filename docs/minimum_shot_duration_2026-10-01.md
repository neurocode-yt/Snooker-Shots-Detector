# Minimum shot viewing time

The user's latest export (`20261001-191247-f7dc27f9`, source
`software_check_2.mp4`) contains 52 shots. At output 107.167 seconds, shot 18
kept only 0.467 seconds after cue impact. Shot 22 kept 0.1 seconds. The existing
2–3-second end trim overrode the minimum post-strike hold, and cross-dissolves
shortened the clear viewing time further.

## Timing priority

1. Preserve at least four seconds of clear footage per shot, including at least
   two seconds after impact, when the source contains that footage.
2. Add context for incoming/outgoing mixes outside those four seconds. At the
   default 0.24-second transition, a short source clip targets 4.48 seconds.
   Frame quantization and both neighboring overlaps count toward this budget.
3. Apply the two-second end trim for shots under seven seconds, or three-second
   trim for longer shots, only when the viewing minimum permits it.

The physical-stop metadata stays separate from the edited endpoint. Source EOF
is a hard limit; no freeze frames are invented to fill missing time. Fast shots
give up transition padding before clear footage, and padding alone cannot
discard a neighboring strike. User-edited intervals remain authoritative.

The exporter validates the minimum duration and post-impact hold. The mix planner
protects the clear-time budget at integer and fractional frame rates. Mode and
transition-context changes invalidate final clips while preserving detection
caches, so changing this editing rule needs no new ball detection.

## Verification

- 200 tests pass. New checks cover short shots with two neighboring transitions,
  25/30/23.976/29.97 fps, source EOF, cut export, fast consecutive shots, undersized
  automatic clips, and cache invalidation for transition-context changes.
- Rebuilt all 52 shots from the actual job's cached detection evidence; cue
  strike timestamps and the shot count are unchanged.
- Every shot has at least 4.000 seconds without a transition over it. The
  shortest post-impact hold before an outgoing transition is 2.233 seconds.
- Inspected the source and rebuilt output for shots 18, 19, and 22, including
  frames 0.1, 1.1, and 2.1 seconds after impact and the subsequent mix.
- Full corrected output: 347.064 seconds, 1920 x 1080, 30 fps, with audio.
  FFmpeg fully decodes its video and audio without errors. Export took 64.85
  seconds; cached detection was reused.

Ignored local evidence: `data/editor_audit/minimum_duration/verification.json`,
`short_shots_source.jpg`, `corrected_short_shots.jpg`, and
`corrected_video/highlights.mp4`. The original export remains available.

This verifies editing duration and the reported section; it does not establish
100% detection accuracy for the entire source match.
