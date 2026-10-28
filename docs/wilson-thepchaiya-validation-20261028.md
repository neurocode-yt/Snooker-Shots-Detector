# Wilson–Thepchaiya source review, 28 October 2026

The saved Pots + fouls export omitted the opening break and first red, then
lost further contacts throughout the six-frame match. The first selected
contact was at 154.36 seconds. The source is 5590.250667 seconds, 1280 × 720,
25 fps, SHA256
`b3663c8a9c50fb87288e421eddfa4842d2a43f9936d121f68427e34d19d3c003`.

## Shared detector and selection fixes

- A dissolve from a large foreground ball to a wider low camera retained the
  previous ball scale. Major cloth-geometry changes now reset that scale at
  usable low-view confidence. Fresh native extraction finds the opening break
  at 60.60 seconds with the ordinary strike detector.
- The bridge hid the first red's stationary white from the normal tracker.
  A score-requested fallback now independently checks original pixels for a
  circular, stationary white, changed appearance at that location, cue geometry
  and coherent measured travel. It bounds this contact at 135.08–135.24 seconds.
  Scores alone, hand movement, replay, sparse samples or a later continuously
  tracked white cannot satisfy the proof. Segmentation verifies its evidence.
- Score recovery scans the entire original video, rather than relying on small
  proxy digits or windows centred on already detected contacts. Its resumable
  cache is bound to source identity, duration, reader policy and digit assets,
  and is copied to filtered versions. The final source scan retained 6246
  readable score observations at 2 fps.
- Thin antialiased sevens disappeared at the original dark-digit threshold.
  A fallback requires two brighter masks to agree while retaining the existing
  confidence and ambiguity limits. The real 28–17 overlay now survives; this
  restores the red between the previously observed 16 and 22.
- WST replay detection requires the measured central glyph/triangle and its
  full-screen wipe field. Both opening and closing graphics and preceding live
  play are required to label a replay. Source score scans retain these markers;
  both new analysis and rechecking a slim saved timeline use them. Replays do
  not end a live stroke's score window or consume its delayed foul update.
- Longer missing-overlay baselines and delayed updates are bounded by genuine
  neighbouring live strokes. A long gap may corroborate an existing stroke;
  it cannot manufacture a new one. Delayed zero overlays can search before a
  frame's displayed reset. Independently verified native openings remain frame
  context even when that camera never exposes an intact rack.

Feature cache policy is 46, result policy 68 and selection policy 8. There are
no match timestamps or player names in runtime detection rules. Existing native
short-collision recovery and its source fixtures are included with this change.

## Review and saved results

All 216 original records were checked against original-source contact sheets.
The entire source was also scanned for scores at 2 fps and replay wipes at
4 fps, followed by checks of every scoring gap, all frame bookends and finer
source images around corrections. This was systematic source sampling and
contact inspection, not continuous real-time playback.

The fingerprint-bound review is in
`benchmarks/wilson_thepchaiya_source_review_20261028.json`. It adds 35 scoring
contacts and three missing breaks, restores eleven excluded pots and the live
foul, and identifies fourteen replay records. Three replays had been selected;
the old foul belonged to the live stroke at 4004.60, not its replay at 4015.48.
A separate missed attempt had incorrectly claimed the next player's red.
The already saved live red at 3444 seconds is kept instead of its replay.
Contact intervals remain review bounds, and added clip ends are explicitly
editing decisions rather than claimed physical stops.

| Saved version | Before | After |
| --- | ---: | ---: |
| Timeline records | 216 | 254 |
| Pots + fouls selected clips | 151 | 202 |
| All shots selected clips | 216 | 240 |
| Identified frame bookends | 0 | 12 |

Pots + fouls contains 191 pots, three fouls and eight non-scoring frame-context
clips. All six openings and endings are included. No score-gap proposals remain
against the reviewed timeline. This is a source-reviewed repair, not a claim
that all 38 additions were recovered automatically by the detector.

Both saved versions were updated in place after backing up their original
analyses, timelines, metadata and exports under
`data/editor_audit/wilson_thepchaiya_20261028/original_jobs/`. The Pots + fouls
job is `20261028-082306-deebe81b`; its All shots parent is
`batch-35407c2c-3468-4ad2-9ae9-a65e45c652ac`. The existing external CapCut project
was preserved; its old 151-clip link was archived so the next Open action can
generate a project for the repaired timeline.

## Validation

Regression fixtures include original opening views, native contact features,
cropped first-red pixels, the thin-seven score overlay, and paired WST wipes.
Negative cases remove rest pixels, motion, cue geometry, departure pixels,
cadence, live context, graphic lettering and closing-wipe ownership. Additional
tests cover interrupted full-source scans, cache invalidation/copying, delayed
foul attribution across replay, and frame reset timing.

The full Python suite passes: **1859 passed, 17 skipped**. The library browser
tests pass (13 tests), and Ruff passes for the changed detector modules and new
regression tests.

Both finished exports passed complete FFmpeg audio/video decoding and
comparison of a source frame inside every selected output clip, allowing native
frame quantization. All 442 clip comparisons exceeded 0.99976 luma correlation.
Output durations match their mix plans: 1262.0 seconds for Pots + fouls and
1563.6 seconds for All shots. Machine-readable output checks are saved alongside the
review manifest in `benchmarks/wilson_thepchaiya_source_review_result_*.json`.
The user-owned PowerShell server on port 8007 was left running; it serves the
repaired saved videos immediately and loads the detector changes on restart.
