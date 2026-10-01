# Final detection logic check

The final review reproduced five failures before fixing them:

- Missing observations could count toward stationary confirmation. Gaps longer
  than 0.25 seconds now restart consecutive evidence and record uncertainty.
- Camera cuts reset ordinary stillness but left the stale-occlusion quiet timer
  running. Both timers now reset together.
- Motion samples separated by an invalid frame could count as consecutive
  launch evidence. Invalid frames now break the motion run.
- A stationary false candidate could survive as a duration-capped shot in a long
  video. Candidates without sustained motion now receive the same rejection
  reason regardless of source length.
- A failed dense refinement cleared one acceptance flag but retained an earlier
  sparse acceptance flag. Each refinement now resets that sparse flag.

The broadcast check then exposed a related adaptive-window problem: a native
stop scan could finish at the cheap tracking pass's earlier boundary, before the
native tracker had confirmed stillness. Native stop scans now continue until
their own confirmation plus the short tracking tail, bounded by the existing
safety horizon. They still start near the tentative stop, avoiding native-rate
decoding of the entire roll. A regression covers both matching and later native
stops and verifies early exit.

Cache version 7 invalidates results produced with the earlier logic. Automatic
export remains enabled; diagnostic uncertainty never requires user approval.

## Validation

All 163 tests passed. Ruff, JavaScript syntax, and Git whitespace checks passed.
The segmentation-mode fixture now uses the production travel cadence (10 fps);
2 fps proposal observations cannot establish continuous half-second stillness.

A fresh analysis of the 120.033-second sample retained all five shots and took
123.73 seconds with the existing proxy (proxy creation excluded). There were
2,309 observations. Native stop confirmation completed for the first four shots;
the fifth remains moving at EOF and is exported to the end automatically.

A separate API request resumed that saved result and rendered the combined video
in 6.12 seconds. The MP4 contains H.264 video and AAC audio at 1280 x 720, 30 fps.
Its duration is 56.867007 seconds against 56.866457 seconds requested. The final
cold analysis and checkpoint-based recomputation agreed on all five boundaries.
Endpoint contact sheets still show some stationary footage before selected
ends; continuous detector confirmation does not establish exact physical stops.

Local evidence: `data/editor_audit/automatic_summary.json`,
`automatic_rescore_summary.json`, `automatic_export_verification.json`,
`automatic_endings.jpg`, `final_check_analysis.log`, and `final_check_export.log`.

The two-minute saved broadcast contains five visually checked shots. This is a
development smoke test, not an independent labeled benchmark. Detector
confidence scores are not measured accuracy percentages. Precision, recall,
and frame-exact boundaries on unseen matches remain unmeasured; neither perfect
logic nor 100% detection accuracy is established by these checks.
