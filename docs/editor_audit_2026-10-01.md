# Snooker editor audit — 1 October 2026

The audit found and repaired reproducible crashes, stale analysis, export freezes,
incorrect handling of review edits, and avoidable processing work. The application
passes 143 tests. Automatic editing is **not yet 100% accurate**: the real broadcast
sample retains all five observed shots, but its stop boundaries still need review.

## Repairs

| Problem | Repair and verification |
| --- | --- |
| Latest failed job reports `object of type 'NoneType' has no len()` | Reproduced this exact exception with a blank frame on this machine's OpenCL backend. Empty GPU feature results now become unknown camera observations instead of crashing. A regression covers empty `UMat` results; 43 observations near the failed job's last checkpoint also processed successfully. |
| Cached results ignore changed detection settings | Removed five duplicate methods that silently replaced the configuration-aware cache implementation. Feature caches now fingerprint source identity, modification time, configuration, and cache version. Completed results additionally fingerprint segmentation settings. Old results are rebuilt while manual corrections are retained. |
| Proxy/audio can belong to an old source | Added a source/configuration manifest. Replacing a video under the same filename or changing audio sampling settings regenerates the affected media. CPU encoding fallback is recorded correctly and its valid proxy is reused. |
| Audio recovery accepts irrelevant movement | Recovery now requires valid table observations and ball-specific movement with quiet pre-strike context. Player flow, graphics, replay views, camera cuts, and continuing motion cannot independently validate an audio peak. Confirmed visual strikes remain eligible when audio recovery is ambiguous. |
| Whole audio recording scanned for every frame | Replaced repeated full-array scans with timestamp searches over small local windows. Nearest-sample ties and peak results retain their prior behavior. Peak/candidate spacing checks also search neighboring timestamps. |
| Recovered shots analyzed twice | Complete native-rate observations are reused during audio recovery. A missing interval still triggers fresh analysis. |
| Low overall table flow cuts a rolling ball | Global quietness cannot override independently observed ball movement. Regression covers a small moving ball and intermittent zero flow from duplicate broadcast frames. |
| Web interface freezes during export | FFmpeg export runs in a worker thread. Progress remains available; simultaneous processing/export requests for the same job are rejected. |
| Manual edits, merges, and splits fail export validation | Review timestamp aliases are synchronized when saving and loading, including older saved corrections. Splitting before a strike keeps the resulting intervals valid. |
| Deleted false detections return after re-analysis | Deleted strike timestamps are preserved in corrections. Other edits retain those deletions; explicitly adding the shot back restores it. Fresh automatic replay decisions are no longer overwritten by old automatic include decisions. |
| Decode resources remain open after an analysis exception | Feature extraction closes the decode iterator and releases the capture in `finally`. |

## Verification

- Before repairs: **108 tests passed**, despite the reproduced runtime crash.
- After repairs: **143 tests passed** in 10.23 seconds.
- Ruff passed for the changed Python modules and regression tests.
- Existing unrelated working-tree changes were preserved.
- Real input: the first 120.033 seconds of the uploaded Mark Allen versus
  Thepchaiya Un-Nooh broadcast. Frame sheets were inspected before assessing shot count.
- Five real shots were observed; final analysis retained five, near 4.633,
  36.900, 61.700, 85.267, and 110.000 seconds.
- Combined export succeeded: **45.000 seconds**, with audio, matching the total
  selected clip duration. This checks media generation and duration; it does not
  establish frame-perfect shot timing or perceptual audio synchronization.

Local evidence is saved in `data/editor_audit/`: analysis JSON, frame sheets,
benchmark summaries, the crash probe, and `export/highlights.mp4`. Audit media
is excluded from Git.

## Measured performance

| Measurement | Before | After |
| --- | ---: | ---: |
| 400 audio observations against a four-hour timeline (450,000 audio hops) | 4.540 s | 0.0099 s |
| Re-analysis of the 120-second sample using valid feature caches | 47.093 s | 31.715 s |
| Resume an unchanged completed sample | — | 0.089 s |
| Export its five selected shots as one video | — | 5.375 s |

The audio benchmark returned identical values and measures lookup work only.
The re-analysis comparison measures the native-observation reuse change. These
are single runs on this machine, with no claim of equivalent gains for every match.
An earlier patched cold analysis took 188.462 seconds. The original 710.376-second
run used profiling, so it is not a fair end-to-end speed comparison.

## Remaining accuracy limits

All five final sample endings remain flagged for review and use the configured
seven-second post-strike cap. False ball tracks still prevent reliable stationary
confirmation; those capped boundaries are estimates, not observed physical stops.
Strikes can also be estimated several frames early. The sample therefore demonstrates
shot retention and successful export, **not perfect automatic editing**.

The installed object detector is a heuristic blob/circle implementation; its learned
model runtime is a scaffold. No independent human-labeled match benchmark was found
in the repository, so this audit does not claim measured broadcast precision or recall.
Reliable accuracy assessment needs reviewed strike/stop labels across complete matches,
including slow balls, occlusions, referee handling, replays, and non-play footage.

Restart the editor server to load the repairs, then re-analyze affected jobs. Legacy
analysis caches and proxies without valid fingerprints are regenerated once. Saved
manual corrections are preserved.
