# Murphy–Day classical detector repair

The October 15 upload exposed three separate failures in the default detector:

- Its white scores sit beside a grey frame-count panel. The reader did not
  support that layout, so an empty score cache made almost every shot unknown.
  Zero selected shots was not evidence that no balls were potted.
- Morphological closing connected the table to a thin green arena light. The
  full table became a false close-up, preventing native stillness confirmation
  and extending unresolved clips.
- Green shading displaced a cue-ball observation onto its upper highlight.
  Some close-up contacts consequently lacked a coherent departure track.

The reader now separates the frame panel, points and active-player markers.
Twenty-one manually checked digit samples extend the existing fixed template
matcher; regression images use different source times. No neural model is used.
Both successful and empty older score caches are invalidated by selector policy 5.

Table recovery requires an independently enclosed cloth component. Real cropped
tables remain partial views. Cue recovery fits measured sphere edges and retains
cloth, shape and foreground guards. Large sharp coloured spheres can supply
close-up scale when the white is defocused; uncertain contacts remain uncertain.

Native extraction now uses decoded presentation timestamps. Stop-tail searches
retain measured launch history, and geometry revalidation starts at the strike
rather than the last 20 seconds of an already extended clip. Existing motion,
occlusion and source-frame matching requirements still apply.

Stable point changes now request additional native visual checks. A score change
alone cannot create a shot. Replay, preparation, rack and promotional footage
checks still run. Replay pairing receives all existing contacts when examining
new contacts: pairing only the new contacts erased earlier replay boundaries.
Previously rejected candidates cannot reappear solely because a recovery window
changed the nearest context row.

## Observed repair results

The source fingerprint was checked before rebuilding separate versions. Original
analysis and exports were preserved. The repair reused observed score samples
and independently decoded source windows, not invented classifications.

| Measurement | Original | Corrected |
| --- | ---: | ---: |
| Detected shot records | 258 | 269 |
| Exported live shots | 257 | 268 |
| Selected pots + fouls | 0 | 177 (172 pots, 5 fouls) |
| Unknown scoring classifications | 257 | 37 |
| Longest clip | about 40.9 s | 19.4 s |
| Clips longer than 20 s | present | 0 |

The repair contains 132 native stop corrections and 11 additional live contacts.
Both versions retain the same excluded replay record; it is not exported.
An initial set of 12 contact candidates lost one candidate at the final replay
check. The shaded-white red has a measured source contact at 888.6 seconds.

This is not a 100% recall measurement. Twenty-one score-update intervals still
lack enough visible contact evidence. The library lists those source intervals
for review, separately from the 37 unknown scoring classifications. Some hidden
or defocused strokes still need human review.

## Recovery and validation

`Recheck scores` creates a separate filtered version from saved shots without
rerunning detection or replacing the original export. Unsupported-score results
now say that the scoreboard could not be read instead of implying no pots.
Unresolved scoring intervals survive creation of additional highlight versions.

Regression coverage includes held-out scoreboard images at three resolutions,
the arena-light mask, shaded and defocused cue balls, a native missed contact,
PTS drift and invalid timestamps, stop history, referee/occlusion controls,
replay preservation and independent score-recheck exports. The complete Python
suite and browser-side tests are run before committing. The isolated browser
check verifies queued rechecking, completion and preserved previous versions.
Validation: **1,438 Python tests passed, 16 skipped; 47 JavaScript tests passed**.
Observed counts and review intervals are saved in the
[repair report](../benchmarks/classic_murphy_day_repair_20261017.json).

Restart the running server to load the updated backend before processing more
matches. Previously completed videos need an explicit recheck or reanalysis;
loading their old MP4 cannot change its contents.
