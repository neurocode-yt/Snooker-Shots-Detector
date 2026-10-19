# Mechanical-rest red during a zoom

The red reported around 5:55 in the Robertson–Hallworth Pots + fouls draft
was absent from both saved detector timelines. It lies between the black
contacts at source 1931.68 and 1980.56 seconds. The source score changes
from 29 to 30 at 1966.01 seconds.

The mechanical rest obscures part of the white, whose observations jump
between its visible rim and nearby edges. Touching reds form a single colour
component before the stroke. The camera zoom also prevents the previous
quiet-component fallback from maintaining a continuous red track.

The default classical pipeline now has a bounded optical-flow fallback for
unresolved one-point score gaps. A measured circular red rim near a quiet
white seeds corners on the actual red pixels. Forward/backward Lucas–Kanade
flow follows those same corners without changing ball identity. Camera
registration must agree with stationary points on at least two other measured
red spheres. No neural model is involved, and ordinary detector thresholds,
sampling and existing red/depth recovery paths remain unchanged.

The proof requires at least seven quiet observations, four moving observations,
four surviving object corners, six reference corners across two spheres,
0.65 ball diameters of travel, directional consistency of 0.90 and recent
cue-contact geometry. Reference drift and object-flow spread must each stay
within 0.10 ball diameters. Native cadence, source PTS matching, camera bounds,
live context, replay, handling and scene continuity are checked. A score
change alone never supplies a contact, and partial flow evidence cannot
bypass segmentation.

The measured red departs at 1961.28 seconds and is confirmed by 1961.60 seconds.
Its estimated contact is 1961.13 with a retained uncertainty interval of
1960.93–1961.28. Normal stop, referee-entry and trim checks produce a clip
from 1959.13 to 1963.61 seconds (4.48 seconds). Source-derived score samples
confirm the one-point increase. Result policy 64 refreshes detection results
on reanalysis; the unchanged feature cache remains version 45.

Separate corrected versions contain 253 All shots clips and 165 Pots + fouls
clips (159 pots and six fouls). All prior clip ranges and scoring selections
are preserved. The corrected CapCut draft retains every saved user segment
and manual trim while inserting the red. Previous drafts and MP4s are kept.

Validation: **1,623 Python tests passed, 17 skipped; 119 targeted tests passed.**
Consecutive lossless source crops exercise the real native fallback and normal
segmentation. Negative controls cover pure camera motion, incorrect transforms,
missing reference spheres, poor white observations, broken optical tracks,
scene changes, replay, handling and incomplete proof packets. The other three
unresolved scoring intervals were checked and still lack sufficient evidence.
This repair does not establish perfect full-match recall.

Source/export checks and corrected job/draft identifiers are recorded in the
[repair report](../benchmarks/classic_rest_red_20261019.json). Restart the
running server before future analyses to load the new backend code.
