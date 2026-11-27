# Un-Nooh vs Zak Surety source review

The original Pots + fouls edit jumped from the red at source 525.52 s to the
next selected shot at 906.52 s. During that jump, the broadcast changes camera
as Un-Nooh strikes the black at 533.88–534.00 s. The black drops in the left
corner and the score goes from 27 to 34. The detector missed the strike.
The following red attempt at 547.88 s is a live shot with no score change.
The corrected highlights include it as context and retain its no-pot outcome.

The review sampled the original footage around all 20 unresolved score updates
across the 11 frames. For each restored stroke, the manifest records the observed
contact interval, editing boundaries and score update. For the foul at 11323.72 s,
Un-Nooh plays the red and the white goes in off. Surety receives four points.
The review also removes a false detection at 577.68 s. Surety lifts his cue and
the balls stay still, then he takes the actual shot at 584.24 s.

The corrected detection has 470 contacts, up from 451. The Pots + fouls edit has
322 clips, up from 301. On both timelines, a pass over 16,009 saved scoreboard
readings found 711 stable score windows and matched each restored contact to
its corresponding score change. The pass produced zero unmatched scoring
changes. The visual review covers the 20 flagged score windows and the
reported sequence out of the three-hour source.

The report manifests bind both repairs to the original source and saved
analysis hashes. A targeted source review keeps saved frame context and remaps
frame bookend IDs when an insertion renumbers shots.

- Pots + fouls manifest: `benchmarks/zak_surety_source_review_20261127.json`
- All shots manifest: `benchmarks/zak_surety_all_source_review_20261127.json`
- The corrected CapCut draft has 322 editable clips. The restored black and red
  contacts are at 1:39.22 and 1:44.32 in that timeline. The draft opened with
  linked footage in CapCut 9.8.0.4213.
- The same contacts are at 1:35.58 and 1:40.44 in the Pots + fouls MP4. The
  export uses short transitions that change the timing relative to CapCut.

Each export matched the reviewed clip IDs in order. A frame sampled near each
contact was compared with the source, and both exports decoded from start to
finish without errors.

| Export | Clips | Duration | Lowest frame luma correlation |
| --- | ---: | ---: | ---: |
| Pots + fouls | 322 | 31:48.88 | 0.999818 |
| All shots | 470 | 51:49.80 | 0.999817 |
