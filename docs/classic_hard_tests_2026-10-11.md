# Harder original-mode validation

The original detector now recovers a green-graded white ball beside a bridge,
recognizes paired ITV red-sphere replay wipes, rejects repeated fingertip/cue
identity swaps without measured pre-contact stillness, and rechecks uncertain
shot endings against source-native table geometry. The separate DL mode stays
hidden. This work changes the original pipeline and its source benchmarks.

## Contact evidence

Thirteen completely labelled minutes from eleven match sections were freshly
extracted at native presentation timestamps with feature cache version 37.
Contact matching allows 0.25 seconds outside a source contact bracket.

| Source candidate benchmark, with original replay filtering | Matched | Missed | Extras |
| --- | ---: | ---: | ---: |
| Previous six-section regression | 14 | 0 | 0 |
| Three additional sections from the previous round | 11 | 0 | 0 |
| Fresh Fu–Ng and Zhang–Un-Nooh sections | 4 | 0 | 0 |
| Total source candidate stage | 29 | 0 | 0 |

The two fresh 90-second sections were reviewed chronologically before predictions
were read. Native consecutive images bracket both safeties in each section.
Their first test also found 4/4 with no extras. Follow-up changes were driven by
development failures; the reserved contact labels were not changed to match
predictions. These are AI-reviewed annotations, not human-approved ground truth.

The full Trump–Un-Nooh development interval `[180,300)` initially lost the red
at 273.76–273.80. The current full pipeline selects all five live shots without
an extra. Its 36.56-second render decodes completely and covers all five complete
contact brackets. Five sampled native frames prove the lossless review copy has
the same pixels as the original source, with its 180-second time offset.

The Robertson–Un-Nooh `[0,160)` case is independently run through the entire
updated original pipeline and rendering: 10/10 live contacts, no misses or
extras, and all ten complete contact brackets covered. Its 84.24-second output
decodes completely, compared with 102.96 seconds before this round. Both full
pipeline cases together select 15/15 contacts without extras. Clip boundaries are
in `benchmarks/classic_hard_tests_20261011_results.json`. Candidate-stage counts
and full-pipeline counts overlap; they must not be added together as distinct
shots or described as a whole-library accuracy result.

## Fixes and their limits

The additional ivory-mask pass separates a cooler white-ball outline from warm
skin. It retains the existing sphere, neutral-highlight and cloth-support gates.
Long native tracking exposed another false launch while the cue and fingertips
exchanged identities. The contact check now rejects a very large, inconsistent
path when neither measured pre-contact positions nor independent object-ball
motion corroborate it. The following genuine red remains detectable.

The ITV replay matcher requires a substantial enclosed rendered sphere and a
source-derived reflection pattern. Red colour or a photographed red ball alone
is insufficient. A matching opening/closing pair and preceding live play are
still required. The closing wipe does not exclude the following live black.
The three-phase 36,216-byte grayscale reference is packaged with the software;
it is a visual template, not a neural model.

End verification retries failed 640-pixel corner fitting at 960 pixels and
revalidates a previously measured surface against current cloth. Cuts, camera
motion, changed scenes and disrupted cloth coverage invalidate the geometry.
Source checks locate a native referee action at 49.48 seconds and cap its blue
clip at 49.44, before the ball is placed. A later verified glove footprint can
retain its own clear-cloth evidence when the preceding cushion fragment has
none. Prior occupancy now requires substantial footprint overlap, so a nearby
glove at the cushion cannot veto a different, previously clear location. The
cue, continuity and motion checks remain. Native verification includes more
lead-in for an unobscured table, and actual observed cloth augments fitted edges.

For unresolved ends, an additional source-native pass uses that measured table
coverage with the existing ball-stop detector. Geometry supplies no stillness
by itself. Moving or occluded balls, replay context, missing native timestamps,
sparse observations and unmeasured geometry cannot use this recovery. Cached
features are unchanged. Recovered stop times keep their review flag, and the
editor keeps the recovered boundary visible instead of subtracting early trim.
The development break now has a confirmed detector boundary at 13.16 instead
of an unresolved edit bound at 22.40; this is not independently labelled exact
physical stop ground truth.

Manual-review flags remain for uncertain timing. Contact recall and successful
decoding do not establish that every pot outcome, referee appearance or exact
ball-stop time has been independently verified. Different graphics, the rest of
the 54.6-hour library, and future videos remain unmeasured.

## Reproduction and validation

`tools/classic_validate.py` scores pinned native feature evidence with the
original replay filter. It rejects obsolete feature-cache versions and mismatched
source identities. Its output is explicitly a candidate-stage result; it does
not simulate full segmentation or rendering. Threshold calibration retains its
previous unfiltered default for comparable historical calibration reports.

```powershell
.venv\Scripts\python.exe -m pytest tests -q --ignore-glob=tests/test_dl_*.py
.venv\Scripts\python.exe tools/classic_validate.py data/editor_audit/original_2026_10_11/holdout-corpus-v37/manifest.json --output data/editor_audit/original_2026_10_11/holdout-live-v37.json
```

Validation: 907 original-mode tests pass, Ruff and whitespace checks pass, and
the wheel includes both replay reference assets. New source regressions cover
the bridge, the subsequent real red, repeated identity swaps, referee geometry,
the recovered break end, and preservation of motion/occlusion/native-timing
vetoes. Feature cache 37 and final-result policy 51 invalidate older observations
and edit decisions on new analyses. Existing exports are not automatically rebuilt.

Tracked evidence includes `benchmarks/classic_hard_holdouts_20261011.json`,
`benchmarks/classic_hard_tests_20261011_results.json`, and bounded source fixtures
under `tests/fixtures/classic_hard`. Raw audit images, frozen first-test results,
native feature manifests, full-pipeline results and renders are under
`data/editor_audit/original_2026_10_11`. The updated local preview is port 8005.
Automatic approval review blocked restarting 8003, so its process was preserved.
