# Continued original-mode validation

This round fixes hidden-white contact timing, rejects a referee reset that was
selected as a shot, and trims sustained upright player portraits from uncertain
clip endings. The DL mode remains hidden and preserved.

## Fresh source test and subsequent development

The complete Yuan Sijun–Michael Holt interval `[600,720)` was reviewed at 1 fps,
then at consecutive native presentation timestamps around all six contacts and
the interval boundary, before detector predictions were read. The labels were
frozen with SHA-256
`07fef59be073239bb34d8d0cc4f54c3414a009d2c78648098383281203bf9c10`.
They are AI-reviewed labels, not human-approved ground truth.

The first native candidate test matched 6/6 within the 0.25-second tolerance,
but the first full pipeline matched only 5/6 and selected two unmatched events.
Its first contact was dated 2.12 seconds into the review copy rather than the
source bracket `[1.68,1.72]`, and a referee replacing the white was selected at
111.28. All six contact brackets happened to appear in that first export; this
did not make its shot selection correct.

This source group was subsequently promoted from test to development because
its failures informed changes. The contact labels remain unchanged, and the
untouched first reports and analysis snapshot are preserved in the audit
directory. Six sampled frames show identical decoded pixels between the
lossless review copy and original source, with the 600-second offset.

The corrected pipeline dates the first contact at 1.72, selects all six labelled
shots with no extras, and exports 43.16 seconds. Complete contact brackets are
covered and the entire export decodes successfully.

## Detection and ending changes

Temporal suppression now retains the first independently supported occlusion
onset instead of replacing it with a stronger residual peak later in the same
event. Measured cue-address evidence also dates its short disappearance.
For classified native footage, missing-white fallback needs cue address or
several independently measured object-ball departures. A missing white plus
movement elsewhere on the table is insufficient to establish a strike.
Legacy observations without native classification retain their existing path.

An additional ending pass requires an upright face far above a shallow,
wide foreground strip of cloth. It uses OpenCV's bundled classical boosted Haar
cascade; no neural model is run or trained. Quiet, valid, non-replay context,
continuous source timestamps, the same camera and sustained confirmation are
required. Missing cascade data is a safe no-op. A source-native backtrace dates
the first portrait frame rather than the later sampled confirmation.

The Robertson portrait starts at 153.16. The availability cap is 153.12, one
native frame earlier. This limits editing footage and preserves physical-stop
evidence and review flags. It cannot delete independently supported next shots
through overlap repair. Excluded shots, replays and user edits are preserved.

## Validation scope

All three previous native benchmark sets still match 29/29 labelled contacts
with no extras. They cover the previous six-section regression, three additional
sections, and the reserved Fu–Ng and Zhang–Un-Nooh sections. Native candidate
metrics and full-pipeline metrics overlap and must not be added as distinct shots.

Validation uses feature cache 37 and result policy 53, 929 passing original-mode
tests, Ruff and whitespace checks. The new source regressions cover the real
occluded contact, the referee negative, portrait and actual ball closeup images,
continuity/motion/replay vetoes, native timestamp failure, missing optional data,
physical-stop preservation and export contracts.

Raw source review images, frozen first-test evidence, pipeline analyses and
renders are under `data/editor_audit/continued_2026_10_11`. Tracked source labels
are in `benchmarks/classic_fresh_yuan_20261011.json`; bounded regression evidence
is under `tests/fixtures/classic_hard`.

These measurements concern the stated source-reviewed intervals. Exact physical
stop times and every pot outcome have not been independently labelled. The rest
of the 54.6-hour library and future videos remain unmeasured. Existing exports
require reanalysis to receive the updated decisions.
