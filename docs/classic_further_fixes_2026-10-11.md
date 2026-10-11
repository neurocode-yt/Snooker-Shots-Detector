# Further default-detector fixes

These changes use the original rule-based pipeline. They add no neural model or
training operation. The separate hidden mode is preserved.

## Fresh source failures

Two complete intervals `[600,720)` from Lisowski–Selby and O'Sullivan–Trump
were reviewed chronologically at one-second cadence, followed by consecutive
native 25 fps images around all ten contacts and interval boundaries. The
source labels were frozen before predictions were read. They are AI-reviewed,
not human-approved ground truth.

The first full pipeline found 3/3 Lisowski contacts. O'Sullivan initially had
only 3/7 live contacts, plus a replay selected at 16.2 seconds and a referee
reset selected at 57.56 seconds in its review copy. Blue, two reds and black
were missed. That source was promoted to development when its failures informed
the changes; the source contact labels and first-test evidence were preserved.

## Changes

A brightness split separates an enclosed bright playing surface from darker
green advertising above it. Geometry and spatial separation must independently
support the split. Uniform dark cloth and clipped local views retain their
original mask. This removes the referee's head and billboard lettering from
the playing-surface search area.

An additional shaded-sphere pass handles the strong green lighting on the white
ball in close cameras. It retains measured outline, highlight and cloth-support
requirements. Broader recovery cannot displace an already measured ordinary
ivory sphere toward a fingertip; the existing source regression caught and
guards that interaction.

Replay matching now measures the full-height gold chevron and dark teal field.
Two independently recognised opening/closing wipes plus preceding live play
are required. Ordinary cushions and a single graphic do not establish replay,
and the following live red remains available.

The final red was visible but rejected because unrelated object-ball tracks
were noisy. Native measured white-ball rest, a recent cue approach, sustained
bounded departure and consistent view/timestamps can now independently confirm
a large close-up contact. Overlap repair preserves that confirmed stroke rather
than deleting it because the other-ball quiet ratio is poor. Missing cue proof,
invalid observations and replay context stay rejected.

Unresolved long tails are sampled from just after the strike on the proxy.
Otherwise an early referee entry can lie outside the former final-20-second
search. Native verification still decides the boundary. The black's availability
cap is 75.60 seconds in the review copy, one frame before verified entry at
75.64; it avoids retaining the later reset and waiting period. This is an editing
availability limit, not a claim that every ball physically stopped at that time.

## Validation

| Full default pipeline | Initial matched / missed / extras | Final matched / missed / extras | Render |
| --- | --- | --- | --- |
| Lisowski–Selby, 120 seconds | 3 / 0 / 0 | 3 / 0 / 0 | 21.92 seconds |
| O'Sullivan–Trump, 120 seconds | 3 / 4 / 2 | 7 / 0 / 0 | 68.64 seconds |

Contact matching allows 0.25 seconds outside a frozen native contact bracket.
Every complete bracket is covered by an included export clip and both joined
exports decode successfully. Eleven sampled original/copy frame pairs have
identical decoded pixels. The O'Sullivan crop has an objective 40 ms offset due
to its nonzero stream start timestamps: copy time maps to original OpenCV source
time plus 599.96 seconds, compared with 600 seconds for Lisowski. Evaluation
uses that measured mapping without changing the source labels.

The default detector and existing shared tests account for 952 passing cases.
Seventeen separate CapCut workspace integration cases also passed, yielding
969 passes in the local run; that unrelated pending work is not included in
this change. DL test files were excluded. Ruff lint and whitespace checks pass.
FFmpeg fixture generation now uses the application's compatible-binary resolver,
so a stale PATH executable cannot break the export checks.

Feature cache 38 and final-result policy 59 invalidate older observations and
results for new analyses. Existing rendered exports need reanalysis to receive
these fixes. The labels, output metrics and bounded source fixtures are tracked;
raw source review images, first failures, analyses and renders remain under
`data/editor_audit/further_2026_10_11`.

The full default test command is:

```powershell
.venv\Scripts\python.exe -m pytest tests -q --ignore-glob=tests/test_dl_*.py
```

The current contact results concern these two intervals. Exact physical stop
times, every pot outcome and the remaining library have not been independently
labelled. Timing review flags remain, and these results do not certify universal
100% accuracy on future footage.
