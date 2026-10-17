# Murphy–Day mini audit and CapCut handoff

The focused source test **does not establish complete shot coverage**. The
previously recovered blue is detected, but two visible red strokes still have
no accepted contact in the saved classical timeline. They remain absent from
the current Pots + fouls selection. Passing the regression suite is not a
measurement of full-match recall.

Fresh source frames were decoded at native 30 fps, using native presentation
timestamps, then processed by the existing classical feature extractor,
contact refinement and scoring-recovery guards. No neural model, learned
weights, detection thresholds or saved shot boundaries were changed.

| Source window | Frames | Source finding | Accepted contact |
| --- | ---: | --- | --- |
| 592.5–607.0 s | 436 | Visible close-up red stroke around 10:03; Murphy 14→15, Day 4 | None |
| 2757.0–2772.0 s | 451 | Visible close-up red stroke around 46:07; Murphy 52→53, Day 25 | None |
| 1294.0–1301.0 s | 211 | Previously recovered blue control | 1297.266667 s |

The two close-ups still fail native contact acceptance. A points change alone
has not been used to manufacture an accepted stroke. Further close-up tracking
work is required; this test is not a successful zero-miss audit.

Recomputing score-gap proposals against the latest 270 records yields **20**
intervals, rather than the inherited 21. The extra old warning described the
blue already recovered at 1297.266667 s. Both latest jobs' diagnostics were
updated under their processing leases. Their shot lists and exports remain
unchanged.

## Delivered output

- All shots job: `20261017-195119-223b313f`, 269 included live shots.
- Pots + fouls job: `20261017-195643-e37b5c36`, 178 included clips: 173 pots and
  5 fouls. Its existing finished MP4 is reused; unknown outcomes are excluded.
- CapCut project: **Snooker - Shaun Murphy vs Ryan Day - Pots + fouls - a910dce1**.
- The CapCut manifest has exactly the 178 selected shot IDs, in source order.
  Every source range is frame-quantized, all clips reference the existing
  source, and target ranges are contiguous. Full source handles remain editable.
- Native CapCut **9.6.0.4097** opened the project and played source video.
  Its 21:15 timeline uses individual source clips and hard cuts; the rendered
  MP4 uses the existing export transitions and has a different total duration.
- All 103 prior project-index entries were preserved. A duplicate reference
  to this new draft, caused by C:/D: junction aliases, was removed with backups
  while CapCut was closed. One project card remained after restarting CapCut.

CapCut discovery now accepts Windows 9.6 as well as 9.5 using the existing
neutral draft schema. Other versions remain guarded. Discovery preserves the
project-root spelling recorded by CapCut instead of resolving its junction,
which prevents duplicate cards for newly generated projects. Pots + fouls
project names now identify the selection.

## Validation

- 82 targeted Python tests passed; one directory-symlink test was skipped
  because Windows developer mode is unavailable. The actual moved-project
  junction was verified locally through CapCut restart and index inspection.
- 4 CapCut frontend tests passed.
- 1,098 newly decoded source frames across three windows were checked.
- This was a small targeted check, not an exhaustive match or holdout test.
- [Machine-readable results](../benchmarks/classic_mini_audit_20261018.json).

Raw native features, source contact sheets, index backups and handoff diagnostics
are local artifacts under `data/editor_audit/mini_20261018/` or alongside the
CapCut index; they are not committed to the repository.
