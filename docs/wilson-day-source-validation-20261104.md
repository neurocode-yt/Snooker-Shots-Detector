# Wilson-Day source review — 4 November 2026

The latest supplied Kyren Wilson vs Ryan Day source runs 7,604.786667 seconds
(1280 x 720, approximately 25 fps). Its saved Pots + fouls version selected two
of 298 records, with 294 unknown outcomes.

The white points were too dim for the existing bright-digit mask. In addition,
the yellow frame-count panel bled into adjacent digit crops, and the turn-dot
lookup checked beside the points instead of the outer ends of the name panels.
The fix masks saturated panel/break-counter pixels, requires agreement between
independent lower-threshold masks, and keeps the existing glyph confidence and
ambiguity checks. Full-source score observations increased from 531 to 14,689.
Score/cache and result policy versions invalidate the obsolete observations;
old library selections offer Recheck scores.

## Review scope and outcome

Reviewed source contact sheets for all 298 original records, 25 long gap windows,
139 shorter gap windows, all scoring-gap proposals, and every frame opening and
ending. Suspect windows received one-second/finer inspection and cue-ball position
comparisons to distinguish strokes from addressing, measuring and referee work.
This was sampled visual review, not continuous real-time playback.

The corrected All shots version contains **286 live contacts**: 24 missed contacts
restored, eight early timestamps replaced, and 36 false preparation/referee records
removed. The manifest expresses the replacements as 32 additions and 44 removals.
The corrected Pots + fouls version selects **208 clips**: 193 pots, three fouls,
and 12 additional opening/final-stroke context clips. All seven frames retain their
break and final stroke. The first red at 154.84 seconds is selected.

Reviewed clips stop before the next contact's preparation, so restoring a missing
shot does not duplicate it inside the previous clip. Frame-ending cuts omit the
subsequent re-racking. Reviewed intervals describe contact uncertainty and editing
boundaries; they do not claim a newly measured physical ball stop. Original motion
observations are retained when an existing clip is trimmed.

Source-reviewed frame bookends survive score rechecks, with complete, ordered,
nonoverlapping pairs required before skipping automatic rack recovery. Review
imports now support explicit false-contact removals, excluded safety additions,
and bounded replacement cuts, while remapping events and replay links.

## Artifacts and reproducibility

- Source SHA-256: `8233f54d87e238f5f0c9570c8a521108c589f18e308c2a1f23b9ff5550bc934f`.
- Fingerprint-bound review: `benchmarks/wilson_day_source_review_20261104.json`.
- Parent All shots: `batch-f3af8e95-1626-488c-82c5-e78a6d5bd46e`.
- Pots + fouls: `20261104-085750-cc9d0ac5`.
- Original analyses, metadata, score cache and exports are backed up locally under
  `data/editor_audit/wilson_day_20261104/backup_before_repair/`.
- Twelve original score-bar crops and a real 22→29 black-pot regression live under
  `tests/fixtures/wilson_day_scores/` and `tests/test_wilson_day_scores.py`.

`tools/rebuild_source_review.py` can apply the manifest to the original parent and
create a separate reviewed version. It checks both the source and original
analysis SHA-256 before applying changes. The current library versions are
repaired in place only after their replacement videos pass validation.

## Validation

Full suite before the final import-boundary additions: **1,881 passed, 17 skipped**.
Final focused score, frame-context, source-cache and review-import suite:
**154 passed**. Ruff passes for the changed Python files.

`tools/validate_source_review_export.py` checks exact selected clip IDs, compares a
source frame inside every rendered clip, checks output/mix duration, calculates
the video SHA-256 and fully decodes audio/video with FFmpeg. Per-version reports
are stored as `benchmarks/wilson_day_source_review_result_<job-id>.json`.

| Version | Compared clips | Output/mix duration | Minimum luma correlation | Full decode exit |
| --- | ---: | ---: | ---: | ---: |
| Pots + fouls | 208 | 2801.68 s | 0.99989607 | 0 |
| All shots | 286 | 3660.36 s | 0.99989574 | 0 |

Both complete audio/video decodes produced no errors. No unmatched scoring-gap
proposals remain against the reviewed contacts. The published MP4 hashes match
the validated replacements.
