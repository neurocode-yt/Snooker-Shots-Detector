# Williams–Gilbert saved-match correction

The complete 11,913.048333-second source was scanned for scores at 2 fps
(23,827 samples). The review compared all 354 saved shot records with original
source frames, examined all 79 uncovered score changes, and inspected combined
scoring runs and the opening/final contacts of all ten frames. Focused samples
bounded the restored contacts. This was a sampled source review, not continuous
real-time playback of the three-hour match.

The automatic result omitted live contacts during changes of camera angle,
excluded valid pots when scores appeared late, and treated broadcast replays as
new contacts. Larger score changes sometimes covered several live pots. For
example, frame 9's first red was already present as record 294 but excluded;
the next blue's evidence combined both pots. It is restored without adding a
duplicate. Record 275 was a replay of the preceding black, incorrectly selected
as the next red; the actual subsequent live red is restored instead.

| Saved result | Before | Reviewed version |
| --- | ---: | ---: |
| Timeline records | 354 | 441 |
| Selected clips | 200 | 316 |
| Selected pots | 197 | 293 |
| Selected fouls | 3 | 4 |
| Non-scoring frame context | 0 | 19 |
| Frame bookends retained | 1 | 20 |
| Confirmed replay records marked | 0 | 36 |

The correction adds 87 source-reviewed contacts (85 pots, one foul, and the
missing seventh-frame break), restores 30 excluded records (12 pots and 18
frame bookends), and excludes one previously selected replay. It includes the
opening break and first live red. All 199 retained original clips preserve
their original cuts. The original source, parent analysis, export, and existing
CapCut data are preserved.

The reviewed job is `20261025-053135-6bcc3add`; its parent is
`20261025-021055-bde48e86`. Corrections and source/parent SHA-256 fingerprints
are recorded in `benchmarks/williams_gilbert_source_review_20261025.json`.
Rebuilding requires matching fingerprints:

```powershell
.\.venv\Scripts\python.exe tools/rebuild_source_review.py `
    benchmarks/williams_gilbert_source_review_20261025.json --render
```

The source-review importer now permits explicit replay classification and
named additions used as frame bookends. Non-scoring additions must be
referenced by a reviewed frame; included replays, invalid/duplicate references,
overlapping frames, duplicate contacts, and unbounded intervals are rejected.
The parent is copied before applying corrections, and shot/event references
are remapped after inserting contacts.

This repairs this saved match. It does not train or change the automatic
detector. Contact intervals are visual sample bounds, up to three seconds
wide, not frame-exact cue instants. Added clip ends are editing boundaries
(`stop_confirmed: false`), not invented physical ball stops.

Validation: 44 source-review, export, and freshness tests passed; Ruff passed.
The rendered H.264/AAC file lasts 2,149.080 seconds (35:49.080), matching the
frame-rounded mix plan. FFmpeg decoded the complete audio/video with
`-xerror`, exit code 0, and no errors. All 316 selected clips were compared
with their source positions; the lowest luma correlation was 0.999580.
The local API serves the completed job and HTTP range downloads. The
file fingerprint and per-clip source/output comparisons are recorded in
`benchmarks/williams_gilbert_source_review_result_20261025.json`.
