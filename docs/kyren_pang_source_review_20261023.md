# Wilson–Pang saved-match correction

The 7,451.921278-second source was scanned for scores at 2 fps across all seven
frames: 14,904 samples, of which 6,757 produced readable scores. All 111 excluded
records in the previous Pots + fouls version and the 40 scoring-gap proposals
from the full source scan were inspected against source footage. Camera
cutaways, delayed updates, incorrect broadcast totals and replay contacts made
the saved automatic selection incomplete.

| Saved result | Before | Reviewed version |
| --- | ---: | ---: |
| Timeline records | 235 | 274 |
| Selected clips | 124 | 201 |
| Selected scoring clips | 124 | 188 |
| Additional non-scoring frame context | 0 | 13 |
| Opening/final frame contacts retained | 1 | 14 |

The correction adds 39 visually reviewed contacts, revises 26 existing
classifications, and retains all seven opening and final shots. Twenty-five of
the reclassified records were previously excluded. The former foul label on the
following red is corrected without removing that clip. The reviewed selection
contains 185 pots and three fouls, plus 13 non-scoring context clips. A final
scoring shot already supplied the fourteenth frame bookend.

All 124 previously selected clips keep their original source cuts. The original
source, saved analysis, MP4 and CapCut project are preserved. The corrected job
is `20261023-225250-a915e3fa`; its parent is `20261023-075154-f6e49efd`.

This is an explicit correction of this saved match. It does not change or train
the automatic detector. Added contact intervals are bounded visual observations,
up to three seconds wide; their midpoints are not claimed as exact impact
frames. Added clip ends are editing boundaries, with `stop_confirmed: false`,
rather than invented physical ball stops. Unreadable outcomes on final frame
shots retain their existing unknown classification and are included as context.

The source has a video timestamp gap before the red around 5,331 seconds.
Seeking to 5,328 returns its first decodable image at 5,330. Starting that reviewed
clip at the observed 5,330 frame avoids shifting its footage forward in export;
the cue action remains visible.

The checked correction list is
`benchmarks/kyren_pang_source_review_20261023.json`. Both the source SHA-256 and
parent analysis SHA-256 must match before the importer creates a new job:

```powershell
.\.venv\Scripts\python.exe tools/rebuild_source_review.py `
    benchmarks/kyren_pang_source_review_20261023.json --render
```

The importer copies the parent, validates reviewed intervals, rejects duplicate
contacts and replay inclusion, remaps shot/event IDs, preserves existing source
cuts, and saves a separate version. A rendering failure leaves a recoverable
child job and the parent intact.

Validation: 139 source-review, export, export API, freshness, frame-context and
selection tests passed; Ruff passed. The rendered H.264/AAC video is compared
against all 201 selected source intervals, decoded completely for audio/video
errors with FFmpeg's fail-on-error option, and checked against its frame-rounded
mix duration. It lasts 1,348.873 seconds (22:28.873); all 201 source/export frame
comparisons exceed 0.99955 luma correlation. Per-clip comparisons and the delivered file fingerprint are recorded in
`benchmarks/kyren_pang_source_review_result_20261023.json`. The local API serves
the corrected job and HTTP range downloads.
