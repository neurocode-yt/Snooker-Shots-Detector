# Broadcast shot recall

`selby_lisowski_recall.json` contains complete, bounded source intervals from
`Jack Lisowski vs Mark Selby - 2026 Quarter-Final.mp4`. Times are source seconds.
The video is not distributed with the repository.

The labels were made by inspecting continuous source-image sweeps, then checking
the potential contacts at 10–25 fps. Every live target-match shot inside a window
is listed. Referee handling, replays, waiting, and preparation are negatives.
Contact bounds describe visible uncertainty; they are not invented exact impact
frames. Physical ball stops and clip viewing quality are separate checks.

The opening and return-from-break windows are **development** data: their misses
informed the detector changes. The middle, late-frame, and arena-break windows
were inspected without consulting predictions for the `2c4fa77` iteration.
The middle and late-frame failures then informed the 4 October fixes, so those
two windows are now development data. The arena-break negative interval remains
an independent check. The late-frame endpoint was
extended after checking a baseline prediction at the original boundary; its
annotation note records this limitation.

A 5000–5300-second interval originally had seven contacts frozen and hashed
before predictions were inspected. Reviewing an unmatched prediction revealed
an eighth genuine shot at 5044.84–5044.88 seconds. The original annotation had
incorrectly grouped it with earlier referee repositioning. The original frozen
file and hash remain preserved locally; the manifest records the correction.
The corrected interval is development data, and its missed soft shot subsequently
informed detection changes. It must not be described as a complete blind holdout.

The manifest now contains 49 contacts. Compare iterations against this same
manifest, while retaining the original frozen evaluation for provenance. Native
reinspection also corrected two late-frame labels to the white ball's departure,
before later object-ball motion. Contact timestamps use decoded presentation
time relative to video start. The local proof records frame indexes, since index
divided by average frame rate differs slightly on this source. Only the negative
arena-break interval remains an independent holdout for the current changes.

Run the evaluator from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_broadcast_recall `
    "path\to\analysis.json" `
    "benchmarks\selby_lisowski_recall.json" `
    --output "path\to\recall_report.json"
```

The report counts one-to-one matched contacts, missed live shots, and extra
contacts inside labelled windows only. The default contact tolerance is 0.5
seconds outside the labelled bounds. A duplicate prediction is an extra contact.
Report development and held-out windows separately when describing results.

These short sections do not establish full-match accuracy, performance on other
broadcasts, or universal coverage of camera angles. A larger output clip count
alone is not evidence of better recall or precision.
