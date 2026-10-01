# Full-match workflow, classic editor, and mix transitions

Automatic mode accepts an entire match and exports selected shots without an
editing/review step. The upload page also offers **Classic editor (previous
interface)**, which opens the timeline, source controls, shot list, preview,
and export controls after analysis. Existing jobs have a direct classic-editor
link. Both paths retain the current strike-minus-two / stop-minus-two policy.

## Waiting and setup footage

The coarse pass checks a small image for a compact red rack and stationary
cloth before running expensive tracking. It wakes immediately on movement,
missing observations, or loss of the rack and requests a retrospective contact
window to protect the first break-off. Native tracking is never gated this way.
Audio peaks in the interior of continuously observed waits are discarded.

A nearly cleared table followed by returned reds and a completed rack confirms
a preparation interval retrospectively. A camera cut or missing observation
breaks that inference. Movement of the white while the rack stays intact must
have cue-contact evidence; hand placement alone cannot qualify as a shot.
Cache version 10 prevents resuming results from before these checks.

These are visual heuristics. Concessions with many reds remaining, obstructed
tables, unusual cameras, and missed cue contact can remain ambiguous and use
the ordinary detector. This is not a scoreboard-based frame winner/end detector.
The coarse scan still observes the source; it does not blindly jump ten minutes
and risk skipping the next break-off. Users do not need to seek manually.

## Mix export

Combined videos use a 0.24-second cross-dissolve and matching audio crossfade.
Very short clips shorten the overlap to protect cue contact. Individual clips
remain unchanged. The renderer seeks directly to retained source intervals,
works in bounded batches, and copies the encoded batch video into the final
file. Temporary PCM audio avoids repeated AAC priming gaps; final audio is
encoded once. Timing metadata and chapters account for overlaps and output
frame quantization. CSV/EDL files remain source cut lists.

The filters follow the official [FFmpeg xfade](https://ffmpeg.org/ffmpeg-filters.html#xfade)
and [acrossfade](https://ffmpeg.org/ffmpeg-filters.html#acrossfade) interfaces.
The interactive preview shows source cuts; the finished combined MP4 includes
the mix. Set `export.transition: cut` to disable it.

## Validation and limits

- All 176 tests passed, including actual video blending, audio/no-audio outputs,
  duration checks across several render batches, preparation inference, camera
  discontinuities, and break-off recovery. Ruff and JavaScript syntax checks pass.
- A JavaScript workflow check verified both automatic export and classic-editor
  submission/redirect behavior. Existing API tests cover both choices.
- Inspected coarse overviews across the supplied 8,004.992-second Louis Heathcote
  vs Chatchapong Nasa match. Ran detailed analysis on two 120-second excerpts.
- In the 2,190–2,310-second excerpt, the last real shot is retained. Two
  referee-handling false shots are rejected by a preparation interval covering
  local 33.5–102.5 seconds.
- In the 2,480–2,600-second excerpt, hand placement at local 31.37 seconds is
  rejected, while break-off at 45.63 and shots at 66.93, 89.53, and 110.83 seconds
  are retained. Contact sheets visually corroborate these four launches.
- A demonstration combines these five detected shots directly from the original
  1080p source, skipping 307.47 seconds between frames. It rendered in 13.22
  seconds and is 57.36 seconds long, 1920 x 1080 at 25 fps, with AAC audio.
- A fresh regression run on the previous Mark Allen vs Thep sample still
  retains all five shots. Its runtime overlapped other validation and is not
  used as a comparative speed benchmark.

The entire 2-hour-13-minute match has **not** been analyzed shot-by-shot or fully
exported in this check. The demo and two excerpts are not an independent accuracy
benchmark. No claim of perfect detection, exact physical-stop labels, or 100%
accuracy is supported. Analysis and proxy generation still take time; the
13.22-second figure measures demo export only.

Local, ignored evidence lives under `data/editor_audit/full_match/`, including
`setup_verification.json`, `restart_verification.json`, `final_contacts.jpg`,
and `frame_break_demo/highlights.mp4` with its metadata and verification report.
