# Make Shorts — 15 October 2026

The library and classic shot editor now expose **Make Shorts** for finished
analyses. The dialog provides player/match names, optional names per shot, three
vertical canvas presets, color/zoom/placement/text controls, and a server-rendered
preview. Each included live shot becomes its own 1080×1920 H.264/AAC MP4 using
the original analyzed source and saved trims. The original shot ID appears in
the video and filename. Audio is copied through the cut and encoded as AAC.
There is no description, voiceover generation, publishing, or external AI call.

The layout follows the other desktop editor's fit-to-canvas approach: a centered
source layer, selectable background, and centered player/shot text with outlines.
The default zoom preserves the full source frame; optional zoom can crop it.
Classic, green, and bright presets are included. This is a web implementation of
that layout, not an import of its Tkinter interface or local authentication data.

## Rendering and recovery

- The compact saved timeline supplies include/exclude decisions and boundaries.
  Replays are filtered using the same policy as existing exports. Detection,
  combined highlights, and the parent job state remain unchanged.
- Each batch captures its own immutable shot selection, source fingerprint, and
  settings. Later editor changes apply to a new batch, preserving prior outputs.
- A single background worker bounds rendering concurrency. OS leases prevent two
  processes from generating the same video's Shorts simultaneously. Closing the
  dialog does not stop rendering.
- Each MP4 is written to a partial file, probed for complete duration, dimensions,
  and expected audio, then published before its checkpoint is recorded. GPU
  encoding follows the existing configuration, with CPU fallback for NVENC failure.
- Retrying the same settings and selection reuses valid checkpointed clips;
  changed or missing clips render again. Completed MP4s are zipped without a
  second compression pass. Archives contain only the selected finished videos.
- Stale active manifests become resumable after their OS lease is free. Status
  re-reads the manifest while holding that lease so a concurrent completion cannot
  be overwritten as interrupted. Manifest reads retry brief Windows sharing errors.
- Source changes prevent unsafe reuse. Errors retain completed clips and expose
  a Resume action. Graceful shutdown finishes the current clip and saves the batch
  as interrupted before leaving queued work for an explicit resume.

## API

- `GET /api/jobs/{job}/shorts`: presets, current included shots, and latest batch.
- `POST /api/jobs/{job}/shorts/preview`: `{settings, shot_id}` → PNG layout preview.
- `POST /api/jobs/{job}/shorts`: visual settings plus `player_names` and optional
  `shot_names` → queued/reused batch. The web client sends resolved preset values.
- `GET /api/jobs/{job}/shorts/{batch}`: durable progress and finished clip list.
- `GET /api/jobs/{job}/shorts/{batch}/clips/{shot}`: download one finished MP4.
- `GET /api/jobs/{job}/shorts/{batch}/zip`: download a completed batch archive.
- `POST /api/jobs/{job}/shorts/{batch}/open-folder`: open its local output folder.

Output location: `<jobs_dir>/<job>/export/shorts/<batch>/`.

## Validation

The complete Python suite passed: **1,349 passed, 16 skipped**, including 23
Shorts tests. All **18 JavaScript tests** passed. Python lint, JavaScript syntax,
and Git whitespace checks passed.

Focused tests cover included/replay selection, unsafe names and invalid settings,
bounded layout, immutable snapshots, duplicate handoffs, cache invalidation,
interrupted resume, the status/completion race, source changes, API/download
behavior, and real FFmpeg rendering both with and without source audio.

Browser verification used a separate job store and real rendering: two included
six-second shots produced the expected 1080×1920 files and player titles; the
excluded shot was absent. Closing the library dialog and reopening it in the
editor restored progress/settings and download links. The downloaded ZIP passed
its integrity check and contained exactly those two MP4s. A frame extracted from
the actual output confirmed its text, background, and source placement.
