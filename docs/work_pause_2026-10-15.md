# Current work and pause/resume — 15 October 2026

The library and editor share a Current work panel. It lists active, waiting,
paused, and interrupted classic match jobs, manual exports, and Shorts batches.
Each row shows the video, operation, stage/progress, and available Pause/Resume
controls. Shorts rows include completed/total clip counts and open the exact
batch in the Shorts dialog. Closing that dialog does not hide its background work.

## Pause and recovery

- Pause writes a durable request scoped to the match or specific Shorts batch.
  Pausing remains an active state until the worker acknowledges the request.
- Coarse extraction saves all streaming state at the sample where it pauses.
  Resume restores the state, with no duplicate/missing observations. Completed
  refinement windows remain reusable; an unfinished window may repeat.
- Pause propagates to contact-prefetch workers. Its control-flow exception passes
  through codec fallback handlers without being mistaken for a decoding failure.
- FFmpeg commands in managed work poll for Pause and terminate only the subprocess
  launched for that command. Unfinished encodes repeat when resumed. Existing
  completed analysis stages and Shorts clips are retained.
- Waiting matches leave the dispatch queue when paused, retaining the original
  request. Resume preserves the request's mode/resume/force/export settings.
  Startup recovery leaves paused waiting jobs untouched.
- Queued Shorts can be cancelled before their worker starts. A paused batch keeps
  its immutable snapshot, settings, and completed clips. Resume uses that snapshot
  even if the current editor selection changed while it was paused.
- Manual exports retain their complete request and resume through the match queue.
  They keep clip/joined choices, confidence filters, and output names.
- Paused state and requests survive a server restart. Work is resumed explicitly.
  Long synchronous analysis steps acknowledge Pause at their next safe check.

## Activity data and races

`GET /api/work` combines compact job status with a small index of Shorts batches
that still need attention. It avoids scanning completed batch history on every
poll. Pausing matches uses `POST /api/jobs/{job}/pause`, followed by the existing
Continue endpoint for resume. Shorts uses `POST /api/jobs/{job}/shorts/{batch}/pause`
and `/resume`. Read-only snapshot setup/preview can target an exact batch.

Status recovery re-reads job metadata while holding OS-owned leases before
declaring an active job interrupted. This prevents a stale polling read from
overwriting a concurrent completion. Queue startup completes recovery before
starting local workers, and dispatch retries brief Windows sharing errors when
removing a waiting request.

## Validation

The full suite passed: **1,358 Python tests passed, 16 skipped**. All **20
JavaScript tests** passed. After the final index-write adjustment, all **32
pause/Shorts tests** passed again. Lint, syntax, and whitespace checks passed.

Regression checks cover a real owned subprocess being stopped, queued recovery,
active match Pause/Resume through the API, exact coarse-scan equivalence after
pause, Shorts checkpoint reuse after editor changes, immediate queued-Shorts
pause, exact manual-export request reuse, stale-status completion races, and
Windows queue-file sharing retries.

Browser verification used separate jobs and cooperative test gates: a match and
a real Shorts export were paused, remained visible after closing the dialog,
and resumed from Current work. The Shorts batch retained its first completed
clip and produced both final MP4s after resume. The user server was not stopped.
