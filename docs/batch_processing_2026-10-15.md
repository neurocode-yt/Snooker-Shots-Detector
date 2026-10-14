# Multiple-match processing

The upload page now accepts multiple files from the picker or drop zone. Each
video is uploaded and submitted separately with the chosen highlights mode.
The **Processing order** control offers **One at a time (queue)** and **Parallel**.
In queue mode, **Add videos to queue** submits waiting matches and automatically
starts each after the previous analysis and automatic export finish. More videos
can be added while it works. The order persists in the jobs directory across
server restarts. Changing to sequential leaves active jobs running and starts
the next waiting job only after all previously parallel work has finished.
Two entire analysis/export jobs run concurrently by default; further jobs wait
in submission order. The form becomes available once submission finishes, so
more matches can be added while existing ones process.

`jobs.max_workers` in `configs/default.yaml` controls the limit (1–4). Worker-slot
leases share that limit across servers configured identically against the same
jobs directory. Each queued job has its own ownership lease; queued jobs cannot
be edited, exported manually, restarted or submitted twice. `GET /api/queue`
reports the current server's capacity and counts, and job metadata includes
`queued` and a local `queue_position`.
`PUT /api/queue` accepts `processing_order: sequential | parallel`; responses
include `processing_order` and the effective `concurrency_limit`. A shared
dispatch lease checks all occupied worker slots before admitting another job,
including when a server changes the limit while matches are already active.

Waiting requests are stored atomically in each job's `queued.json`. Startup
recovers pending requests with their original source, resume/force settings and
export choice. Active jobs use the existing durable detector checkpoints and
Continue workflow after an interruption. Failed jobs release their slots.
Malformed waiting requests offer Continue instead of remaining stuck in the
queue. Shutdown allows a short drain, then preserves waiting requests.

Upload/submission errors are isolated per file. Retry keeps the original settings
and trims, skips accepted jobs, and reuses successfully uploaded/trimmed sources.
A stable submission token prevents a lost HTTP response from starting a second
copy of the same request. Batch uploads analyze full matches; single-file source
trimming remains available. The default detector and hidden DL interface are
preserved.

Validation on Windows:

- Python suite: **1,302 passed, 16 skipped**. Browser presentation/submission
  tests: **12 passed**. JavaScript syntax, Ruff and Git whitespace checks passed.
- Controlled worker checks proved two simultaneous calls, waiting positions,
  failure releasing a slot, duplicate exclusion, shared limits across two server
  instances, persisted queue recovery, and interruption/Continue behavior.
- The browser submitted three separate H.264 fixtures, displayed two processing
  plus one waiting, accepted further file selection, preserved single-file trim
  controls, and produced three downloadable exports. Test data and server were
  isolated from existing user jobs; the temporary server was stopped afterward.
- Two real-match excerpts of 45 seconds each were analyzed separately and then
  through two concurrent classical workers. Both retained two shots, with
  exactly matching cue times, clip boundaries, inclusion and replay flags.
  Observation counts also matched (1,056 and 901). One excerpt's intermediate
  camera/motion measurements differed; the other's feature rows matched exactly.
  Bitwise feature reproducibility across executions is therefore not asserted.

This validation covers queue behavior and those specific excerpts. It does not
establish perfect detection on unseen videos or a throughput improvement on
every laptop. Simultaneous matches share CPU, GPU, memory and disk bandwidth.

Sequential-mode follow-up validation:

- Full Python suite: **1,306 passed, 16 skipped**; **12** JavaScript presentation
  and submission checks passed. Ruff and JavaScript syntax checks passed.
- Tests cover FIFO starts after export, switching from two active videos to one
  at a time, persisted settings, API validation, and a shared sequential limit
  across two servers configured with two available slots.
- The browser queued three videos, displayed one processing and two waiting,
  and accepted a fourth while the first was active. When the first finished its
  actual fixture export, the second started automatically; the third and fourth
  retained their waiting order. Reload preserved the selected processing order.
  All four fixtures eventually produced separate downloadable exports.
