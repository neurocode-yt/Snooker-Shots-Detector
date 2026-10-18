# Classical analysis performance

The October 2026 refinement optimizations remove repeated calculations and
decoder setup. They preserve the classical detector's sampling cadence,
candidate recovery coverage, thresholds, tracking rules and stop confirmation.
The DL mode is unaffected.

## Changes

- Cache proxy FPS/dimensions by resolved path, size, modification time and decode
  setting. Native-source windows open the original decoder directly after the
  first proxy inspection. Proxy scans still open their own decoder; replacing
  the file invalidates the metadata. This cache holds one proxy per analyzer.
- Reuse each contact worker's analyzer and bounded pixel caches. Reset tracking,
  scene-local state and chronological broadcast context for each request.
- After the parent learns a scoreboard identity, drain obsolete speculative
  work and copy the exact new context before allowing parallel work again.
  Foreign, incomplete and returning-target intervals still use the parent.
  A context correction found inside a worker still requires serial fallback.
  Eligibility includes the canonical source-decoder warm-up and its identity
  observations, preventing a worker from dropping earlier foreign-table evidence.
- Reuse byte-identical colour-mask contour calculations within a frame, and
  reuse identical contour occlusion proof across its colour passes. Preserve
  proposal order, duplicate counts and private copies used during merging.
- Share the already measured scene thumbnail and same-frame raw classification
  with the scene stream. Preserve its independent fade look-ahead and avoid
  substituting the pipeline's full-table view override for raw classification.
- Reuse the existing table-flow percentile for the identical noise-floor
  calculation rather than calculating the same percentile twice.

Checkpoint formats and analysis signatures are unchanged. Existing compatible
coarse and native caches remain reusable. Restart the app server to load code
changes. For an active match, use **Pause**, wait for **Paused**, restart the
server, then **Resume** to retain its completed work. Running processes do not
automatically load modified Python files.

## Reproducible parity and timing checks

From PowerShell in the repository root:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_classic_optimizations.py --baseline-ref f4bc3ab --job-dir data\jobs\YOUR_JOB_ID --output data\classic-parity.json
```

Use a trusted Git revision for `--baseline-ref`. The tool loads the baseline's
classical pipeline and detector into separate modules. It compares complete
object detections on the difficult reference images and complete native frame
features on overlapping video windows. It hashes the exact serialized outputs,
reports elapsed time, checks signature compatibility, and exits unsuccessfully
if any output differs. It reads the supplied job but writes only its report and
temporary scratch directories. Repeat `--job-dir` for other matches and
`--start` for other source times. Default native windows are 73–75s and 74–76s.
Add `--parallel-contacts` to compare ordered serial evidence with the parallel
contact path, including its conservative serial fallbacks. This additionally
reads the job's coarse checkpoint and compares the detector's full contact
windows (normally four seconds plus the original decoder warm-up).

The optional `--opencv-threads` argument tests a CPU thread budget for the new
pass. It does not change the app's configuration. A two-thread experiment did
not establish a material gain for the parent, whose OpenCV thread policy remains
unchanged. The isolated process workers below use their own inner thread budget.

Validation on Shaun Murphy / Ryan Day, Michael Holt / Ian Burns and David Lilley
/ Liu Wenwei found exact equality for all six native windows (320 complete
frame-feature records) and 28 hard reference images. With the unchanged thread
policy, combined native-window timing was 29.50s before and 28.69s after in that
run (about 2.7% less time). An earlier isolated profiled window measured 13.90s
versus 10.20s, but that result should not be generalized to a full match. The
window benchmark does not measure the newly restored long-run parallel
refinement path or full export time. Timings vary with machine load and footage.

A separate two-contact serial/parallel comparison measured 19.48s versus
15.95s (about 18% less time), with all 240 frame-feature records identical. This
measures eligible contact throughput; travel/stop scans and unsafe identity
intervals remain synchronous. It is not a full-match speed guarantee.

Regression tests cover cache invalidation, unchanged native observations,
proposal ordering/multiplicity, exact context refresh, worker reuse, bounded
parallelism, serial fallback and fade look-ahead. Exact agreement on these
inputs is evidence of unchanged behavior, not a guarantee of perfect detection
on every unseen broadcast. Measure full-match runtime and accuracy on labelled
footage before claiming a universal percentage or speedup.

## Single-video CPU workers (19 October 2026)

The default `analysis.native_contact_backend: process` executes independent
contact windows in persistent spawned processes. Python contour/shape checks can
then execute on separate CPU cores. Each child keeps its decoder and bounded
pure-pixel cache and runs OpenCV with one inner CPU thread, avoiding a separate
eight-thread OpenCV team for every active window. The parent retains its existing
OpenCV policy, GPU motion path, chronological confirmation, merging and stop scans.

`analysis.native_contact_workers: auto` selects up to four children, capped by
half the logical CPU count and available physical RAM. The memory budget reserves
1.5 GiB for the parent/other work plus 768 MiB per child. Unknown RAM availability
limits the pool to two. `native_contact_workers: 1` keeps the synchronous path;
`native_contact_backend: thread` retains the previous two-thread backend.
Changing either scheduling option preserves saved detection signatures.

All source frames, masks, circle/contour gates, thresholds, timestamps, warm-up,
context eligibility and proposal order are unchanged. The thread and process
paths share the same extraction helper. Children receive the exact learned
float32 broadcast template and the complete relevant coarse context. New identity
evidence still forces parent fallback. Queue lookahead and retained results stay
bounded by the worker count. Pause requests reach children through the job's
existing pause file; a failed pool falls back to synchronous extraction.

On the user's i5-10300H (4 cores / 8 logical CPUs), RTX 3050 laptop and 24 GB RAM,
eight native windows from Robertson/Hallworth produced identical SHA-256 hashes
for all 408 complete returned frame-feature records. Two paired measurements
took 31.77 / 22.30 seconds and 36.38 / 25.88 seconds for two threads / four
processes: about 29–30% less time in this contact stage. The normal match job was
also running, and timings include child startup. These are bounded throughput
checks, not full-match runtime or ground-truth accuracy measurements. Coarse
scanning, travel/stop tracking and unsafe identity intervals remain chronological.
Full regression validation passed 1,518 tests with 17 skips; real spawn tests
also compare every feature field and exercise cross-process pause propagation.
Results: [classic_contact_workers_20261019.json](../benchmarks/classic_contact_workers_20261019.json).

To reproduce on a job with a completed coarse checkpoint, run the commands below
from the repository root. They read the saved context and source video, use
temporary worker directories and write reports outside the existing job. Hashes
for corresponding windows must match; the tool rejects windows that require
chronological parent context.

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_contact_workers data\jobs\YOUR_JOB_ID --starts 1800 2000 3000 4000 1810 2010 3010 4010 --backend thread --workers 2 --output data\contact-thread.json
.\.venv\Scripts\python.exe -m scripts.benchmark_contact_workers data\jobs\YOUR_JOB_ID --starts 1800 2000 3000 4000 1810 2010 3010 4010 --backend process --workers auto --output data\contact-process.json
```

The benchmark does not replace the active job's code. To apply this change to an
already running job, use **Pause**, wait for **Paused**, restart the same server
command, then use **Continue/Resume**. Existing coarse and completed native-window
checkpoints remain compatible; processing need not restart from the beginning.
