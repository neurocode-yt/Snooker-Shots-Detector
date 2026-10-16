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
not establish a material gain, so the production OpenCV thread policy remains
unchanged.

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
