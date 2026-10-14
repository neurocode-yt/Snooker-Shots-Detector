# Further optimization of the original detector

The default detector now does cheaper rejection checks before inspecting the
colour of large foreground components. Components that fail the existing shape
rule are rejected immediately; components too small to need the existing red
cluster test skip that calculation. Dense connected-component ids use an indexed
boolean lookup instead of a general membership search. Rejected hand/cue
proposals also skip colour scoring, and ball annulus calculations reuse the same
squared-distance image. The existing gates and returned observations are kept.

Temporal table masks contain binary 0/255 pixels. Smoothing now sums them into
one integer image and applies the same mean threshold, retaining ties in even
histories. It avoids stacking the entire history and converting it to float.
History, seek/reset behavior and saved checkpoints remain compatible.

Audio/video export verification reads FFprobe once and uses that snapshot for
both metadata and timing checks. Audio presence, duration, start alignment and
audio/video stream durations are still checked. Every verification reads fresh
media, so a replaced export cannot inherit metadata from the previous file.

Sampling rates, source windows, detection thresholds, clip boundaries, export
quality, queue modes and DL code are unchanged.

CPU measurements used real-match excerpts already available in the local audit
corpus. Baseline classes came from commit `24bf13c`; paired trials alternated
execution order. The complete measurements are in
[classic_optimization_20261015.json](../benchmarks/classic_optimization_20261015.json).

| Operation | Before | After | Validation |
| --- | ---: | ---: | --- |
| Ball detection, overhead view (50 frames) | 1.624 s | 1.579 s | All 731 detections identical |
| Ball detection, low view (50 frames) | 0.920 s | 0.871 s | All 884 detections identical |
| Ball detection, handling interval (50 frames) | 2.523 s | 2.356 s | All 562 detections identical |
| Analysis loop, overhead window (51 observations) | 12.114 s | 10.606 s | All feature fields and strike candidates identical |
| Analysis loop, low-view window (51 observations) | 4.440 s | 3.606 s | All feature fields and strike candidates identical |
| A/V timing verification (six trials) | 0.110 s | 0.056 s | Same checks passed; one probe instead of two |

Five 1280×720 table masks used about **23.0 MB → 5.5 MB** peak temporary memory
during smoothing, a **76% reduction**, excluding the input history. The mask
contents matched exactly. Eight ten-call trials also timed the calculation;
individual trials are retained in the measurement file.

These are measured windows and component costs. Low-view analysis timings had
substantial variation. Complete match processing has not been timed under this
revision, so these percentages do not predict every video's total runtime.

Validation covers even/odd mask histories, history overflow beyond 255 masks,
unchanged input arrays, incompatible dimensions, reused-probe metadata equality,
all export timing failure cases, and fresh probing after an export replacement.
The existing object, close-up and export tests also passed. Restart the server to
use the changes; saved analyses and checkpoints remain reusable.

Full regression suite: **1,322 passed, 16 skipped**. Ruff and Git whitespace
checks passed. No production jobs or source videos were modified by benchmarking.
