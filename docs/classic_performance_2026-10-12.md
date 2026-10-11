# Default detector processing speed, 12 October 2026

The Marco Fu vs Mink Nutcharu job (`20261011-202935-4f685f09`) checked 682
coarse proposals for a 2,718.656-second match and produced 88 shot records. Its
coarse checkpoint had no supported scoreboard identity. That kept the existing
two-worker contact prefetch disabled and made every native contact check serial.
Older local servers were idle during the investigation. NVIDIA OpenCL was active.

The default detector now allows independent contact checks while scoreboard
context remains unknown. Each worker retains its own detector and context. If it
learns a target or discovers a context return, its result is discarded and the
parent repeats the window in chronological order. Once the parent learns a
target, outstanding unknown-context results cannot be consumed. Foreign,
discontinuous and inadequately covered reference windows retain serial handling.
The existing limit of two workers and two outstanding requests is unchanged.

Cue recovery also caches identical contour calculations within an image and
identical pixel/mask calculations across nearby windows. The latter cache holds
at most 512 entries and retains proposals rather than source images. Copies keep
proposal merging from modifying cached values; proposal order and multiplicity
are retained. Pixel, mask and cloth-bound changes invalidate a cached calculation.
Tiny fragments still fail the existing area requirement before cache lookup.

Foreground colour measurements and referee component/ring calculations now use
the measured component's rectangle and required halo. Image-edge clipping and
the physical entry/clearance tests remain unchanged. No source frames, candidate
windows, thresholds, stop requirements or DL code were changed. Existing analysis
caches remain compatible. No decoder-thread change was retained.

## Measurements

The machine has an i5-10300H (4 cores, 8 logical processors) and RTX 3050 Laptop
GPU. Timing trials ran without concurrent test runs. Results are saved in
[`classic_performance_20261012_results.json`](../benchmarks/classic_performance_20261012_results.json).

| Trial | Before | After | Time reduction | Output comparison |
| --- | ---: | ---: | ---: | --- |
| Three native contact windows from the slow match, serial vs two workers | 27.65 s | 15.87 s | 42.6% | All 300 observations identical |
| Same match windows, original vs optimized pixel calculations, serial | 37.31 s | 33.51 s | 10.2% | All 300 observations identical |
| Labelled referee-entry pixel sequence, five CPU repetitions | 2.75 s | 1.88 s | 31.8% | All 80 observations and entry/confirmation times identical |
| Full tail repeat on seven already capped clips from the O'Sullivan/Trump review | 74.78 s | 84.77 s | -13.4% | Both returned zero new entries |

The last trial did not improve. Those final clips already end before referee
entry, so it does not measure positive-entry recall. Whole-phase timings include
decode, geometry and context checks; a faster component calculation does not
guarantee a faster complete tail pass. Timings vary and these single samples must
not be multiplied into a promised whole-match runtime. A complete 45-minute
match has not been timed under this revision.

Decoder-thread limiting and a cache shared across changed table masks were
tested and rejected after their timings worsened. The retained implementation
uses the simpler calculation cache and guarded parallel contact checks.

## Validation and use

Regression coverage includes cache isolation/eviction, changed pixel/mask/bounds,
proposal multiplicity, unknown-context prefetch, new-target fallback, parent
context invalidation, native frame cadence and the existing source-shot/referee
cases. The full non-DL suite passed 977 tests in 128.82 seconds in this workspace: 960 relevant committed
tests and 17 local CapCut cases outside this change. Lint and diff checks passed.
Run it with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --ignore-glob=tests/test_dl_*.py
```

Restart the existing server to load the new implementation: press Ctrl+C in its
PowerShell terminal, then run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8007
```

Saved completed analyses can still resume without repeating detection. Newly
analysed matches use the updated processing path. DL remains outside this work.
