# Additional default-detector storage improvements

Internal coarse/dense checkpoints now omit feature values restored by model
defaults. Replay ownership remains explicit even when false, so the legacy
replay migration cannot overwrite current caller-owned markers. All nondefault
observations, including motion, contact, camera transforms and uncertainty,
remain present. Older full checkpoints still load, and signatures stay valid.

Full analysis JSON now omits indentation in both the pipeline and job store.
Every public result field is retained. Review timelines remain readable and the
JSON data is unchanged. Existing saved files are not rewritten by this change.

Using the slow Marco Fu vs Mink job's 5,438 coarse observations, three trials
in reversed orders produced these means:

| Checkpoint operation | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| File size | 11,678,513 bytes | 6,715,135 bytes | 42.5% |
| Save | 0.369 s | 0.303 s | 18.1% |
| Load | 0.440 s | 0.322 s | 26.8% |

Every restored feature value matched the original. Re-serializing the complete
46,872-observation analysis without indentation reduced its size from
129,184,071 to 99,850,312 bytes (22.7%). Two repetitions averaged 1.304 seconds
with indentation versus 0.959 seconds without it. Parsed JSON was identical.
These timings concern storage, not video feature extraction or whole matches.

The full non-DL suite passed 980 tests in 154.14 seconds: 963 relevant committed
tests and 17 local CapCut cases outside this change. The new checkpoint tests
cover defaults, arrays/transforms, explicit false replay markers, caller-owned
replay markers, inferred replay ownership and legacy migration. Lint and diff
checks passed. Measurements are in
[`classic_storage_performance_20261012.json`](../benchmarks/classic_storage_performance_20261012.json).

Per-thread detector reuse did not improve reversed-order timing trials. Larger
worker counts and smaller OpenCV thread budgets did not show a reliable gain
while a separate `python main.py` job was consuming CPU. Direct JSON validation
was also slower than the existing resume reader. None of these experiments was
retained. The existing two-worker and OpenCV settings remain in place.

Restart the server to apply this change. Previously completed videos can still
resume from their saved analyses. DL algorithms and training were not modified.
