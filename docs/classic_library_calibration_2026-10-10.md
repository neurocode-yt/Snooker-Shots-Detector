# Original-mode match-library calibration

The original detector now retains its ball-size estimate when a referee hides a
main-camera table corner, rejects apparent cue movement during camera dissolves,
and recovers a shot whose white ball briefly disappears into the red pack. The
short proposal-window recovery also survives the pipeline's separate native
confirmation pass. DL remains hidden; no DL training, dependencies, weights or UI
controls were changed by this work.

The source library was `G:\Adobe Premiere Pro Auto-Save\Snooker yt shorts\nWindows`.
All 51 videos were recursively inventoried and readable: 54.64 hours in total,
including 24 match-file candidates, edited copies, short clips and advertisements.
Filename-based grouping and split suggestions are provisional. All 51 received
sparse overview sheets; this is not full chronological annotation of 54.64 hours.

Six complete one-minute sections were reviewed chronologically at 1 fps and their
contacts checked against original native frames. Development used Bai–Graham,
Williams–Higginson and Evans–Nutcharut; rule selection used Jiang–Un-Nooh. Two
reserved match sections were Wilson–Un-Nooh and Tirapongpaiboon–Norris. Underlying
match groups do not cross these splits. These are Codex-reviewed labels, not
human-approved ground truth. The initial 13-contact labels are retained separately:
a true green at the end of the Evans section was subsequently adjudicated from
native source frames, bringing the reference to 14 contacts. That adjudication is
explicitly marked as development work; reserved labels were frozen before any
predictions were checked.

An 18-combination sweep of existing strike thresholds failed to recover the two
missing Jiang reds. Changing rules and tracking fixed the measured failures;
the final numeric settings remain the existing defaults. The fitter reads only
development and selection feature cases, verifies their hashes and native-frame
coverage, and writes a candidate configuration without activating it in the app.
It rejects match-group leakage, unreviewed labels, and prediction-assisted label
adjudication outside development. Feature extraction runs the original OpenCV
pipeline with native source frames and does not use detector predictions as labels.

The first reserved assessment found all six contacts with no extra contacts. Those
same sections were rechecked after an additional Jiang pipeline-window fix and
remain regression checks. Their predictions did not drive that fix. The saved
comparison measures the pre-change detector from commit `f16400e` on version-32
features against current rules on freshly extracted version-33 features, using
the same 14 adjudicated contact brackets and a 0.25-second tolerance.

| Reviewed contact cases | Before | After |
| --- | ---: | ---: |
| Matched contacts | 12/14 | 14/14 |
| Missed contacts | 2 | 0 |
| Extra contacts | 1 | 0 |

The final original-mode suite passes 865 tests. Ruff and Git whitespace checks
also pass. Regression tests cover native referee frames, camera dissolves, both
long and proposal-window pack occlusion, repeated pipeline confirmation, and
rejection when object departure, continuous native coverage or a genuine roll
is missing.

The whole pipeline was also run on a losslessly encoded source minute from Jiang.
Its first pass still dropped the last red, exposing a shorter-window confirmation
gap. The corrected rerun retains all five shots with no extras and produces a
27.60-second highlight video. All five complete contact brackets are covered;
six source/copy decoded frame pairs are pixel-identical, and the rendered output
passes full decoding. Source clip starts, contacts, rolling frames and ends were
visually inspected. Exact physical-stop ground truth was not independently labelled.

A separate pre-change full-pipeline baseline on the first 620 seconds of
Robertson–Un-Nooh found 30/32 originally labelled contacts and 11 extras. Native
review corrected the initially late break-off label, yielding 31/32 and 10 extras
on the additive corrected reference. A current-rules rescore of its old feature
evidence still has referee, replay and advertisement failures. That rescore is
not a fresh full-pipeline run with the corrected ball-scale extraction. It remains
an explicit unresolved benchmark, not evidence of perfect accuracy.

Reproducible data and results:

- `benchmarks/classic_library_sections_20261010.json`: initial source-only labels.
- `benchmarks/classic_library_sections_20261010_adjudicated.json`: additive development correction.
- `benchmarks/classic_library_calibration_20261010_results.json`: before/after contact results and rule hashes.
- `benchmarks/robertson_un_nooh_first_frame*.json`: retained longer baseline references.
- `tests/fixtures/classic_library/`: native source frames and bounded feature regressions.
- Local audit directory: `data/editor_audit/original_2026_10_10` (ignored by Git).
- Local rendered sample: `pipeline-section-export/highlights.mp4` under that audit directory.

Commands from the repository root:

```powershell
.venv\Scripts\python.exe tools/classic_library.py 'G:\Adobe Premiere Pro Auto-Save\Snooker yt shorts\nWindows' --output data/editor_audit/library.json
.venv\Scripts\python.exe tools/classic_extract.py benchmarks/classic_library_sections_20261010_adjudicated.json --source-root 'G:\Adobe Premiere Pro Auto-Save\Snooker yt shorts\nWindows' --output data/editor_audit/classic-native
.venv\Scripts\python.exe tools/classic_calibrate.py data/editor_audit/classic-native/manifest.json --grid benchmarks/classic_library_threshold_grid.json --output data/editor_audit/classic-fit.json --candidate-config data/editor_audit/classic-candidate.yaml
.venv\Scripts\python.exe -m pytest tests -q --ignore-glob=tests/test_dl_*.py
```

The grid is the Cartesian product of motion-start speed `[0.8, 1.0, 1.2]`, minimum
track confidence `[0.35, 0.45]`, and stationary speed `[0.5, 0.65, 0.8]`, under their
existing `strike_fusion` configuration keys. Threshold values and labels are kept
separate from extracted features. No candidate YAML needs to be enabled: the
validated changes are in the original detector's production rules. Feature cache
version 33 and result policy 36 prevent future analyses from reusing obsolete
observations or final clip selection. Previously saved user exports are not rebuilt
automatically. This calibration improves measured cases; accuracy across the full
library and future footage remains unmeasured.
