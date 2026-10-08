# DL Algo reviewed data and evaluation

These JSON manifests freeze **121 reviewed cue-contact uncertainty brackets**
from four local source files belonging to **two underlying matches**. Raw videos,
feature archives, checkpoints and audit images are local assets. The manifests
contain source-review provenance and SHA-256 hashes of the annotation files.

| Split | Match | Reviewed contacts | Scope |
|---|---|---:|---|
| Train | Zhao Xintong–Judd Trump, 2026 quarter-final | 66 | One complete supplied frame with 45 contacts, six earlier reviewed intervals with 21 contacts, and four handling intervals from another cleaned frame |
| Validation (`dev`) | Jack Lisowski–Mark Selby, 2026 quarter-final | 55 | Seven complete reviewed intervals; all other source times are unknown |
| Independent holdout | None available | 0 | A third unseen match with independently frozen labels is required |

Different frames, original uploads, cleaned cuts and replays from the same match
belong to one group. Adjacent-frame random splits would leak source content and
are prohibited. Older classic benchmark split names are retained as provenance;
they do not override the neural match split. Validation participates in model
selection and threshold calibration, so it cannot also be a final neural test.

The original reviews are development annotations. Some labels were refined after
classic predictions revealed omissions. This dataset can train and compare a
development model; it cannot establish that DL is better on unseen matches.

## Labels

The five heads are `event`, `keep`, `end`, `replay`, `handling`, in that order.

Contacts retain `lower`/`upper` source-time uncertainty. One contact creates one
multiple-instance bag of plausible samples. Bag samples are excluded from the
framewise event-negative loss. When a narrow native-frame bracket lies between
sampled features, the nearest half-cadence cell can support it. An extraction
that has no sample close to a labelled contact raises an error.

Event negatives are supervised only inside intervals reviewed exhaustively for
contacts. Unreviewed times are masked. Replay/handling positives require explicit
source-review intervals. Their negatives are supervised where a clip was
explicitly confirmed free of that behavior. Non-replay is also inferred only
inside reviewed live-contact uncertainty brackets, with
`derived_from_reviewed_live_contact` provenance and an explicit inference note.
That inference does not label adjacent time or imply that the referee is absent;
handling can coexist with a live stroke. All other unknown behavior is masked.
Annotation weights must be finite numbers in `[0,1]`; malformed weights are
rejected both when loading manifests and constructing targets.

Training `keep`/`end` preferences come from visually reviewed classic clip
boundaries. They carry **0.35 weight** and explicit
`classic_generated_source_output_reviewed` provenance. They are editorial
preferences, not independent human truth or physical ball-stop annotations.
The referee followup provides four handling intervals and three reviewed clip
preferences; its algorithm-estimated strikes are not converted into invented
contact uncertainty brackets.

Those handling examples have no exhaustive contact review. They therefore
supervise handling and the weak positive clips, while contact and negative
keep/end supervision remain unknown. Handling can coexist with a live stroke.

For validation, `selby_opening_editorial_review.json` freezes an additional source
review of the first 12 Selby contacts. The reviewer inspected 1,456 unique original
source frames without consulting neural predictions or classic clip boundaries.
It supplies 12 editorial start/end preferences with uncertainty bounds, four
referee-handling spans, 12 explicitly inspected handling-free spans, and two
replay spans, all at **0.70 weight**. Unsampled neighboring behavior remains
unknown. These AI visual judgments have not been human approved and are
development labels, not a final independent test. Their ends may precede a
physical ball stop to avoid referee handling.

One pink contact was adjudicated from `441.00–441.08` to `441.92–441.96` seconds
using adjacent original-source frames. Only the DL manifest changes. The original
classic benchmark stays intact, and `selby_contacts_before_adjudication.json`
retains its original labels. Evaluation reports both original and adjudicated
contact metrics. Evidence-sheet hashes, native frame indexes and source times
are recorded in the frozen visual-review JSON; audit images remain local assets.

## Files and reproduction

`manifest.json` contains every video and split; `train.json`, `dev.json` and
`holdout.json` contain the corresponding subsets. `paths_relative_to` resolves
paths from the manifest directory to the repository root. External media paths
must be relocated when the dataset is moved. `source_fingerprint` records local
file size and modification time; it is explicitly not a cryptographic video hash.

Rebuild from the original local review files without running the classic detector:

```powershell
.venv/Scripts/python.exe tools/dl_prepare_data.py
```

Use `--root`, `--output` or `--selby-source` when needed. The importer requires the
local audit JSON files listed in annotation provenance, plus the source videos.
The tracked manifests remain usable for training after paths are updated, even
when those original local audit files are not available.

Feature archives use numeric `timestamps[T]` and `features[T,D]` arrays and a JSON
scalar `feature_spec`. Timestamps are source presentation seconds, monotonically
increasing. Archives are loaded with `allow_pickle=False`. Features must come from
the declared frozen neural RGB/learned-motion extractor, not classic detections.

## Evaluation

```powershell
.venv/Scripts/python.exe tools/dl_evaluate.py `
  --video-id zhao_trump_full_frame_oct07 `
  --predictions data/jobs/DL_JOB/export/export_metadata.json `
  --classic data/jobs/20261008-011704-f177057e/export/export_metadata.json `
  --output data/dl/evaluation/zhao_comparison.json
```

The evaluator matches contacts one-to-one, maximizing count before minimizing
error outside each uncertainty bracket. Duplicates reduce precision. Contact
predictions outside exhaustively reviewed intervals are counted as ignored,
never silently treated as correct or incorrect. It separately reports signed
boundary errors against reviewed editorial preferences and clips overlapping
explicit replay or referee-handling intervals. Absence of handling labels is
not evidence that an export is free of handling.

The default contact tolerance is 0.5 seconds and is printed in results. Use
`--tolerance 0` for strict uncertainty-bracket matching. Precision/recall are null
when their denominator is empty; an all-negative interval does not manufacture
perfect shot recall. A report identifies its split and whether an independent
holdout exists. Compare against the current corrected classic export on exactly
the same video and reviewed intervals, then inspect the actual clips. Broader
accuracy and highlight quality need an independent match and new boundary labels.
