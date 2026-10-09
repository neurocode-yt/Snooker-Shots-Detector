# DL Algo

`DL Algo` is a separate neural detection algorithm. The existing algorithm stays
available as `classic` and remains the default. The algorithm choice is separate
from the existing `strict` editing mode. Switching algorithms creates a separate
job instead of replacing a previous analysis, corrections or exported video.

The neural pipeline learns its own selection decisions from source images and
learned motion features. It does not train on the classic detector's handcrafted
feature vectors or call its strike detector. Source decoding, the review UI and
the final video renderer can be shared without sharing detection decisions.

## Model design

The initial backbone combines frozen **DINOv2 ViT-S/14** RGB embeddings with
pretrained **RAFT-small** optical flow. DINOv2's official repository provides
pretrained visual backbones, including the approximately 21-million-parameter
small model. This implementation uses transfer learning rather than attempting
to pretrain a visual foundation model on the local clips.
[DINOv2 official repository](https://github.com/facebookresearch/dinov2).

Global RGB context and four spatial crops expose the temporal model to both the
broadcast view and local action. RAFT predicts dense movement between image
pairs, providing a learned motion signal. TorchVision supplies RAFT-small and
its pretrained transforms; its input dimensions must be divisible by eight and
its weight transforms normalize image values appropriately.
[RAFT-small documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html),
[TorchVision optical-flow example](https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html).

The snooker-specific component is a new multi-scale residual temporal
convolutional network. A shorter branch captures the immediate stroke; a longer
branch supplies preparation and follow-through context. A spatial convolution
head learns local flow patterns with shared filters across table positions.
Channel normalization also shares statistics across positions. RGB and motion
embeddings are fused before temporal prediction. Its five heads are:

| Head | Learned decision |
|---|---|
| `event` | A new live cue contact |
| `keep` | Useful footage for a highlight |
| `end` | An editorial clip endpoint |
| `replay` | Repeated broadcast footage |
| `handling` | Referee ball handling |

Only the temporal network is optimized on the reviewed snooker data. Frozen
backbones allow feature extraction to be reused across training runs. Neural
probabilities still require temporal peak selection and clip constraints; those
operations are editing policy rather than additional learned evidence. The
`end` target represents an editorial boundary and does not establish that every
ball physically stopped.

This is a practical first architecture for the available hardware and labels.
It has not been established as the best snooker algorithm. Larger or newer
backbones need a controlled comparison on independently labelled matches before
they can justify more computation or an accuracy claim.

## Training data and provenance

The frozen manifests in `data/evaluation/dl/` contain **121 reviewed contact
brackets from two underlying matches**:

| Neural split | Match | Contacts | Reviewed scope |
|---|---|---:|---|
| Train | Zhao Xintong–Judd Trump, 2026 quarter-final | 66 | One complete supplied frame, six earlier source intervals, and separate handling examples |
| Validation | Jack Lisowski–Mark Selby, 2026 quarter-final | 55 | Seven bounded source intervals |
| Independent holdout | None available | 0 | A third unseen match is needed |

Every cleaned frame, original source and replay belonging to one match uses the
same `group_id`. Training and validation contain different matches. Random
adjacent-frame splits would exaggerate generalization and are not used.

The original reviews were development audits. Some timing and completeness
corrections followed inspection of classic predictions. The manifest preserves
this history, source-review descriptions and SHA-256 hashes of the annotation
JSON files. Earlier classic holdout names remain annotation provenance; once
used for neural model selection, those windows are neural validation data.

Unreviewed times are **masked**, including the unreviewed parts of otherwise
labelled sources. Contacts retain uncertainty bounds rather than invented exact
impact frames. A contact produces one multiple-instance bag; the loss encourages
one localized event within that bag and does not label every frame in it as a
separate shot. Negative contact supervision applies only to intervals reviewed
exhaustively for contacts. Replay and handling labels require explicit source
review; unknown behavior is masked. Non-replay is also inferred inside reviewed
live-contact brackets only, with `derived_from_reviewed_live_contact` provenance.
No adjacent footage or handling-negative label is inferred from a live stroke.
Every annotation weight must be a finite number in `[0,1]`.

There are also 44 **weak timing preferences** from source-reviewed classic
contact estimates, clipped inside the existing contact brackets at 0.35 weight.
They supply approximate timing anchors, rather than new exact contact truth.
They are training labels only; inference does not call the classic detector.
Their positive loss is averaged separately so easy negative frames do not
dilute it. Online hard-negative mining emphasizes the hardest 10% of reviewed
negative event frames. Unknown frames remain masked.

The separate referee-followup examples supervise handling and the reviewed weak
clips only. Referee presence supplies neither a negative contact label nor a
negative useful-footage label: a live stroke may occur at the same time.

The training `keep` and `end` heads use visually reviewed classic export
boundaries as weak preferences with **0.35 weight** and explicit provenance.
These are not independent human ground truth. Additional blinded source-image
review supplies 12 Selby editorial intervals, four handling-positive spans, 12
explicitly inspected handling-free spans and two replay spans at **0.70 weight**.
Start/end uncertainty is retained. These AI judgments have not been human
approved; all other boundaries and unreviewed behavior remain masked. Validation
replay negatives include reviewed replay-free clips and live-contact brackets.
The selector uses automatic replay suppression only when the report has both
positive and negative labels. Handling is always advisory: it cannot veto a cue
event that passes the event threshold, including a weaker occluded contact.
Label availability is distinct from independently verified accuracy.
The training report records each head's label availability.

The current dataset supports development and group-separated validation. It
does not support a claim that the neural editor surpasses the corrected classic
editor on unseen matches. See [the dataset notes](../data/evaluation/dl/README.md)
for exact source paths, intervals, labels and evaluation semantics.

## Runtime and local hardware

An **NVIDIA GeForce RTX 3050 Laptop GPU with 4096 MiB** of GPU memory was verified
with `nvidia-smi` during setup. Availability can change with the system's graphics
configuration; the worker checks CUDA at runtime and can use the CPU. The initial
design uses small frozen
backbones, bounded image sizes and small inference batches to fit that budget.
Pretraining large foundation models or processing an entire full-resolution
match in one tensor is outside this local design.

The optional `.venv-dl` interpreter runs neural work in a subprocess. The
existing `.venv` continues to run the API and classic algorithm. Runtime paths,
backbone settings and checkpoint selection belong to the separate
`configs/dl_algo.yaml`. Initial weight downloads, feature extraction and temporal
training have different costs; changing temporal hyperparameters should reuse
compatible features instead of repeating backbone inference.

Runtime availability and verified highlight quality are distinct. Missing
dependencies, incompatible features or an absent checkpoint must make the DL
choice unavailable with a reason. The app must not silently substitute classic
when a user selects `dl_algo`. Successful model loading or training alone does
not establish that the resulting export preserves every shot.

## Feature archives and checkpoints

Each source entry declares a `features_path`. The feature archive contains
numeric `timestamps[T]` and `features[T,D]`, plus a JSON scalar `feature_spec`.
Timestamps use source presentation seconds and increase strictly. Extraction
must retain reviewed contact neighborhoods. Disjoint reviewed windows remain
separate in temporal training and inference; a missing minute cannot be treated
as an adjacent feature sample.

Use the same declared feature specification for every training source and for
deployment. Changing backbone weights, spatial crops, preprocessing, sampling
cadence or the motion representation requires new compatible features and a
retrained checkpoint. The loader rejects a checkpoint when its feature
specification disagrees. Archives are read with `allow_pickle=False`; temporal
checkpoints load through PyTorch's tensor-only `weights_only=True` path.

The temporal model stores training-only feature normalization. Validation and
holdout features never contribute to those statistics. AdamW optimization,
masked focal losses, contact-bag loss, gradient clipping and validation early
stopping fit the network. Event thresholds are selected on validation with an
explicit recall target; reaching that target is measured and recorded, not
assumed. Temperature values currently remain one, so threshold selection should
not be described as fitted temperature calibration.

The default deployment artifact is `models/dl_algo/temporal-v1.pt`; its adjacent
`temporal-v1.pt.report.json` records configuration, seed, source/feature
fingerprint, group splits, training history, head supervision and calibration
results. Training initially records `deployment_quality_verified: false`.
Model weights, downloaded backbones, generated feature archives and raw videos
are local assets rather than repository source code.

## Commands

Create the optional environment without changing the classic environment:

```powershell
py -3.12 -m venv .venv-dl
.venv-dl/Scripts/python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv-dl/Scripts/python.exe -m pip install -r requirements-dl.txt
.venv-dl/Scripts/python.exe -m snooker_ai.dl.features --prepare-assets
```

The versions above are the locally tested runtime. Use the official CPU wheel
index instead of `cu128` on systems without a compatible NVIDIA GPU. Model
downloads require internet access; inference uses the saved local assets.

Run extraction, temporal training and decoded-clip benchmarking in one command:

```powershell
.venv-dl/Scripts/python.exe -u tools/dl_train_pipeline.py --device auto
```

This is one run with no scheduler. CUDA is selected when available; otherwise
the command uses four CPU threads. CPU backbone extraction can take hours.
Compatible feature archives are reused, and persisted feature chunks resume
after an interrupted extraction. Stage logs go to stdout. Atomic
`data/dl/training/state.json` records the process ID, current video, progress,
training epoch, artifact paths, completion or failure. An OS-owned exclusive
lock prevents another command using the same state file from overwriting the
active run. Rerunning trains the temporal head again after feature reuse.

New DL jobs can reuse a fully extracted source from `feature_cache` when its
source identity, entire requested coverage and feature specification match.
Partially reviewed training sources cannot substitute for a whole-source job.
Keep runtime-test checkpoints in a separate output directory; they are not
deployment artifacts.

`data/dl/training/benchmarks/` receives per-video probabilities, decoded clip
records, selection diagnostics, uncertainty-aware contact metrics and editorial
boundary/handling/replay measurements. The decoder uses the checkpoint's exact
temperatures, thresholds and validated heads with the deployment settings.
Original and adjudicated contact metrics are both reported where labels were
corrected. Existing classic exports are compared when the manifest declares
a local baseline. These benchmarks inspect selected clip records; an MP4 render
and source-to-output visual review remain separate checks.

Use `--manifest`, `--assets`, `--output`, `--state`, `--benchmarks`, `--device`,
`--epochs` or the other training options to select explicit paths and settings.
The default output is the deployment checkpoint configured above; a custom
checkpoint requires a matching runtime configuration before the app can use it.
Completing the command does not establish accuracy on unseen matches or a
guarantee that every shot is preserved.

After recording match-separated development results, an explicit final refit
can use both training and development groups:

```powershell
.venv-dl/Scripts/python.exe tools/dl_refit.py --all-reviewed --device auto
```

This fits flat-motion and spatial-motion neural variants, selects their
probability mixture, and writes a deployment checkpoint. Any declared holdout
group remains excluded from fitting and calibration. Refitting the current
two-match dataset leaves no independent test; calibration and known-source
benchmarks are explicitly reported as in-sample.

Rebuild source-label manifests from the original local review files:

```powershell
.venv/Scripts/python.exe tools/dl_prepare_data.py
```

Extract compatible neural feature archives for the source entries in the
isolated DL environment:

```powershell
.venv-dl/Scripts/python.exe -m snooker_ai.dl.features `
  --manifest data/evaluation/dl/manifest.json --assets models/dl_algo/backbones
```

Once the declared archives exist, train the temporal checkpoint:

```powershell
.venv-dl/Scripts/python.exe -m snooker_ai.dl.training `
  data/evaluation/dl/manifest.json `
  --output models/dl_algo/temporal-v1.pt --device cuda
```

Frozen manifests already specify training and validation groups. Their splits
cannot be contradicted with command-line overrides. A development-only run
without a validation match requires the explicit `--development-only` option
and must retain its unvalidated status.

Select the independent algorithm explicitly for a new analysis:

```powershell
.venv/Scripts/python.exe -m snooker_ai.cli analyze "SOURCE.mp4" --algorithm dl_algo
```

Omitting `--algorithm` selects classic. API clients use the independent
`algorithm` field (`classic` or `dl_algo`) in `POST /api/jobs`; `GET /api/algorithms`
reports readiness. The browser presents the additional choice as **DL Algo**.

Evaluate a neural export against the corrected classic export on the exact same
reviewed source:

```powershell
.venv/Scripts/python.exe tools/dl_evaluate.py `
  --video-id zhao_trump_full_frame_oct07 `
  --predictions data/jobs/DL_JOB/export/export_metadata.json `
  --classic data/jobs/20261008-011704-f177057e/export/export_metadata.json `
  --output data/dl/evaluation/zhao_comparison.json
```

The evaluator uses one-to-one uncertainty-aware matching, penalizes duplicate
contacts and excludes unreviewed predictions from measured precision/recall.
It separately records errors against weak editorial boundary preferences and
overlap with explicitly labelled replay or handling intervals. The default
report also checks whether each complete contact uncertainty bracket falls
inside a delivered source clip. This distinguishes timing errors from footage
omissions; it still does not prove that the complete shot outcome is visible.
The default
contact tolerance is 0.5 seconds; `--tolerance 0` checks strict uncertainty-bound
coverage. Generated videos still need source-to-output inspection and a complete
decode check. Freeze new source annotations before consulting either algorithm's
predictions when adding the missing independent match.

## Independent video test, 9 October 2026

The deployed experimental checkpoint failed its first two unseen-match tests.
Source-only visual labels were frozen before predictions; neither match was
used for fitting, normalization, threshold calibration or model selection.
Both complete cleaned clips were processed at 8 fps with the original model
weights and deployment thresholds.

| Unseen match | Source duration | Reviewed shots | Retained shots | Missed shots | Extra shots |
|---|---:|---:|---:|---:|---:|
| Hawkins–Un-Nooh | 8:34 | 32 | 1 | 31 | 0 |
| Nutcharut–Dikme | 12:55.48 | 36 | 2 | 34 | 0 |

Combined shot recall was **3/68 (4.41%)**. Both strict contact-bracket matching
and a 0.25-second tolerance produced the same counts. The delivered source
windows fully preserved only those three contact brackets. Event candidate
counts were also one and two: the omitted strokes failed the frozen event
threshold, before highlight boundary selection. The median event peak within
reviewed contact brackets was 0.0182 and 0.00102 respectively, against the
deployed event threshold of 0.95.

The source labels are AI visual reviews at 1 fps, with uncertainty intervals;
they are not human-approved or frame-exact ground truth. One Hawkins red uses a
two-second bracket. Neither cleaned clip contains a replay. Zero extra clips
and zero labelled handling overlap are insufficient evidence of good exclusion
when 65 shots are omitted. The original 121/121 in-sample result therefore
cannot be presented as accuracy on unseen footage.

Both actual highlight exports passed complete FFmpeg decoding. Six decoded cue
frames matched the expected source footage on visual inspection. A separate
post-prediction source review found two retained clips ending while balls were
still moving: Hawkins at source 503.82 seconds and Dikme's brown at 613.82
seconds. These observations are output QA, not new blind training labels.

Frozen source labels, every missed timestamp, model probabilities at labelled
contacts, compact results and output QA are stored in
[`data/evaluation/dl/independent_20261009`](../data/evaluation/dl/independent_20261009).
Large feature archives, frozen weights, source sheets and MP4s stay local under
`data/dl/independent_20261009`. The original/classic algorithm and existing jobs
were not changed by this evaluation. The DL checkpoint was not retrained or
retuned to improve this test's score.

Reproduce the inference-only evaluation in the isolated DL environment:

```powershell
.venv-dl/Scripts/python.exe tools/dl_holdout.py `
  --manifest data/evaluation/dl/independent_20261009/manifest.json `
  --checkpoint data/dl/independent_20261009/frozen-model.pt `
  --output-dir data/dl/independent_20261009/reproduced --device cpu
```

The command checks the planned checkpoint hash, fitting and model-selection
groups across every ensemble member, feature specification, source identity
and complete feature coverage. It refuses a reused output directory associated
with different frozen inputs. It never fits model weights or thresholds, and
keeps deployment quality unverified. If either test match is later used for
improvement, that follow-up result must be labelled development data and a
fresh unseen match must be reserved for the next independent test.
