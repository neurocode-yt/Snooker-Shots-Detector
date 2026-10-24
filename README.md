# Snooker AI

**Production-oriented automatic snooker shot detection and video editing.**

Upload a full match or highlights reel → detect genuine cue strikes and ball-stop
points → remove dead time between shots → automatically export a joined video.
The web timeline remains available for optional inspection.

> **Phase 1 baseline:** rule-based visual pipeline (scene cuts, table mask,
> camera-motion compensation, residual table motion, cue-ball trajectories, state machine,
> replay heuristics). Learned detectors/temporal models are scaffolded for Phase 2/3.
> **No accuracy numbers are claimed without measurement on your broadcasts.**

## Features

- **Strict edit mode:** 2s lead-in, at least 4s clear footage and 2s after impact; trim 2–3s from the end when those minimums allow
  (pre-roll and safety horizon configurable)
- **Visual shot detection** — cue/ball trajectories, camera compensation and table motion; commentary does not trigger shots
- **Multiple camera angles** — resets local tracking at cuts, links continuing shots, adapts close-up scale and requires complete table coverage to confirm all-ball stillness
- **Replay association** — excludes verified repeated motion/layouts; incomplete unmarked replays can remain uncertain
- **Referee handling and broadcast cutaways** — suppresses supported ball-handling actions and sustained mismatching score overlays
- **Pre-analysis match editor** — split/delete frame breaks with a 1×–64× zoomable timeline; original uploads remain untouched
- **Review UI** — move boundaries, add/delete shots, mark replays, export labels
- **Selected-shots preview** — immediately play included shots as one continuous virtual timeline
- **Export** — individual clips, one combined MP4, CSV, EDL, training labels
- **Open in CapCut** — create a Windows CapCut 9.5/9.6/9.7 project with separate editable shots
- **Library projects** — name a match across saved versions and delete selected projects with their resources and linked CapCut drafts
- **Make Shorts** — turn included shots into separate vertical videos with player names, layout preview, and a ZIP download
- **Current work / Pause / Resume** — see active and waiting matches, exports, and Shorts, and pause without losing saved progress
- **Jobs** — progress, resume, batch CLI
- **Automatic workflow** — upload and start once; detection, cutting, and combined export finish without review approval
- **Adaptive analysis** — native cadence at cue contact and stop boundaries; cheaper tracking follows the roll and exits on confirmed stillness
- **Windows-first** + **Docker** for servers
- **GPU optional** (Phase 2 torch); CPU fallback always works

## Requirements

An **on-device Android application** is also available in `apps/android`.
It embeds the existing classic detector and editor, processes videos on the
phone, and exports through Android's save/share picker. See
[Android installation, build and platform notes](docs/android.md).

- Python 3.10+
- FFmpeg + FFprobe on `PATH`
- Windows 10/11 (primary), Linux (Docker)

## Install (Windows)

```powershell
cd snooker
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[dev]"
```

See [docs/windows_setup.md](docs/windows_setup.md).

## CLI

```bash
snooker-ai analyze input.mp4
snooker-ai review <job-id>
snooker-ai export <job-id> --output highlights.mp4
snooker-ai batch ./matches
snooker-ai evaluate ./test-dataset
snooker-ai train ./training/dataset_config.example.yaml
snooker-ai serve --port 8000
```

Strict mode starts at the first confirmed cue-ball launch minus 2.000 seconds and
chooses an end trim using the shot's duration
from cue strike to that stop (excluding pre-roll and confirmation look-ahead):

- Under 7 seconds, including shots under 5 seconds and the 5–7-second range: remove 2 seconds.
- At least 7 seconds: remove 3 seconds.

Minimum viewing time takes precedence over that end trim: each shot keeps at
least **4 seconds of clear footage**, including **2 seconds after cue contact**.
Mix transitions use extra context outside the four-second minimum. Short shots
therefore receive less end trimming, rather than cutting away at impact. A source
that ends too soon keeps all available footage; the editor does not invent frames.
The detected physical stop remains unchanged in the metadata. The 0.50-second
stationary confirmation supplies evidence and does not itself add edit padding.
Automatic tracking follows longer rolls beyond seven seconds, with a configurable
60-second safety horizon. Uncertainty remains in the diagnostics and does not
block automatic export. A video that ends during a shot keeps its remaining footage.

Completed highlights have editable end offsets. In the editor, open **Trim shot
endings** to change the normal-shot trim, long-shot trim and long-shot threshold,
then apply them to included shots. For one shot, select it and use **Trim ending
(seconds)** in **Source & trim**. The initial bulk values are the existing 2/3
seconds with a 7-second threshold. Settings are absolute: applying them twice
does not remove extra footage. **Restore previous endings** undoes these edits.
The controls work on saved detections, including uncertain endings; no detector
run is required. Minimum playback and cue contact are protected. Physical stop
measurements, point classifications and review flags are preserved. Export again
to update an MP4; CapCut and Shorts use the edited source ranges.

Native contact frames use decoded presentation timestamps and source-frame
identity rather than average-FPS timing. Native stop refinement retains the
observed launch when checking a later tail, so a quiet end window can finish
without decoding the entire safety horizon. Table localisation also keeps
separate green arena lights from joining the enclosed playing surface.

Camera cuts inside a shot retain their original timing; mix transitions join
separate shots. When contact or the final stop is hidden, the editor records a
bounded estimate rather than pretending to observe the exact instant. The
[multicamera detection notes](docs/multicamera_detection_2026-10-02.md) describe
the verified behavior and visibility limits.

The [broadcast recall benchmark](benchmarks/README.md) measures missed and extra
shots against independently inspected source intervals. It separates sections
used to develop fixes from held-out checks; it does not claim full-match accuracy.

`strict` is the only editing mode, with the viewing-time minimum taking priority. When the next
shot starts before the previous window would end (fast break play), the boundary
is trimmed between the two shots so neither shot is lost. Legacy mode names are
accepted and coerce to strict.

## Web UI

In **Your videos**, use **Name project** to set the name for every version of a
match. The name is saved across restarts, appears in the editor, and is used
for new CapCut handoffs. Renaming does not change the source filename or shots.

Select project checkboxes (or **Select matching projects** for the current
search/filter), then **Delete selected**. The confirmation covers all versions
and earlier runs on each selected match card, including analysis, proxies,
clips, exports, Shorts, and every linked CapCut draft created by Snooker AI.
Unshared uploads in the app's upload folder are removed too. Shared sources
and originals outside that folder are retained and reported in the preview.
New preprocessed uploads record their input ownership so unused inputs can
also be cleaned; older files without that receipt are not guessed from names.
Pause processing/Shorts and close CapCut before deleting its linked drafts.
If a resource remains locked, **Retry resource cleanup** finishes the saved
deletion. Cancellation leaves projects untouched; permanent deletion has no undo.

```bash
snooker-ai serve --host 127.0.0.1 --port 8000
```

- Upload: http://127.0.0.1:8000/
- Review: http://127.0.0.1:8000/review/`<job-id>`
- API docs: http://127.0.0.1:8000/docs

Select one or several videos and click **Create highlights** (or **Submit videos**
for a batch). The queue processes **two matches at once** by default; extra matches
wait for a free slot. Each match has separate shots, progress, checkpoints and
exports. You can submit more matches while others process. A failed upload or
submission does not stop the remaining files; retrying skips submissions that
already succeeded and reuses uploads that reached the server. Batch selection
processes full matches; select a single video to trim frame breaks first.

Choose **Processing order → One at a time (queue)** to process matches one after
another. Click **Add videos to queue**; the next waiting video starts automatically
when the previous analysis and automatic export finish. You can keep adding videos
while it works. **Parallel** uses the configured capacity instead. The processing
order is saved across server restarts. Switching to one at a time lets already
running matches finish before starting the next waiting video.

Set `jobs.max_workers` to **1–4** in `configs/default.yaml` and restart the server
to change the limit. Servers sharing that jobs directory also share the worker
slots when configured with the same limit. CPU, GPU, RAM and disk bandwidth are
shared, so simultaneous matches improve batch throughput only when resources
are available; they do not make every individual match faster. This does not
change the default detector, thresholds, sampling or shot selection.

Waiting jobs are saved on disk and automatically requeued after a server restart.
Matches that were already processing retain their checkpoints and offer
**Continue processing** after an interruption. Keep the source uploads in place.
`GET /api/queue` reports this server's configured capacity, active matches and
waiting count; each job exposes `queued` and its local `queue_position`.

The finished MP4 appears
as a download when processing completes; reviewing shots and exporting again are
optional. Source trimming is also optional: split/delete sections before starting
if needed. Keeping the entire source skips that extra re-encoding step.

Upload the entire match in automatic mode; no manual frame-break trimming is
required. The coarse pass skips expensive tracking on stationary racked tables,
wakes on cloth movement, and checks the preceding seconds for the break-off.
When a nearly cleared table is replenished with reds and then racked, the
preparation interval is excluded retrospectively. Hand placement of the white
with an intact rack needs cue-contact evidence to count as a shot. These are
conservative visual heuristics, not an official frame-score detector; ambiguous
views fall back to normal shot detection.

Combined exports use a short **mix (cross-dissolve)** with an audio crossfade.
The default overlap is 0.24 seconds, shortened for very short clips to keep cue
impact visible. Export seeks directly to retained footage and renders bounded
batches, avoiding decoding every long break or loading a whole-match filter
graph. Set `export.transition: cut` to disable mixing or adjust
`export.transition_seconds`. Individual clips retain their selected boundaries.
The interactive shot preview shows cuts; the finished combined MP4 contains the
transitions. Export metadata and chapter times account for overlap; the CSV/EDL
remain source cut lists.

Choose **Classic editor (previous interface)** to analyze first and open the
preview, timeline, individual-shot controls, and export buttons. Existing jobs
also have an **Open classic editor** link. Automatic mode remains the default.

Click **Open in CapCut** in a finished job or the shot editor, then select the new
**Snooker - …** project on CapCut's Home screen. Included live shots are placed
consecutively on its timeline, with source audio. Each shot references the full
analyzed source, so you can extend trims, rearrange clips, add transitions, text,
or music. This handoff uses straight cuts; the combined MP4's rendered mix
transitions are not included. Keep the source video in its current location.

The integration creates a fresh project on every click, preserves existing
CapCut edits, and backs up CapCut's project index. It supports **Windows CapCut
9.5, 9.6 and 9.7**; other versions show a reason instead of writing an unverified
draft. Native opening has been checked in 9.5 and 9.6; 9.7 discovery and draft
generation are covered by automated tests. CapCut must be installed on the PC
running Snooker AI. For a custom installation or project location, configure
`capcut.executable_path` and `capcut.projects_dir` in the YAML configuration.
Discovery checks Windows' registered app-data and installation locations as well
as the terminal environment, including junctions to another drive. It remembers
the last verified executable in `data/capcut_installation.json` and rechecks that
file on each request. A stale configured path falls back to automatic discovery.
Every click checks again; an earlier browser status check cannot block opening.
If opening fails, the action shows the actual reason and remains available to retry.
Preparing the project reuses the saved compact timeline when available.

Click **Make Shorts** on an analyzed video or in the shot editor to create one
1080×1920 MP4 per included live shot. Enter the player or match names, choose an
editor-style layout preset, and check the shot preview. **Canvas & video** and
**Text size & position** adjust colors, zoom, placement, and text. Optional
per-shot player names override the match name. Titles and filenames use the
original editor shot number, such as `Judd Trump - Shot 03.mp4`.

Shorts render directly from the analyzed source and saved trims, preserving
original audio. Replay and excluded records stay out. No description, voiceover,
AI request, or upload is generated. Existing completed videos work immediately
without detecting shots again. Review flagged shots in the editor if needed.

Rendering continues when the dialog closes. Reopen **Make Shorts** to see progress,
download an individual MP4, **Download all ZIP**, or **Open Shorts folder**. Files
are stored in `data/jobs/<job>/export/shorts/<batch>/` by default. One Shorts batch
renders at a time per server. Completed clips are checkpointed; after a failure
or server restart, reopen the dialog and click **Resume Shorts**. Keeping the same
settings and shot selection reuses validated completed clips. New settings or
edited selections create a separate batch. An in-progress batch keeps the
selection captured when it started.

**Current work** appears above the workspace on both the library and shot editor.
It shows active, queued, paused, and interrupted match processing and Shorts
exports, including stage, progress, and the number of Shorts already finished.
It remains visible after closing the Shorts dialog. Use **Pause** on a task and
wait for **Pausing…** to become **Paused**; then **Resume** continues saved work.
Shorts also has a Pause button inside its dialog. Paused waiting videos do not
start automatically and paused work remains available after a server restart.

Pausing a detector scan saves the current streaming state; completed refinement
windows remain reusable. Pausing an encode stops that owned FFmpeg command; the
unfinished encode is redone on resume. Finished Shorts are kept. Resuming a
paused Shorts batch uses its original selection and layout, and resuming a manual
export keeps its original output/filter settings. Some analysis routines finish
their current step before acknowledging Pause; the UI shows **Pausing…** until
the worker has stopped.

API clients can set `auto_export: false` on `POST /api/jobs` to request analysis
only. Automatic web jobs export a combined video and avoid encoding
separate numbered clips unless explicitly requested later.

If processing stops or the laptop shuts down, restart the server and click
**Continue processing** in Your videos or the editor's More menu. It uses the same job, source,
and saved edits. The original detector saves coarse observations and tracking
state about once a minute, reuses completed refinement windows, and reuses saved
analysis if only export needs retrying. Export itself is retried from that stage.
Jobs whose worker has stopped show **Interrupted** with an error. Decoder failures
and incomplete proxies cannot silently finish as a successful full-video scan.
API clients can resume with `POST /api/jobs/{job_id}/continue`.

Choose **Pots only** to run the original detector and keep confirmed
scoring pots, plus the opening and final live shot of each identified snooker frame.
It compares repeated stable points scores before/after a stroke,
uses the pre-shot turn indicator to exclude foul points awarded to the opponent,
and waits for delayed updates. Centre frame/match counters are not points scores.
Other unreadable scores, corrections and ambiguous outcomes are excluded automatically.
Frame context is labelled separately in the editor; a non-scoring opening or final
shot keeps its actual score status. Frame identification requires a stable score
reset and a visually intact opening rack. A bounded native check can recover an
opening contact obscured by a camera cut. Unsupported overlays or missing source
footage can prevent frame identification.

The default classical detector also reads supported score graphics during its
coarse pass. Stable points changes without a corresponding detected contact
request an additional native visual check. A score change alone never creates
a shot: cue motion/contact, replay, handling, and match-context checks still apply.
This helps recover rapid red/colour sequences while preserving non-scoring shots
in **All shots**. Disable with `analysis.scoreboard_recall: false` if needed.
Supported graphics include paired score boxes, older outer-point layouts, and
white points beside a neutral grey frame-count panel. Glyph matching remains
classical; no DL mode or neural training is used.

Choose **Pots + fouls** to also keep foul points and confirmed point
increases whose pot/foul type is uncertain. It also retains each identified frame's
opening and final shot. Other unchanged scores are skipped; unconfirmed scoring
outcomes remain recorded in the saved result.

Older filtered exports offer **Update frame shots**. This creates a new version
using saved detection and compatible score observations, preserving the earlier
export and explicit inclusion/exclusion choices. Opening-rack checks have their
own source-bound checkpoints, so completed frame checks are reused on Continue.

Completed videos in Your videos have a one-click **Pots + fouls** button. The
editor offers **Create pots + fouls version** under More. It creates a separate version from the saved
shot timeline without a new full-match scan or upload. Compatible
score checkpoints are reused; the first score check on a video still takes time,
and export encodes only the selected clips. Repeated clicks return the finished
version or resume its interrupted work. Original jobs, exports and explicit
review choices are preserved. Use **Show more videos** to reach older results.

Clicking **Pots + fouls** opens a progress dialog for score checking and export.
If the highlights already exist, it shows **Pots + fouls highlights already
made**, then asks **Open in CapCut?** The same prompt appears when a new version
finishes. CapCut receives the pots + fouls version. You can also download it or
open its editor. **Keep working in background** closes the dialog while server
processing continues; click **Pots + fouls** again to follow that job. Failures
show a reason, and inconclusive scores offer the editor instead of an empty video.

Filtered versions with unresolved scores also offer **Recheck scores** in the
library. It creates a separate version, preserving the earlier result and edits.
The summary records readable score counts and layouts; a completely unreadable
scoreboard is shown as **Score check needed**, rather than claiming no pots occurred.

API clients can call `POST /api/jobs/{job_id}/points-highlights` for this fast
completed-video action, or set `selection: "pots_and_fouls"` or `"pots_only"`
on `POST /api/jobs` for new matches. Both default workflows continue to use
`selection: "all"` and skip the additional score scan. Restart the app server
after updating to enable these options.

The video library groups versions of the same source match together. Each match
shows **All shots**, **Pots + fouls**, and **Pots only**, with independent statuses
and actions. **Not created** identifies a missing version; **Create** reuses saved
shot detection and cached score evidence, then exports the new selection. Existing
videos and edits are preserved. Earlier runs remain under an expandable history.

Each ready version has its own **Download MP4**, **Open editor**, **Open in CapCut**,
and **Make Shorts** actions. **Shots ready** means detection is saved but its MP4
has not been exported; choose **Export video** for that version. Paused/interrupted
versions offer Resume/Continue. Filter matches by Processing, Ready, Needs attention,
or Missing versions, and search across their saved versions. Restart analysis and
processing details remain in each version's More menu. Restart the app server after
updating to enable creating versions from the library.

The editor shows one player at a time: **Highlights preview** or **Source & trim**.
Filter the shot list and use its checkboxes to include/exclude shots, then export
the updated video. Saved edits preserve the previous file but mark its download
as out of date until a new combined export succeeds. Editing controls pause
during export. Optional frame-break trimming stays collapsed until needed.

An editor opened during processing follows its progress and loads the saved
shots when the job finishes. Individual-clip export keeps an existing combined
download available. Replay changes and other saved edits require a new combined
export; shot edits are blocked while that job is processing in another tab or
server instance. Refreshing saved shots preserves the current media and playhead.

The timeline stays inside its own horizontally scrollable viewport at every
zoom level. Drag the yellow playhead to seek. Use `Z` to split, `Ctrl+Z` to
undo, and `Ctrl+Shift+Z` to redo.

## Docker

```bash
cd docker
docker compose build
docker compose up
```

## Configuration

All thresholds live in [`configs/default.yaml`](configs/default.yaml):

- Proxy resolution / analysis FPS  
- Motion & strike fusion weights  
- Mode pre/post-roll  
- Export codec / CRF  
- Confidence bands (fail-safe keep extra footage)

## Repository layout

```text
snooker_ai/          # Core pipeline modules
apps/api/            # FastAPI
apps/web/            # Review UI
configs/             # default.yaml
training/            # Phase 2/3 train scaffold
annotation/          # Labeling spec
tests/               # Unit + synthetic e2e
docker/              # Dockerfile + compose
docs/                # Architecture, API, setup
```

## Pipeline (Phase 1)

1. **Ingest** — FFprobe metadata, validation  
2. **Proxy** — lower-res analysis video with preview audio
3. **Table mask** — HSV green cloth + contour  
4. **Camera motion** — affine from LK features; residual flow on table  
5. **Scenes** — histogram cuts + view heuristics  
6. **Audio** — retained for playback and export; excluded from shot detection
7. **Strike fusion** — cue-ball transition scoring, contact geometry, and visual occlusion recovery
8. **Dense refinement** — native-FPS windows at strike/stop edges only  
9. **Ball stop** — settling period on ball-specific and residual motion  
10. **Segments** — mode rolls, overlap merge, confidence review flags  
11. **Export** — accurate re-encode cuts + concat  

## Testing

```bash
pytest -q
```

## Evaluation

Prepare a dataset folder per video with `ground_truth.json` and `predictions.json`
(or `analysis.json`), then:

```bash
snooker-ai evaluate ./test-dataset --output benchmark_report.json
```

Template: [docs/benchmark_report_template.md](docs/benchmark_report_template.md)

## Limitations (honest)

- Phase 1 does **not** use a trained ball/cue network by default (optional blob cues).
- View/replay classifiers are **heuristic**, not broadcast-package-specific CNNs.
- Accuracy on multi-hour TV matches must be **measured** on your data.
- Extreme lighting, non-green cloths, or heavy mobile vertical video may need config tuning.
- Prefer keeping extra frames over missing a strike (fail-safe).

## Documentation

- [Architecture](docs/architecture.md)
- [Classical performance and parity checks](docs/classic_performance.md)
- [Murphy–Day detection and score repair](docs/classic_murphy_day_repair_20261017.md)
- [API](docs/api.md)
- [Windows setup](docs/windows_setup.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Example workflow](docs/workflow_example.md)
- [Annotation spec](annotation/SPEC.md)

## License

MIT
