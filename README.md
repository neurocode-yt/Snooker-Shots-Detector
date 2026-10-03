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
- **Jobs** — progress, resume, batch CLI
- **Automatic workflow** — upload and start once; detection, cutting, and combined export finish without review approval
- **Adaptive analysis** — native cadence at cue contact and stop boundaries; cheaper tracking follows the roll and exits on confirmed stillness
- **Windows-first** + **Docker** for servers
- **GPU optional** (Phase 2 torch); CPU fallback always works

## Requirements

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

```bash
snooker-ai serve --host 127.0.0.1 --port 8000
```

- Upload: http://127.0.0.1:8000/
- Review: http://127.0.0.1:8000/review/`<job-id>`
- API docs: http://127.0.0.1:8000/docs

Select a video and click **Create video automatically**. The finished MP4 appears
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

API clients can set `auto_export: false` on `POST /api/jobs` to request analysis
only. Automatic web jobs export a combined video and avoid encoding
separate numbered clips unless explicitly requested later.

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
6. **Audio** — onset / band energy (capped weight)  
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
- [API](docs/api.md)
- [Windows setup](docs/windows_setup.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Example workflow](docs/workflow_example.md)
- [Annotation spec](annotation/SPEC.md)

## License

MIT
