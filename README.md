# Snooker AI

**Production-oriented automatic snooker shot detection and video editing.**

Upload a full match or highlights reel → detect genuine cue strikes and ball-stop
points → remove dead time between shots → automatically export a joined video.
The web timeline remains available for optional inspection.

> **Phase 1 baseline:** rule-based multimodal pipeline (scene cuts, table mask,
> camera-motion compensation, residual table motion, audio onsets, state machine,
> replay heuristics). Learned detectors/temporal models are scaffolded for Phase 2/3.
> **No accuracy numbers are claimed without measurement on your broadcasts.**

## Features

- **Strict edit mode:** 2s before cue contact → hold until every ball stops  
  (pre-roll and safety horizon configurable)
- **Multimodal detection** — residual motion after camera compensation + table mask + audio support
- **Replay-aware** — replays flagged and excluded by default
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
normally ends at the first physical all-ball stop. The 0.50-second stationary
confirmation is look-ahead evidence only and is not included in `clip_end`.
Automatic tracking follows longer rolls beyond seven seconds, with a configurable
60-second safety horizon. Uncertainty remains in the diagnostics and does not
block automatic export. A video that ends during a shot keeps its remaining footage.

`strict` is the only editing mode (2s before strike → balls stop). When the next
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

API clients can set `auto_export: false` on `POST /api/jobs` to request analysis
only. New web jobs export a combined video automatically and avoid encoding
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
2. **Proxy** — lower-res analysis video + mono WAV  
3. **Table mask** — HSV green cloth + contour  
4. **Camera motion** — affine from LK features; residual flow on table  
5. **Scenes** — histogram cuts + view heuristics  
6. **Audio** — onset / band energy (capped weight)  
7. **Strike fusion** — linear-time cue-ball transition scoring + supporting audio  
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
