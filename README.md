# Snooker AI

**Production-oriented automatic snooker shot detection and video editing.**

Upload a full match or highlights reel → detect genuine cue strikes and ball-stop
points → remove dead time between shots → export a smooth joined video with
perfect audio sync — with a web timeline for reviewing uncertain detections.

> **Phase 1 baseline:** rule-based multimodal pipeline (scene cuts, table mask,
> camera-motion compensation, residual table motion, audio onsets, state machine,
> replay heuristics). Learned detectors/temporal models are scaffolded for Phase 2/3.
> **No accuracy numbers are claimed without measurement on your broadcasts.**

## Features

- **Three edit modes:** Action Only · Natural Highlights · Full Shot Sequence  
  (configurable pre/post-roll — not hardcoded)
- **Multimodal detection** — residual motion after camera compensation + table mask + audio support
- **Replay-aware** — replays flagged and excluded by default
- **Pre-analysis match editor** — split/delete frame breaks with a 1×–64× zoomable timeline; original uploads remain untouched
- **Review UI** — move boundaries, add/delete shots, mark replays, export labels
- **Selected-shots preview** — immediately play included shots as one continuous virtual timeline
- **Export** — individual clips, one combined MP4, CSV, EDL, training labels
- **Jobs** — progress, resume, batch CLI
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
snooker-ai analyze input.mp4 --mode natural
snooker-ai review <job-id>
snooker-ai export <job-id> --output highlights.mp4
snooker-ai batch ./matches --mode action-only
snooker-ai evaluate ./test-dataset
snooker-ai train ./training/dataset_config.example.yaml
snooker-ai serve --port 8000
```

Strict mode starts at the first confirmed cue-ball launch minus 2.000 seconds and
normally ends at the first physical all-ball stop. The 0.50-second stationary
confirmation is look-ahead evidence only and is not included in `clip_end`.
Unresolved motion is capped ten seconds after the strike and flagged for review,
so a false track cannot produce a 40–50 second clip.

Modes: `strict` (default — 2s before strike → balls stop) | `action_only` | `natural` | `full_sequence`

## Web UI

```bash
snooker-ai serve --host 127.0.0.1 --port 8000
```

- Upload: http://127.0.0.1:8000/
- Review: http://127.0.0.1:8000/review/`<job-id>`
- API docs: http://127.0.0.1:8000/docs

Selecting a video opens the pre-analysis editor. Split at both edges of any
between-frame break, delete the middle section, zoom the timeline as needed,
then start analysis. A cleaned MP4 is created from kept sections without
modifying the original upload.

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
