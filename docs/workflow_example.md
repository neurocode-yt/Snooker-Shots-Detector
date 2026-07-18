# Example workflow

## 1. Analyze

```bash
snooker-ai analyze match.mp4 --mode natural
```

Note the printed `job_id`.

## 2. Review

```bash
snooker-ai serve --port 8000
```

Open `http://127.0.0.1:8000/review/<job_id>`.

Keyboard:

| Key | Action |
|-----|--------|
| J / K | Next / previous shot |
| Space | Play / pause |
| I / O / S | Set start / end / strike from playhead |
| A | Apply boundary edit |
| R | Toggle replay |
| X | Include / exclude |
| N | Add shot at playhead |
| Del | Delete shot |

## 3. Export

```bash
snooker-ai export <job_id> --output highlights.mp4
```

Outputs under `data/jobs/<job_id>/export/`:

- `highlights.mp4`
- `clips/shot_0001.mp4` …
- `shots.csv`
- `timeline.edl`
- `training_labels.json`

## 4. Batch

```bash
snooker-ai batch ./matches --mode action_only
```

## 5. Corrections as training data

Review saves `corrections.json`. Export writes `training_labels.json` for
retraining pipelines (Phase 2+).
