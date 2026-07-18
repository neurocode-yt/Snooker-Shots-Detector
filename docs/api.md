# REST API

Base URL: `http://localhost:8000`

Interactive docs: `/docs`

## Endpoints

### `GET /health`
Liveness check.

### `POST /api/upload`
Multipart file upload. Returns `{ path, filename, size }`.

### `POST /api/jobs`
Start analysis.

```json
{
  "source_path": "C:/data/match.mp4",
  "mode": "natural",
  "resume": true
}
```

Modes: `action_only`, `natural`, `full_sequence`.

### `GET /api/jobs`
List jobs.

### `GET /api/jobs/{job_id}`
### `GET /api/jobs/{job_id}/progress`
Job status and progress `0..1`.

### `GET /api/jobs/{job_id}/shots`
Detected shots and durations.

### `PATCH /api/jobs/{job_id}/shots/{shot_id}`
Update boundaries / include / replay flags.

```json
{
  "clip_start": 10.0,
  "clip_end": 18.5,
  "cue_strike": 12.1,
  "included": true,
  "possible_replay": false
}
```

### `POST /api/jobs/{job_id}/shots`
Add a missed shot.

### `DELETE /api/jobs/{job_id}/shots/{shot_id}`
Delete a false positive.

### `POST /api/jobs/{job_id}/export`
Export clips + joined video.

```json
{
  "output_name": "highlights.mp4",
  "mode": "action_only",
  "accurate": true,
  "include_replays": false
}
```

### `GET /api/jobs/{job_id}/download/{kind}`
Kinds: `analysis`, `timeline`, `corrections`, `highlights`, `csv`, `edl`, `training`.

### `GET /api/jobs/{job_id}/video`
Stream source video for the review player.

### `GET /review/{job_id}`
HTML review interface.
