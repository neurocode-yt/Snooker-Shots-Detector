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
  "mode": "strict",
  "resume": true
}
```

`strict` is the only mode; legacy names (`action_only`, `natural`, `full_sequence`) coerce to strict.

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
  "mode": "strict",
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

## CapCut Desktop handoff

`GET /api/capcut` reports `{available, version}` or `{available: false, reason}`
for the local Windows CapCut installation. The current draft adapter supports
CapCut 9.5.

`POST /api/jobs/{job_id}/open-capcut` creates a fresh CapCut draft from the saved,
included live shots, registers it in the local project list, and launches CapCut.
The response includes `project_name`, `project_path`, `shot_count`, `duration`,
`capcut_version`, `launched`, and `message`. Select the named project on CapCut's
Home screen to edit the assembled timeline. Source media is referenced in place;
no clip rendering is required. Every click creates a separate project.

A launch failure returns the already-created project with `launched: false`
and `launch_error`; start CapCut manually. Missing analysis returns 404, an empty
selection or invalid source boundaries returns 400, and an unavailable CapCut
installation or busy job returns 409. Cross-site browser origins return 403.
