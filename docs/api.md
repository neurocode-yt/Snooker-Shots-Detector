# REST API

Base URL: `http://localhost:8000`

Interactive docs: `/docs`

## Endpoints

### `GET /health`
Liveness check.

### `POST /api/upload`
Multipart file upload. Returns `{ path, filename, size }`.
The upload directory must have space for the complete video plus
`api.upload_reserve_mb` (256 MiB by default). Insufficient storage returns
HTTP 507 with a readable reason; an over-limit file returns 413. Storage is
rechecked during copying, failed copies are removed, and only complete files
receive a published video path. Copying runs outside the API event loop.

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
List jobs. Each job includes `selection`, `source_fingerprint` when available,
`has_saved_detection`, `can_create_variants`, and its own `download_url` when ready.
The library groups jobs by source path and fingerprint revision, preserving separate
versions and earlier runs. Different revisions of a replaced source stay separate.

### `POST /api/jobs/{job_id}/versions/{selection}`
Create `all`, `pots_and_fouls`, or `pots_only` from a finished classical job's saved
shots, without a new full-match scan. The source must still match its saved identity.
Score-based selections reuse cached score evidence. Returns `{job_id, status}`:
`started` queues a version, `running` identifies an existing active version, and
`ready` includes its `download_url`. Repeated requests reuse the same version while
the parent analysis, source, and relevant settings remain unchanged. Original job
files and manual review choices are preserved. An All shots version restores the
base selection before the score filter.

Filtered selection policy 6 also keeps each identified frame's opening and final
live contact, including non-scoring contacts. `pot_status` remains the actual
scoring classification. `shots[].evidence.frame_context` contains `frame`, `first`
and `last`; `pot_selection_summary` includes `frames`, `frame_context_shots`, and
`non_scoring_context_shots`. Frame numbers are relative to identified frames in
this source. Manual inclusion/exclusion overrides are preserved. The selection
policy participates in version identity, so an older finished selection creates
an updated version instead of returning its obsolete export.

`POST /api/jobs/{job_id}/points-highlights` remains a compatible shortcut for
`pots_and_fouls`. Version creation returns 409 for unfinished/busy jobs, missing
detection, or changed source media; an invalid selection returns 422. Export,
CapCut, Shorts, and editor actions use the returned version's own job ID.

### `POST /api/jobs/{job_id}/recheck-scores`
Recheck a completed `pots_only` or `pots_and_fouls` version using the current
reader and saved shot detection. Returns a separate version with the same response
shape and idempotent behavior as `/versions/{selection}`; the original is preserved.
All-shots jobs return 400. The pot selection summary includes `score_readings`
and `score_layouts`; zero readings means the scoreboard could not be recognized,
not evidence that all shots failed to score. Inconclusive selections remain
`ready_for_review` without exporting an empty video.

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

### `POST /api/jobs/{job_id}/trim-ends`
Edit completed shot endings without rerunning detection:

```json
{"seconds": 2, "long_seconds": 3, "long_threshold": 7}
```

Without `shot_ids`, this applies to included live shots. With
`{"seconds": 10, "shot_ids": [56]}`, it affects just shot 56. Long-shot duration
is measured from cue contact to the saved reference ending, excluding pre-roll.
Amounts and threshold must be finite numbers between 0 and 3600 seconds.

Offsets replace earlier settings instead of accumulating. Confirmed automatic
shots can change their existing automatic trim from the known usable stop;
uncertain or manually edited shots use their saved ending as the reference.
`{"reset": true}` restores the endings saved before the first end-trim edit;
`shot_ids` can restrict restoration to specific shots. Zero removes the manual
offset from the reference ending and can undo the automatic 2/3-second trim on
confirmed shots. Cue contact, minimum playback, source and next-shot limits are protected.
Physical stop evidence and scoring/review flags remain unchanged.

Returns `shots`, `changed_shots`, `limited_shots`, `affected_shots`, `settings`,
`edited_duration` and `export_needs_update`. Changed clips invalidate the current
download until a new export is made; the previous MP4 file is preserved. Invalid
settings return 422, missing shots return 404, and active work returns 409. A
single save updates the whole selection under the normal edit lease.

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

`GET /api/capcut` reports `{available: true, version, executable, projects_dir}`
or `{available: false, reason}`
for the local Windows CapCut installation. The current draft adapter supports
CapCut 9.5 and 9.6. Successful discovery remembers the verified executable locally;
each request checks that the file still exists. Missing configured or registered
paths fall back to other discovery sources, including the saved executable.
Discovery runs outside the API event loop so other requests stay responsive.

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
