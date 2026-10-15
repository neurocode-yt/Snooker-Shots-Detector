# Troubleshooting

## Repeated GET requests while no video is processing

These are browser status checks, not video processing. The library and Current
work panel check every 30 seconds when idle (including when all work is paused),
and every 2–2.5 seconds while work is active. Their automatic checks stop in
hidden tabs; returning to the tab or changing a task refreshes status immediately.
Separate open tabs each maintain their own status checks.

Successful status GETs are omitted from Uvicorn's access log. Failed requests,
actions such as Pause/Resume, and processing logs remain visible. Connection
failures retry after 10 seconds. Restart the server and reload existing browser
tabs after updating to apply this behavior.

## Analysis finds no shots

- Confirm the video shows a **green table** under typical broadcast lighting.
- Lower `strike_fusion.min_confidence` in config (e.g. `0.28`).
- Increase `analysis.sample_fps` for short clips.
- Check that proxy generation succeeded (`data/jobs/.../proxy/proxy.mp4`).
- Prefer **failing safe**: use Full Sequence mode and review UI to add misses.

## Too many false shots

- Raise `strike_fusion.min_confidence`.
- Ensure `replay.enabled: true`.
- Exclude segments in the review UI and export with `only_included`.
- Camera pans should be compensated; if not, raise `camera_motion.motion_magnitude_cut`.

## Balls cut mid-motion

- Increase `ball_stop.settle_seconds` and mode `post_roll`.
- Medium/low confidence shots already add `fail_safe_keep_extra_seconds`.
- Manually extend `clip_end` in review (key `o` = set end at playhead, then `A` apply).

## Audio out of sync

- Use **accurate** export (default), not `--fast`.
- Source VFR is handled via presentation timestamps (`-ss` on input).
- Re-export after correcting boundaries; do not stream-copy across concat if drift appears.

## Job stuck / interrupted

- Re-run `snooker-ai analyze ... --job-id <id>`. Completed analysis loads
  `analysis.json`; interrupted analysis resumes `coarse_checkpoint.json` and
  completed files under `refinement_cache/` when their fingerprints match.
- Refinement reports each bounded window instead of remaining at 85%.
- Delete `analysis.json` and use `--no-resume` to force full recompute.
- Check `job.json` for `error` messages.

## Docker

```bash
cd docker
docker compose build
docker compose up
```

Mount host videos into `/app/data/uploads`.
