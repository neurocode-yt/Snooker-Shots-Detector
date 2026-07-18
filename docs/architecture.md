# Architecture

## Overview

Snooker AI is a modular offline video pipeline that detects genuine snooker
cue-strikes, estimates ball-stop times, removes inter-shot pauses, and exports
highlight reels. Analysis runs on a lower-resolution **proxy**; cuts use the
**original** source for quality and A/V sync.

```
Upload → Validate (ffprobe) → Proxy + audio extract
      → Frame sample → Table mask → Camera motion compensation
      → Residual motion + optional ball blobs → Audio onsets
      → Streaming scene/view classification → Strike fusion → Replay filter
      → Native-FPS refinement at strike/stop edges
      → State machine → Segment builder (mode A/B/C)
      → Review UI corrections → FFmpeg export (clips + joined)
```

## Package map

| Module | Responsibility |
|--------|----------------|
| `ingestion` | FFprobe metadata, validation, proxy generation |
| `scene_detection` | Cuts, view types (table / close-up / replay / …) |
| `table_detection` | Green-cloth mask, corners, optional homography |
| `motion` | Global camera affine estimate + residual optical flow |
| `audio` | Onset / band energy (supporting evidence only) |
| `object_detection` | Phase 1 blobs; Phase 2 learned detector hook |
| `tracking` | Prediction-based, label-aware ball tracks with normalized kinematics and occlusion state |
| `temporal_model` | Rule-based state machine (Phase 3: TCN/transformer) |
| `event_fusion` | Strike confidence fusion + ball-stop windows |
| `replay_detection` | Replay heuristics + signature matching |
| `segmentation` | Clip bounds per edit mode, overlap resolution |
| `rendering` | Accurate/fast FFmpeg cut + concat, EDL/CSV |
| `evaluation` | Precision/recall/boundary error reports |
| `jobs` | Filesystem job store, resume, corrections |
| `pipeline` | Orchestration |
| `apps/api` | FastAPI REST + static review UI |
| `apps/web` | Upload + timeline review frontend |

## Design principles

1. **Fail safe** — low confidence keeps extra footage and flags review.
2. **No audio-only shots** — audio weight is capped; motion/table required.
3. **Camera cuts ≠ shots** — residual motion after compensation drives detection.
4. **Swappable components** — each stage is a class with a clean interface.
5. **Config-driven thresholds** — `configs/default.yaml` owns magic numbers.

## Phases

- **Phase 1 (this release):** rule-based baseline, usable clips, full product shell.
- **Phase 2:** YOLO-style balls/cue/player, stronger tracking, better stop detection.
- **Phase 3:** learned temporal model on labelled timelines.
- **Phase 4:** active learning from review corrections.

## Job layout

```
data/jobs/<job_id>/
  job.json
  checkpoint.json
  coarse_checkpoint.json
  refinement_cache/window_XXXX.json
  proxy/proxy.mp4
  proxy/audio.wav
  analysis.json
  timeline.json
  corrections.json
  export/clips/shot_XXXX.mp4
  export/highlights.mp4
  export/shots.csv
  export/timeline.edl
  export/training_labels.json
```

`coarse_checkpoint.json` is fingerprinted from the source file and analysis
configuration. Interrupted refinement resumes completed dense windows instead
of repeating the full match scan. Stale caches are ignored automatically.
