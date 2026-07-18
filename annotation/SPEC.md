# Annotation Specification

## Purpose

Create high-quality timeline and object labels for training and evaluation of
snooker shot detection models. **Never** split random clips from the same match
across train and test sets.

## Timeline labels

Annotators mark the following events on the match timeline (seconds, frame-accurate when possible):

| Label | Description |
|-------|-------------|
| `preparation_start` | Player begins approach / addresses the table for this shot |
| `player_down` | Player settles into the shot stance |
| `final_cueing` | Final feathering / backswing begins |
| `cue_strike` | Exact moment of tip–ball contact (or best estimate) |
| `first_ball_motion` | First visible ball movement after strike |
| `last_ball_motion` | Last meaningful ball movement before rest |
| `balls_stopped` | All relevant balls fully stationary |
| `reaction_end` | End of immediate reaction useful for highlights |
| `camera_cut` | Hard cut / dissolve boundary |
| `replay_start` / `replay_end` | Broadcast replay segment |
| `slow_motion_start` / `slow_motion_end` | Slow-mo portion |
| `advertisement` / `studio` | Non-play commercial or studio |
| `non_play` | Other non-action footage |
| `uncertain` | Annotator unsure — do not use as hard negative without review |

### Rules

1. Every genuine live shot must have a `cue_strike`.
2. Replays are labelled separately and linked to the live shot id when known.
3. Practice strokes / abandoned shots: mark `uncertain` or a dedicated `aborted_shot` note — not as `cue_strike`.
4. If contact is occluded, estimate strike time and set `confidence: low` on the annotation.

## Object labels (keyframes)

| Class | Notes |
|-------|-------|
| `table` polygon | Playing surface |
| `table_corners` | Four corners when visible |
| `cue_ball` | White ball |
| `object_ball` | Reds + colours (subclass colour optional) |
| `cue` | Cue stick |
| `player` | Player body |
| `replay_indicator` | Replay graphic / bug |
| `scoreboard` | Scoreboard / graphics region |

## Export format

```json
{
  "source_video": "match.mp4",
  "annotator": "name",
  "schema_version": "1.0",
  "events": [
    {"type": "cue_strike", "t": 123.45, "confidence": "high", "shot_id": 12}
  ],
  "objects": [
    {"frame_t": 123.40, "class": "cue_ball", "bbox": [x, y, w, h]}
  ]
}
```

## Negative examples (required)

Include clips of: camera pan over stationary table, player walking, referee
replacing balls, applause without strike, replay transitions, scoreboard
animations, feathering without strike, abandoned shots, ball close-ups,
advertisements.

## Quality control

- Dual annotation on 10% of matches; resolve disagreements.
- Match-level splits only.
- Hold out an edge-case set never used for training.
