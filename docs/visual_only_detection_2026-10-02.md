# Visual-only shot detection audit — 2 October 2026

The referee colour respot reported near 2:20 in the previous edit was a false
shot at source time 756.064 seconds. Its old clip covered 754.064–758.544.
An audio transient plus generic table movement bypassed cue-ball launch
confirmation, although the white ball remained stationary.

## Changes

- Remove audio proposal seeding, visual-support bypass, confidence weighting,
  overlap tie-breaking, and importance weighting. Commentary cannot influence
  shot selection. Preview and export keep original audio and audio crossfades.
- Stop extracting a separate WAV by default; feature extraction does not read
  audio. Regenerate visual proposals when resuming cached observations so old
  audio seeds and scores do not survive.
- Recover visual proposals from a quiet white ball despite foreground motion,
  and from consecutive centre displacement when the sparse tracker resets a
  fast ball's velocity. Every proposal still requires dense visual confirmation.
- Tolerate tightly clustered subpixel jitter and a single brief out-and-back
  identity swap during an otherwise coherent launch.
- Reject a candidate if sustained ball movement begins more than the configured
  0.75-second strike timeout later. A preparation event must not borrow movement
  from the next real strike or suppress that strike during overlap resolution.
- Retain the four-second clear-view minimum, two-second post-impact minimum,
  duration-based 2/3-second end trimming, and mix transitions.

## Verification

Source: the 1,732.467-second `software_check_2.mp4` uploaded in job
`20261001-191247-f7dc27f9`. The complete proposal/refinement pipeline was rerun
with cached raw visual observations reused where windows matched. Selection,
scoring and segmentation were recomputed; this is not a cold-speed benchmark.

The new edit contains 57 shots, lasts 370.328 seconds, and retains audio:

- Removed three old false strikes: 325.664, 756.064 and 854.752 seconds.
- Preserved the other 49 shots from the old edit.
- Recovered eight visually inspected real strikes: 145.500, 390.400, 632.600,
  670.967, 828.733, 864.267, 1550.100 and 1710.267 seconds.
- No exported interval overlaps source 754–758.5, the reported respot.
- Minimum clear duration after mixing: 4.000 seconds.
- Minimum post-impact visibility before the outgoing mix: 2.233 seconds.
- Render: 106.3 seconds (tests were running concurrently).
- Full FFmpeg audio/video decode with `-xerror`: passed without errors.
- 214 automated tests passed; Ruff and whitespace checks passed.

Contact sheets were inspected for all 52 previous detections and all newly
recovered shots; clip-end sheets were also inspected. Numeric regression
fixtures exercise actual stationary-white respot, walking/preparation, fast
launch, jitter, and impact occlusion. Silence and loud/changing audio produce
identical visual scores and detections. These cases are not an accuracy benchmark.

Local output and verification artifacts are under
`data/editor_audit/referee_respot_fix/`; large media remains outside Git.

## Remaining limitation

This change fixes the audio dependency and reported false respot, but does not
establish 100% match-wide accuracy. Source shot 145.5 demonstrates a separate
end-boundary limitation: tracking the white inside a red pack while other balls
settle can retain the referee's initial approach/reach near the clip tail. The
export ends at 151.5, before the subsequent placement, but still includes some
retrieval/preparation. Resolving those simultaneous action boundaries needs
stronger ball identity and handling evidence; a stationary white alone cannot
safely end a shot while other balls are rolling. No manual review step is added.
