# Automatic detection and performance update

The web workflow now runs detection, cutting, and combined-video export after one
start request. Uncertainty flags remain diagnostic and never require review
approval. Source trimming and the shot editor are optional.

## Detection changes

- Cue approach cannot count as a strike while the observed white ball is still
  stationary. The impact-occlusion recovery remains available.
- Sparse confirmation requires a quiet white ball and quiet ball tracks before
  launch, rejecting cushion/rebound peaks during existing travel.
- Camera pan/zoom compensates the ball position histories as well as optical flow.
- Measured zero trajectory speed remains zero; raw centre jitter and codec shimmer
  cannot override a stationary track.
- Ball-size estimates reset when seeking or switching cameras.
- Genuine rolls can continue beyond seven seconds. The new safety horizon is
  60 seconds; a confirmed stop exits tracking early.
- Dedicated native-rate stop evidence survives later rejected proposals and
  independently initialized trackers. A truncated final shot keeps footage to EOF.

## Performance changes

- Native contact verification rejects false proposals before a long travel scan.
- Travel tracking runs at 10 fps; contact and stop boundaries retain native cadence.
- Supporting optical flow refreshes at 10 fps. Native-rate ball centres, camera
  compensation, and camera-cut validity continue updating on every sampled frame.
- Tracking assignment uses vectorized costs, coherent-speed calculations are
  cached per observation, and expired identities are removed.
- Existing observations and signed checkpoints are reused. Confirmed recovered
  shots avoid another full travel scan.
- Automatic export creates the combined MP4 directly, avoiding redundant encoding
  of individual numbered clips.

## Validation

The saved 120.033-second Mark Allen vs Thep sample contains five visually checked
shots. This update retained all five and rejected the mid-roll false strike found
during development. Four endings were confirmed by the detector; the fifth remains
in motion at the end of the source and exports to EOF without intervention.

The latest full feature/analysis run took **114.66 seconds** on the local RTX 3050
laptop with an existing analysis proxy, without a profiler. Proxy creation is
excluded from that measurement. A separate real API request resumed that analysis,
automatically rendered the combined video, and finished in **6.41 seconds**.

The resulting video is **51.967 seconds**, **1280 × 720 at 30 fps**, with audio.
Its duration differs from the requested sum of clips by less than 0.001 second.
Export validation proves the media and timing contract; it does not establish
perfect perceptual audio sync or exact physical stop labels.

All **157 tests passed**, with Ruff and JavaScript syntax checks also passing.
Validation covers the automatic API workflow, false rebound/approach
rejection, camera compensation, codec jitter, confirmed-stop preservation, bounded
tracking, legacy jobs, cache reuse, and Windows console Unicode output. A cp1252
logging traceback discovered during real export is fixed.

The five-shot sample is a smoke test, not an independent accuracy benchmark.
Precision, recall, and frame-exact boundary accuracy across unseen matches remain
unmeasured. No claim of 100% accuracy is made.

Local evidence is kept under ignored `data/editor_audit/`:
`automatic_summary.json`, `automatic_export_verification.json`,
`automatic_endings.jpg`, and `automatic_api_jobs/`.

Restart the server to load the changes. New jobs export automatically; API clients
can explicitly request analysis only with `auto_export: false`.
