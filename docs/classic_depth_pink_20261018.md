# Pink omitted at 20:31 in CapCut

The 178-clip Pots + fouls project cut from a red to the next red at about
20:31, omitting the intervening pink. That pink is visible in the original
Murphy–Day source at approximately **2:07:48.7**. The white is hidden behind
it, and the pink travels straight toward the pocket and camera: its apparent
size increases rapidly while its centre barely moves. The ordinary circle
track also breaks as its compressed outline changes during that approach.

The default classical pipeline now has a bounded source-pixel recovery for
this perspective motion. An unresolved, observed score change can request it;
a score change alone cannot create a shot. Recovery requires a measured
coloured sphere resting across at least seven native images, followed by at
least four continuous images of coherent growth over at least 95 ms. Its
coloured area must grow by at least 60%, with a consistent centre, colour
family and compact outline. Native cadence, live match context, stationary
camera, uninterrupted scene and handling/replay vetoes remain required.

The first growing image bounds the hidden contact. The saved contact estimate
is **7668.583333 s**, with an upper bound of 7668.733333 s and pixel confirmation
at 7668.833333 s. This is an inferred contact interval, not a claim that an
unseen cue-tip impact was measured directly. The ordinary two-second lead-in
keeps the actual stroke in view.

The ball-stop stage now recognizes this verified size motion during its
measured interval. It does not invent a centre velocity or treat the pink's
disappearance as proof that all balls stopped. The ending still uses the
existing table-coverage, motion, referee and end-trim logic. Scene IDs can be
renumbered during feature merging; the motion interval must remain continuous
in the resulting scene rather than depend on an old numeric scene ID.

## Repaired version

- New pink clip: **7666.583333–7671.063333 s**, **4.48 s**.
- All shots: **271 records, 270 included live shots**.
- Pots + fouls: **179 clips: 174 pots and 5 fouls**.
- Existing 270 records keep their previous source boundaries and inclusion
  decisions. Prior exports and the previous CapCut draft are preserved.
- Recomputed unresolved score intervals: **19**. This change fixes the reported
  pink; it does not establish complete coverage of the match. The two red
  omissions documented in the previous mini audit remain unresolved.

Result policy **62** invalidates older completed detector results on a fresh
analysis. Pixel feature cache version **45** is unchanged: the recovery reads
the bounded native source interval and uses the existing observations as
context. No deep-learning code, weights or interface modes were changed.

## Validation

The regression fixture contains 24 consecutive, lossless source-image crops
and their measured frame context. Tests recover the actual pink, then reject
still pixels, a single size jump, missing rest, image gaps, sparse cadence,
camera/scene changes, handling, foreign context and replay. They also exercise
normal score recovery and segmentation and require the complete verified
motion proof. Native source checks reject the nearby stationary address and
referee reset. The earlier two close-up red windows are still rejected.

The full Python suite passed **1,504 tests**, with **17 skips**. The final suite
includes the motion-interval continuity guard and pausing coverage; all four
CapCut frontend tests also passed.
The corrected videos use the normal pausable export worker and reuse observed
source scores only after validating the source fingerprint.

Rendered output and CapCut verification are recorded in
[the machine-readable report](../benchmarks/classic_depth_pink_20261018.json).
The running API process loads the detector change on its next restart.

The completed All shots job is `20261018-022858-7b48f3a4`; the completed
Pots + fouls job is `20261018-023315-5fe56536`. Both exports have audio, match
the planned duration, and contain the recovered pink verified against a native
source image. The new editable CapCut draft is **Snooker - Shaun Murphy vs Ryan
Day - Pots + fouls - 59ac2792**. Its manifest has 179 ordered clips and places
the pink at 20:31.2. CapCut 9.6 opened and played the corrected draft; it was
left paused at the restored clip. The earlier `a910dce1` draft remains available.
