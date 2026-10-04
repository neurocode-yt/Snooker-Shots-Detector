# Recall and native-contact processing checks — 2026-10-04

The complete fresh analysis of the Selby–Lisowski source under commit `2c4fa77`
found 33 of 41 labelled live contacts, with two extra contacts, in the audited
intervals. The older delivered edit found 18 of those 41, with four extras.
These are bounded-interval results, not full-match accuracy. The complete fresh
run reused the video proxy only; it extracted all observations again.

The middle and late interval failures are now being used for development. They
must no longer be presented as independent holdouts for subsequent fixes.

## Native-contact prefetch

Native contact requests can run in two isolated workers. The parent still
confirms contacts, merges observations, and finds stops in chronological order.
Unknown or foreign match context, return-to-match corrections, and failed worker
requests use the synchronous path. Outstanding requests are bounded at two.
Set `analysis.native_contact_workers: 1` to use serial extraction.

Two real-source windows, 260–264.5 and 413–417.5 seconds, were extracted at the
source cadence of 25.000239 fps using both paths. All 226 serialized feature rows
matched exactly. The unprofiled benchmark took 14.995 seconds serially and 11.670
seconds through the actual prefetch helper (1.28× for these windows). This is not
a full-match speed measurement. The 92 related pipeline, resume, performance,
and scheduling tests passed.

Changing the worker count preserves feature and result signatures because it
changes scheduling only. Result policy 17 separately invalidates saved final
clips for the verified motion-onset correction in commit `b7ad706`.

## Partially hidden white balls

Low camera views can crop the white at a cloth-mask boundary, or leave only its
ivory hemisphere or crescent visible behind a red or pink. The detector now
retains a small horizontal mask margin and verifies the visible circular arc
against an independently measured round foreground ball. The tracker carries
that geometric evidence with the cue-ball identity. Ordinary skin, glove,
yellow-ball and rectangular-colour fragments cannot use this recovery path.

Fresh native extraction recovers the white in the three investigated source
sequences near 1770, 7593 and 7617 seconds. All 32 object/tracking tests pass.
These object observations alone do not establish final export recall; strike,
replay, and boundary checks still follow. Feature cache version 23 invalidates
older coarse observations and contact windows so existing jobs are reanalysed.

## Contact continuity and false detections

Slow launches can use a cue visible at contact even when it was hidden during
the preceding address. A separate native-cadence check reconnects a measured
two-frame white-ball departure to a nearby coherent roll after a brief
occlusion. It requires continuous observed footage through the hidden interval;
missing images, a camera cut, distant reappearance, or handling cannot prove
that sequence. The investigated 1770-second contact is recovered within its
source bracket of 1770.04–1770.08 seconds.

A single large identity jump followed by a stationary replacement cannot use
the tracker's decaying velocity as proof of a shot. A real rapid collision can
still pass with cue-contact geometry and independently moving object balls.
Repeated observations of the stationary white also contradict a claimed hidden
impact. Source fixtures reject the false events near 411, 1698 and 1717 seconds.

Fresh extraction of the earlier `software_check_2.mp4` regression retains the
145.5-second shot and produces no contact in 750–763 seconds, including the
reported referee replacement near 756 seconds. All 528 automated tests pass;
Ruff passes for the changed Python files. One almost entirely obscured contact
near 1632 seconds remains unresolved. Final pipeline recall and rendered output
require the fresh validation run; candidate-only checks are not final accuracy.
Result policy 18 also forces completed results from the intervening detector
commit to run the updated strike decisions, while preserving version-23 raw
observations. The 29 cache-resume and scheduling tests pass after that change.

Audit outputs and source images are local under
`data/editor_audit/recall_2026_10_03/`; the source video is not distributed.

## Corrected reference and second contact pass

The reference now has 49 contacts. A later source review found that an unmatched
prediction at 5044.86 seconds was a genuine live shot omitted from the original
seven-contact 5000–5300 reference. That original frozen file and its hash remain
preserved. The corrected interval and its soft-shot failure now count as
development data. Two late-frame timing brackets were also corrected from later
object-ball motion to the white's first departure. Only the negative arena-break
interval remains independent for the current iteration.

Against the same corrected 49-contact manifest, the earlier complete `2c4fa77`
analysis matched 40 contacts, missed nine and had four extras. The policy-18
bounded pipeline matched 47, missed two and had four extras. Neither result is
a full-match accuracy measurement. The bounded export decoded without errors,
but source/output inspection found referee footage in four clips, including two
genuine shot tails. It is not a final validated edit.

The next contact pass requires continuous evidence across the launch and retains
configured camera-cut boundaries when merging cached observations. One brief
invalid registration frame is allowed only after an already established roll;
it cannot bridge a cut. Repeated scale-consistent sightings of a stationary white
reject referee movement as an inferred hidden impact. A gentle, visibly addressed
stroke can use a longer continuous white-ball trajectory to confirm its roll.

On all 45,880 saved bounded observations, these rules remove the four known
extra contacts and add the missed soft shot at 5017.41 seconds without losing
previously matched reference contacts. This is a candidate-stage comparison;
the full pipeline and export must still be checked. Source fixtures include
referee, camera-pan, cut, identity-jump and interrupted-footage counterexamples.
Result policy 19 invalidates final timelines while retaining raw feature cache
23. The remaining almost fully hidden 1632-second stroke is still unresolved.
All 543 automated tests pass after this iteration; Ruff and whitespace checks
also pass. Rendering and source validation remain separate from these tests.

The subsequent normal policy-19 bounded pipeline matched 48 of those 49 contacts
with no extras. All 55 selected clips exported and fully decoded without errors;
minimum clear viewing time is four seconds and minimum unmixed post-contact time
is 2.24 seconds. Two referee tails remain, so this is still an intermediate edit.

## Address memory across a camera change

A stationary white and visible cue address in one view can now support a launch
hidden by the following angle. The new view must remain continuously observed
and quiet until the white reappears already rolling, with immediate visible cue
contact and a coherent trajectory. It never compares coordinates across cameras.
Missing images, extra cuts, replay, foreign-match footage, handling, stationary
reacquisition and identity switches reject this recovery.

Across 46,772 bounded observations, the only new candidate is the remaining
1632-second stroke; no candidate is removed. Its timestamp is first reacquisition
(1632.793), with explicit uncertainty back to the prior visible stationary white.
This does not claim exact contact timing during occlusion. All 569 automated
tests and changed-file Ruff checks pass. Policy 20 reruns final decisions while
preserving feature cache 23. Normal pipeline verification is still required.

A new 2800–2920-second source-only holdout has six live contacts, with native
frame brackets frozen before predictions were consulted. It is kept separate
from the 49 development contacts when reporting the next validation.
