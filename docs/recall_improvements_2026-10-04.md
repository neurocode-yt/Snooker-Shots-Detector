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

Audit outputs and source images are local under
`data/editor_audit/recall_2026_10_03/`; the source video is not distributed.
