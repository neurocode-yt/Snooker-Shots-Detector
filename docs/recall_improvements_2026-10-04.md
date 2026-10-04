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

Audit outputs and source images are local under
`data/editor_audit/recall_2026_10_03/`; the source video is not distributed.
