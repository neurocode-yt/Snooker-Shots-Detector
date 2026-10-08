# Zhao–Trump recall fix — 7 October 2026



The earlier cleaned Zhao Xintong–Judd Trump upload omitted 14 live contacts in

six inspected source intervals. Its delivered edit retained seven of the 21

labelled contacts. The intervals are development data selected around failures;

these results do not measure whole-match accuracy or an independent holdout.



## Detection changes



Native confirmation reads original source images at their own cadence, resized

to the proxy geometry. Each original frame receives circle proposals. A

canonical warmup and absolute refresh schedule reduce sensitivity to proposal

start times. Proxies preserve source cadence below the configured FPS ceiling.



White-ball observations retain measured shape quality and observation count;

a newly spawned track's zero velocity cannot prove that the white was at rest.

Established visible white identity is retained in supported full-table views.

Partial arcs can be recovered behind independently verified coloured balls,

including thin arcs and shaded/touching colours. The geometry, visible ivory

pixels and colour overlap must support the recovery together.



Contact recovery supports short measured addresses after a camera change,

slow coherent travel, and an addressed white disappearing before its roll is

reacquired. Hidden contact has an explicit interval; reacquisition does not

pretend to be exact impact. A touching-ball pot can use a verified coloured-ball

departure together with cue geometry and a small white departure.



Replay, handling, foreign-match, missing-source-image and camera-boundary

checks remain required. Native workers inherit the original source input.

Feature cache 29 and result policy 24 invalidate previous observations/results.

The settings apply to future analyses; earlier edits require reanalysis.



## Verification



The nine initial omissions recover under two refinement start offsets. Four

additional source regressions cover a brief camera address, a touching-ball

red, another occluded red and a colour contact behind pink. Fresh source

extraction in the separate Selby–Lisowski match rejects the aiming identity

swap, walking/referee and handling events, and retains the genuine blue shot.



All 711 automated tests pass, including original-frame cadence/seek tests,

source regressions and counterexamples. Ruff and whitespace checks pass.

The final fresh normal pipeline produced 45 records and 44 included shots.
The delivered 5:03.960 MP4 matches all 21 labelled contacts with zero omissions
and zero extras in the six development intervals. It fully decodes without
errors. All labelled contacts lie inside unmixed clip interiors; minimum clear
viewing time is four seconds and minimum unmixed post-contact time is 2.24 seconds.
Final evidence is under `data/editor_audit/zhao_trump_2026_10_06/`.


Source labels initially used approximate late ball-motion times. Native image

inspection corrected those departure brackets; the initial approximate file is

preserved under `before_fix/reference_initial_approximate.json`. The unchanged

old export still matches seven of 21 against the corrected, expanded reference.



The existing visible-hand ending test also had a geometry-warmup failure. It

now requires at least .4 seconds of continuous full-table coverage before entry,

while preserving cut, missing-context, cue, replay and foreign-match vetoes.



A universal guarantee for arbitrary unseen broadcasts is not established by

these bounded checks. Contacts hidden by the broadcast retain uncertainty.

## Follow-up on the newer 1,229.8-second upload

The user identified respotting at output 3:25 and 4:15, two missing reds, and
a black clip beginning after the pot. The newer source required separate checks.
Result policy 28 now checks all automatic clip endings for verified rail entry,
reduces optional transition padding before handling, retains an initial white
launch through a later collision, and recovers an interrupted white departure.
Spatial stillness rejects a rebound being treated as a new cue strike.

Delivered-output review also found that segmentation could erase the recovered
red using a stop beyond unusable footage, and that a closing replay wipe could
be reused as an opening when its replay contact was filtered. Both decisions
are corrected. Replay annotations now retain ownership so recalculation preserves
independent markers and clears prior inferred spans.

The normal pipeline produced 51 review records, five excluded replay contacts,
and **46 included clips in a 4:49.080 video**. Both missing reds and their next
blacks are present in order. The reported handling intervals are absent, and
the false late black entry is replaced by the earlier real contact. All other
previously included contacts are retained. Four seconds of clear viewing time
remain available per clip. Full FFmpeg decode and source/output image checks pass.

**750 tests, Ruff, and whitespace checks pass.** The updated app on port 8002
loads these fixes for future uploads. Detailed final evidence is in
`data/editor_audit/referee_followup_2026_10_07/final_report.md` and
`final_verification.json`. These checks verify the reported development cases;
they do not establish perfect detection on arbitrary unseen footage.

## Entire latest frame — 8 October 2026

The subsequent upload (source job `20261007-215124-f519210c`) is a separate
1,229.771-second Zhao–Trump frame. Whole-source review found 45 live contacts;
its old export contained 37. The eight omissions comprise a black, yellow,
pink, four reds and a soft safety immediately after a replay.

Feature cache 32 records measured per-frame camera transforms. Strike checks
compare positions in consistent camera coordinates, retain short/collision
launches and reacquired soft rolls, and reject preparation caused by zoom drift.
Table-plane registration can use supported coloured-ball patches. Source
regressions also cover the retained brown during zoom and red behind the bridge.

Result policy 34 prevents early ending trims from hiding supported object-ball
travel into a pocket, protects that outcome from the outgoing fade, and bounds
paired replay openings at their visible fade. Verified referee-entry limits
remain authoritative. No match timestamps or manually inserted shots are used
by runtime code.

The final output contains **45 included live clips in 5:01.880**, with one
excluded replay in the 46 review records. All 45 reviewed contacts match the
output once, with no missing, duplicate or extra clips. All contact brackets are
in unmixed clip interiors; minimum clear viewing is four seconds. The late red
pot is visible, the ending replay wipe is excluded, and final boundaries were
visually reviewed. Full video/audio decode has no errors.

**824 tests, Ruff and whitespace checks pass.** The updated app is on port 8002.
Final evidence is in `data/editor_audit/full_frame_2026_10_07/final_report.md`
and `final_verification.json`. This is a complete check of the supplied
development frame, with contact uncertainty recorded where visibility is lost;
it is not a guarantee of perfect recall on unseen footage.
