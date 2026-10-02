# Visual detection across broadcast camera angles

The editor treats a snooker shot as one temporal event that can contain several
broadcast camera views. A cut resets image coordinates and view-local tracking;
it does not create a strike or demonstrate that the balls stopped. Detection
uses visual evidence. Commentary and audio peaks do not influence strikes or
stopping times; the source audio remains available for export.

## View geometry and shot continuity

`scene_detection/table_context.py` separates locally visible cloth from a fully
enclosed playing surface. Four plausible cloth boundaries provide a perspective
transform for normalized table coordinates. A cropped close-up can still provide
cue and ball motion, but cannot supply a complete table layout or prove that all
balls are stationary. Geometry is measured for the current view rather than
carried unchanged across a cut.

The stop detector retains the ongoing moving-shot state through camera cuts,
reaction shots and incomplete views. Unknown intervals interrupt stillness
confirmation. If the table returns already stationary, the first reliable full
view provides an upper bound on the stopping time; the system records that the
exact stop was unseen. It does not label the camera cut itself as the stop.

Strike confirmation combines a previously stationary white ball, subsequent
launch and sustained coherent movement with visible cue geometry where available.
Short impact occlusions can produce a bounded contact estimate before the ball
is reacquired. Cross-view contact estimates carry explicit inference evidence
and uncertainty. They are not proof that the hidden contact frame was observed.
Native refinement recomputes these inferences from its observations. A sparse
proposal cannot retain confirmation when its native window is foreign or
unobservable. A white that returns stationary and launches later belongs to
that later shot, rather than confirming a hidden impact in the previous view.

## Ball handling and match context

The interaction detector looks for a broad neutral glove contacting an object
ball or its recently visible position, sustained within one view. A bridge beside
the white ball alone is insufficient. Strong cue-address evidence suppresses the
handling gate. Bare skin contact remains unknown because colour thresholds alone
cannot reliably distinguish a player's bridge from somebody replacing a ball.

Confirmed handling cannot seed a shot or supply ball-stop evidence. If handling
interrupts unresolved motion, it can supply a practical edit boundary with low
confidence, rather than a fabricated physical stop. Such a boundary can leave
less than the requested minimum footage when sufficient usable source is absent.
Padding after an already confirmed stop also ends before sustained handling or
a confirmed foreign-table interval; satisfying viewing time cannot reintroduce
those actions. The estimated physical stop remains separate from that source
availability limit.

`scene_detection/broadcast_context.py` learns the stable outer name areas of a
recognizable lower score overlay. Sustained incompatible appearances identify
another-match cutaway; changing score digits are excluded from the fingerprint.
The guard compares image appearance, **not OCR text**, and does not read player
names or establish match identity from the filename. Missing overlays, unfamiliar
layouts and weak matches remain unknown. Native refinement inherits measured
coarse context so it cannot relearn a foreign cutaway as the target match.

## Replay association

Generic camera cuts, similar motion curves and identical scoreboard names are
insufficient to exclude a strike as a replay. Automatic geometric association
requires a recent live shot, a transition, strongly similar movement and repeated
normalized colour-ball positions before the strike and at multiple later times.
The repeated white-ball path must actually move; static colour spots do not
establish duplication. The fixed layout uses white, yellow, green, brown, blue,
pink and black; interchangeable red identities are not assigned fixed slots.

Explicit trusted replay annotations remain usable. Unmarked replays shown only
through partial views may lack enough geometry for automatic verification.
The current implementation is conservative about excluding these cases and
does not constitute a general trained replay-recognition model.

The supplied broadcast also uses paired pink/yellow concentric ring wipes.
The classifier records their central appearance only when both rings are
substantially complete against a dark background. An opening and matching
closing wipe, recent live play and a strike inside the bracket establish a
replay interval. Each wipe participates in at most one confirmed pair; live
shots between separate replay packages remain available. Confirmed intervals
also limit clip padding. Other broadcasters' graphics remain unverified.

## Editing and processing

The strict editing policy keeps a two-second lead-in and trims two seconds from
the observed physical stop, or three seconds when contact-to-stop duration is
at least seven seconds. Four clear seconds of footage and two seconds after
contact take precedence over that trim whenever sufficient usable source exists.
Transition padding protects the clear-footage minimum. Mix transitions occur
between separate shots; broadcast cuts inside a shot retain their source timing.
Source end, the following genuine shot or confirmed handling can restrict usable
footage; the editor does not invent additional play to satisfy a minimum.

A sparse first pass proposes candidate events. Detailed decoding runs around
contacts, stopping points and reacquisition; lower-rate tracking follows longer
rolls between those boundaries. Stable table masks, bounded refinement windows,
coarse caches and adaptive waiting skips avoid applying expensive analysis at
native cadence throughout an entire match. Camera cuts refresh view observations
immediately. Saved results are invalidated when the detection policy changes;
legacy feature files retain compatible default fields.

## Verification scope and remaining limits

The supplied Selby–Lisowski source was manually audited around opening play,
tight cue preparation and referee replacement. Independent image measurements
and enlarged ball sequences provide bounded contact and all-ball stop references.
The opening safety is an important regression: white becomes stationary near
193.8 seconds, while a red continues rolling and contacts another red near
194.2 seconds. Its all-ball stop is approximately 194.4 seconds. White-ball
stillness alone would incorrectly end the shot early.

This bounded audit is not a full-match accuracy measurement. Hidden contacts,
fully unseen stopping events, overlapping balls, compression shimmer, bare-hand
replacement and unfamiliar scoreboard layouts remain sources of uncertainty.
Diagnostic uncertainty fields are retained for compatibility; automatic analysis
and export do not require the user to approve every diagnostic flag. Exact
contact/stop times and a universal 100% accuracy claim are not justified when the
broadcast does not show the required visual evidence.

The first verified integration iteration passed 367 automated tests, Ruff and
Git whitespace checks. The source audit found all seven initial reference
contacts and rejected the stationary aiming zoom. Its 147-clip H.264/AAC export
and seven-shot preview decoded without FFmpeg errors. This count is an output
count, not a measured recall or precision percentage.

Additional source checks found missing forward stop coverage, long cutaways,
two broadcast replays and two missed terminal colour shots. The implementation
now revisits sparse quiet intervals before a later declared stop, resets scale
when cloth geometry exposes a missed camera cut, rejects a bridge identity
excursion back to the stationary white and bounds sustained non-table footage.
These source cases now have bounded native verification; the earlier export
is superseded by the consolidated audit. Native stop discovery is distinct from
an inferred return-to-table upper bound. Known unusable spans also constrain
preparation footage, so the two-second lead-in cannot restore referee handling
or a foreign table.

The second code iteration passed 377 automated tests and Ruff. Source-derived
numeric regressions recover the terminal black contact near 1058.2 seconds and
the shaded brown contact near 2139.8 seconds. Compact convex-envelope evidence
recovers a compressed shaded sphere without admitting the glove, skin and
colour-ball negative examples. Native source inspection also identified three
replay passages near 1760, 1809 and 2791 seconds. A subsequent live contact under the final closing wipe
is invisible and cannot receive an exact observed contact time.

The third code iteration passed 421 tests, Ruff and Git whitespace checks.
It recovers an already moving live white on return from a confirmed replay,
only when the preceding live shot has a verified earlier stop. The first visible
roll is recorded as a contact upper bound. A quieter table returning after a
wipe, a single graphic, a camera identity jump and low-confidence stale stop
cannot supply this recovery. The automatic clip starts after the replay span.

Two round red neighbours can independently explain missing cloth around a
compact ivory sphere in a crowded low view. Thin cue attachments are removed
only from elongated outlines, retaining the sphere's colour, shape and cloth
tests. Source regressions cover red near 912.43 seconds, colour near 924.96,
green near 2123.86 and a later red near 2870.25. A coherent launch from measured
quiet white-ball anchors can outvote player movement in aggregate cloth flow.
An excursion back to the original white during cue feathering is rejected.

Timeline rebuilding preserves explicitly edited inclusion decisions; it no
longer restores old automatic inclusion over newly confirmed replay exclusion.
Sustained non-table spans and cache coverage account for timestamp jitter from
25fps sampling of a 30fps proxy. Cache reuse still requires the requested
recorded observation rate, so cheap travel frames cannot replace native proof.

The consolidated source audit retained all 14 selected contact/visible-roll
references, rejected the aiming zoom and bridge identity excursion, and excluded
the four specifically inspected replays near 1760, 1809, 2791 and 5495.81 seconds.
It produces 130 included clips, with no included clip over 20 seconds in this
audit. The audit combines prior full-match observations with fresh bounded
windows; it is not a fresh full-match accuracy benchmark or a labelled recall
measurement. Full match detection across unseen broadcasts remains unmeasured.

The final 14-shot preview and 130-clip consolidated export were rendered at
1280×720, 25fps with H.264 video and stereo 48kHz AAC audio. Complete FFmpeg
decoding with errors treated as fatal succeeded for both files. Their audio
and video streams start at zero and have matching durations of 105.08 and
946.76 seconds respectively, exactly matching the planned export durations.
The combined preview and full export completed in 444.8 seconds on this
workspace; this is an export measurement, not a detection speed benchmark.

The full export uses 125 mixes of 0.24 seconds across 129 joins. Four joins
use a cut because an overlap would cover contact or reduce protected viewing
time. The source-limited black shot near 1058.129 seconds has 3.30 usable
seconds before a sustained broadcast cutaway; frame quantization preserves
3.28 seconds, including 1.28 seconds after the estimated contact. The editor
retains that shot rather than filling its minimum with unrelated footage.
All other included clips satisfy the clear-footage and post-contact minimums
within one frame of export quantization. Preview contact/end images were also
inspected. This validates the rendered timeline and selected source cases,
not detection precision or recall across the entire match.
