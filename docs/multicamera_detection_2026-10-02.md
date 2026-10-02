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
These newly found source cases remain under verification; the earlier export
must be regenerated after those checks. Native stop discovery is distinct from
an inferred return-to-table upper bound. Known unusable spans also constrain
preparation footage, so the two-second lead-in cannot restore referee handling
or a foreign table.
