# Source recheck of long Murphy–Day clips

The previous repair still contained five live clips longer than 15 seconds.
Every one was checked using independently decoded native source frames. The
19.4-second clip started 2.64 seconds before the actual stroke and retained a
referee reset. Another 16.34-second clip contained two strokes, because the
white-ball tracker briefly selected the player's bridge instead of the moving
white during the second stroke.

The default classical detector now recovers that brief interruption only when
there is a stationary measured white, prior cue contact, two coherent departure
frames, a short backward identity jump, and a continued measured roll on the
same launch ray. Replay, foreign-match, handling, missing-frame, remote-return
and motion controls remain required. An identity jump itself cannot create a
shot. No neural model or learned weights were changed.
Native refinement recomputes this proof from observations; saved confirmation
tags cannot approve a shot after its supporting evidence disappears.

A short native portrait/audience cut can now bound usable footage when a
continuous empty-camera interval is followed immediately by measured stillness
on the returning table. A reaction that returns to rolling balls cannot use
this shorter confirmation. Partial tables, sparse observations and unknown
returns retain the conservative existing behavior. This is an editing boundary,
not a claim about the physical stopping time of an unseen ball.

The separate portrait-pixel check now starts two seconds after cue contact.
Searching only the final twelve seconds of an already extended clip could miss
the first portrait and include preparation for the following stroke.
Result policy 61 invalidates prior completed-analysis signatures; existing
pixel-feature caches remain compatible.

## Observed outcomes

These are lengths in the repaired timeline, including the two-second lead-in
and existing configured end-trim behavior. They are not fixed duration caps.

| Previous shot number | Previous clip | New clips | Source finding |
| --- | ---: | ---: | --- |
| 43 | 16.34 s | 6.90 s + 4.48 s | Brown and blue are now separate strokes; cutaway removed |
| 56 | 19.40 s | 5.03 s | Actual strike at 1723.467 s; referee entry at 1726.533 s |
| 105 | 18.67 s | 5.10 s | Referee entry at 2756.267 s bounds the reset footage |
| 171 | 18.40 s | 12.33 s | Native tracking resolves the previously extended ending |
| 17 | 16.63 s | 16.63 s | About eleven seconds of real white-ball travel; uncertain end retained conservatively |

There are now 270 shot records, with **269 live shots exported** and the same
excluded replay. Pots + fouls selects **178 shots: 173 pots and 5 fouls**.
Thirty-six scoring classifications remain unknown. The unchanged 21 unresolved
scoring intervals still require review; this is not a claim of 100% accuracy.
Only the four source-audited records were replaced and the recovered blue was
added. Other accepted clips, exclusions and prior files were preserved.

Separate All shots and Pots + fouls versions are rendered through the ordinary
pausable export worker. Observed source scores are reused only after checking
the source fingerprint. Earlier exports remain available in version history.
The [machine-readable report](../benchmarks/classic_long_shot_recheck_20261017.json)
records the native bounds, durations, new versions and export checks.

Validation: **1,475 Python tests passed, 16 skipped; 50 JavaScript tests passed**.
Source regressions cover both strokes and the false early preparation contact.
Negative controls remove address evidence, initial departure, continued motion,
source continuity or return identity. A routing test now waits for the existing
asynchronous queue before asserting dispatch, removing a timing race in the test.

Restart the running backend before processing future jobs with these changes.
The new rendered versions already contain the source-checked boundaries.
