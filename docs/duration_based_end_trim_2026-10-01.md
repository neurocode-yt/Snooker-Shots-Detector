# Duration-based shot endings

Shot duration means cue strike to confirmed physical all-ball stop. Pre-roll,
confirmation look-ahead, and the exported clip's duration do not affect the rule.

| Detected shot duration | Cut before physical stop |
| --- | --- |
| Less than 5 seconds | 2 seconds |
| 5 seconds to less than 7 seconds | 2 seconds |
| 7 seconds or longer | 3 seconds |

The user specified the under-five and at-least-seven cases. The intervening
range retains the existing two-second trim. The subsequent
[minimum-duration repair](minimum_shot_duration_2026-10-01.md) supersedes the
original 0.1-second impact clamp: four seconds of clear footage and two seconds
after impact now take precedence over end trimming. Unconfirmed EOF/duration caps are not treated as
physical stops, and physical-stop metadata is kept independently of edit timing.

The selected offset is persisted per shot, so serialization, overlap handling,
and export validation use the same boundary. Existing saved exports keep their
recorded policy. Restart and reanalyze to apply the new rule; mode changes
invalidate final results while leaving reusable detection caches intact.

Validation: 191 tests, including below/at/above the 7-second threshold, the 5–7
range, very short shots, long rolls, and serialization/export validation.
Rebuilt the previous five-shot demo from its saved detection evidence through
the updated segment builder: four shots use a three-second trim; the 6.967-second
shot keeps a two-second trim. The 1080p demo is 35.72 seconds with mix transitions.
