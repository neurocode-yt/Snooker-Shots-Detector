# Recorded visual strike regressions

`visual_strike_windows.json` contains numeric observations extracted from the
user-supplied `software_check_2.mp4` (Aaron Hill / Panchaya Channoi). No media
is embedded. Timestamps are seconds into that source.

- 250.4: real launch with foreground activity in the sparse pre-window.
- 342.7: fast real launch whose coarse track association resets velocity.
- 325.664: walking around the table, incorrectly proposed by audio previously.
- 756.064: referee colour respot; the white remains stationary.
- 854.752: player preparation; the actual strike occurs much later.
- 864.3: real launch with subpixel pre-contact jitter and a brief logo identity swap.
- 936.0333: real launch with impact-time cue-ball identity disruption.

Tests compare silence and loud/changing audio values, and exercise both sparse
proposals and native visual confirmation. These focused cases are regressions,
not a representative accuracy benchmark.

`classic_hard/missed-safety-native.json` freezes native features around a
Robertson–Un-Nooh safety at 194.16–194.28 seconds. The accompanying
`missed_safety/` JPEGs are decoded source frames from 194.24–194.44 seconds;
they exercise table-coverage remeasurement when the white reappears after
occlusion. The coarse proposal at 194.535 seconds is also recorded because
that weak proposal must not suppress the native recheck.

`classic_hard/promotion/` contains source frames from the same match: the live
score strip at 590.20, the white title card at 608.00 and 608.50, and countdown
footage at 614.20, 616.20 and 618.20. These stills check that the promotional
countdown is excluded while the real match scoreboard prevents that context
rule from suppressing an ordinary live contact.

`classic_hard/saudi/` records source frames from O'Sullivan–Trump under green
arena lighting. The four addressing frames locate the actual white ball, the
referee frame checks that a head above the table stays outside its mask, and
the opening/closing gold chevrons establish a bounded replay package. Ordinary
gold cushions and live close-ups supply no wipe evidence. `source.json` pins
the source SHA-256, original timestamps and the 960×540 JPEG transform.
Its `closeup-departure-native.json` retains measured native observations of
the nearby-red contact and a separate referee reset. The white was still while
other-ball tracks were noisy; cue approach is measured in the preceding frame.
