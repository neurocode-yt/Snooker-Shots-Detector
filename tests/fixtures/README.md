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
