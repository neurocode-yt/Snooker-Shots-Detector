# Opening and final shots in frame highlights

Pots only and Pots + fouls now retain the opening and final live shot of every
identified snooker frame. These are context selections, not new pot labels.
Other scoring selection, existing clip boundaries and explicit user exclusions
are preserved. The library reports how many frame-context shots are included;
the editor labels opening and final shots. Older filtered versions have an
Update frame shots action that creates a separate export.

Stable zero-score transitions locate possible frame changes. A visually intact
red rack corroborates the next opening; a temporary zero overlay that returns to
the old score cannot create another frame. Existing independent cue contacts are
matched to the opening rack. This prevents the ball-placement clips before the
break from becoming the opening shot. The final live contact before the score
reset supplies the closing context, including a missed shot before concession.
Replays and explicitly unsupported cue contacts cannot become frame context.

A native fallback handles a camera cut at the opening contact. It requires six
quiet measured white-ball observations with visible cue address, a measured
departure in the first view, then four coherent moving white-ball observations
in the new full-table view. The two views have independent coordinate systems.
Missing tracking immediately after the cut is tolerated for at most 0.20 seconds.
Replay, foreign match, handling, sparse sampling and unstable camera observations
veto recovery. The normal segment and stopping logic builds the recovered clip.
The same fallback runs in the default classical detector without another full
video scan. No DL inference or training is introduced.

Selection policy is 6 and detector result policy is 65. Existing native score
observations still use scan policy 5; frame-context changes do not invalidate OCR
caches. The lightweight opening scan is restricted to zero-score intervals and
checkpoints each completed frame. Its cache is bound to source path, size,
modification time and interval ranges, and carried to saved highlight versions.

## Jimmy Robertson vs Steven Hallworth

The source contains six identified frames. The third opening was previously
missing because the broadcast cuts cameras immediately after contact. It is now
recovered at source **1512.76 seconds**, bounded by 1512.72–1512.76 seconds.

| Frame in source | Opening contact | Final contact |
| --- | ---: | ---: |
| 1 | 20.04 | 679.28 |
| 2 | 763.44 | 1425.32 |
| 3 | 1512.76 | 2301.24 |
| 4 | 2402.32 | 3090.28 |
| 5 | 3410.48 | 5279.88 |
| 6 | 5375.80 | 6721.84 |

The saved timeline contains 254 detected shots. Pots + fouls keeps **176 clips**:
159 confirmed pots, six fouls, and 11 additional frame-context shots. All twelve
opening/final contacts are included; one final contact was already a confirmed
pot. Every prior scoring clip retains its inclusion and source cut. Existing
exports and CapCut projects are preserved as earlier versions.

The updated Pots + fouls export is job `20261020-012750-dd3a4aa4` (1069.20 s).
All shots is `20261020-012804-1084498c` (254 clips, 1727.24 s). Both exports have
audio, match their frame-rounded render plans and serve HTTP range downloads.
All twelve source/export bookend comparisons exceed 0.99988 luma correlation.

The new CapCut project ends in **9daa8d58**. It has 176 editable clips and lasts
1100.68 s with hard cuts. All 165 prior segments preserve their source cuts and
editing properties; only timeline positions and shot numbers change. The saved
manual source trim at 3898.64 s remains 8.04 s long, now on shot 149.

Validation: **1657 Python tests passed, 17 skipped; 53 JavaScript tests passed**.
The 34 new Python regressions cover frame context, native camera-cut recovery,
incomplete proof rejection, repeated selection, exclusions, cache continuation,
the default detector and upgrading earlier saved versions. The checked report is
`benchmarks/classic_frame_context_20261020.json`.

Frame identification depends on readable score transitions and visible opening
racks. A clipped or unsupported broadcast may not expose those signals. This
change does not claim complete shot recall: the three previously reported
unresolved scoring intervals remain unresolved.
