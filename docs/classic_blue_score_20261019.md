# Blue pot excluded by an unreadable six

The blue reported around 10:02 in the Jimmy Robertson–Steven Hallworth CapCut
highlights was already detected at source time 3996 seconds. Its source clip
was 3994–3998.48 seconds. Pots + fouls excluded it because the pre-shot score
26 was unreadable; the post-shot score 31 was readable. The missing baseline
prevented a confirmed five-point increase.

The small native six has the correct topology and wins digit matching, but its
best overlap with the previous templates was about 0.766–0.777, below the
unchanged 0.78 confidence threshold. A manually labelled 12×8-pixel six from
source time 3988 seconds extends the existing fixed template bank. No neural
model is involved. Confidence, class-margin, stable-score, active-player,
replay and shot-boundary rules remain unchanged.

Six lossless overlay fixtures use different source frames from the labelled
sample. They check both scores and the turn marker at native and enlarged
resolution. The selection regression proves a stable 26 → 31 increase and
reproduces the unreadable baseline with the new exemplar removed. Cache
identity already hashes the digit asset; a regression checks that changed
templates invalidate old score caches, including empty observations.

The repair rereads scores from the unchanged source with the new templates,
reuses the saved default detector contacts and keeps every source clip range.
It creates a separate Pots + fouls export and CapCut project, preserving earlier
MP4s, drafts and the recovered red from the previous repair. All shots already
contains this blue and needs no new detection or export for this correction.

The complete score pass also recovers six other previously excluded pots. The
new selection has 164 clips (158 pots and six fouls), up from 157. Every earlier
selected shot remains selected. Six other previously unknown strokes now have
confirmed unchanged scores; the unknown count falls from 41 to 28. The new
CapCut draft retains the user's shortened shot 147 and all other saved segments.
Validation: **1,580 Python tests passed, 17 skipped; 80 targeted score tests
passed.** All 536 previous digit templates remain byte-for-byte intact; only
the labelled six is appended.

The source and export measurements are recorded in
[the blue repair report](../benchmarks/classic_blue_score_20261019.json).
The fixtures are a same-video temporal holdout, not a full-match accuracy
measurement. Restart the server to load the updated in-memory template bank
before future jobs or score rechecks.
