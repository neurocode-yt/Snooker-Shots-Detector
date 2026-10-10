# More original-mode tests and shot recovery

The original detector now rejects ivory forearm fragments in the main camera,
recognizes paired blue Championship League title wipes, rejects large identity
returns without physical launch evidence, and preserves a native-confirmed returned
white-ball roll when its uncertain contact time overlaps an unresolved previous
shot. DL stays hidden and its code and models are unchanged by this work.

The editing fix separates contact-time confidence from measured shot existence.
A hidden red was detected at 102.36 seconds but lost during overlap repair because
its 0.68 confidence was below the ordinary 0.70 independence threshold. Its native
confirmation and sustained coherent roll now establish independence while its
timing/end review flags remain. Weak preparation proposals cannot use that exception.

Full pipeline on the Robertson–Un-Nooh source prefix `[0,160)`:

| Contact/selection result | Before (`f16400e`) | Final |
| --- | ---: | ---: |
| Matched live contacts | 9/10 | 10/10 |
| Missed contacts | 1 | 0 |
| Extra live contacts | 4 | 0 |

The rendered result contains ten selected live clips and covers all ten complete
source contact brackets. Full decoding passes. The detector retains manual review
flags: physical stop accuracy and clean referee-free endpoints were not independently
labelled or established by these contact results. The example is a 102.96-second
export from the 160-second source, not a claim of optimal highlight trimming.

Four additional minutes from three new underlying match groups were reviewed
chronologically at 1 fps, then checked at source-native contact frames: Trump–Un-Nooh
2019, Robertson–Ursenbacher, and Hill–Channoi. Their labels were frozen before shot
predictions were read. The first assessment reported 9 matches, 2 misses and 4 extras.
Native source adjudication exposed two annotation timing errors: the Trump black
impact was 286.84–286.88, rather than a later collision at 288.08–288.12; the Robertson
black stayed stationary after a camera cut and was struck at 817.52–817.60, rather
than 817.16–817.28. The original labels and first-test report remain intact. Both
adjudicated match groups are now development cases. Hill–Channoi remains a reserved
test group with two contacts; no rules were selected from its predictions.

Current rules find all eleven adjudicated contacts in those four minutes. One
extra at 230.92 is an alternate-camera ITV red replay with a different graphic;
that replay style remains unresolved. The finger-highlight false event is removed
without losing the following real red. The fourteen contacts from the previous six
sections were freshly extracted with the new original rules and remain 14/14 with
no extra contacts. These are Codex-reviewed labels, not human-approved ground truth.

Validation: 879 original-mode tests pass; Ruff and whitespace checks pass. New
source regressions cover foreground rejection, the genuine white beside it,
returned-roll recovery, blue wipe pairing, rejection of unrelated blue screens
and mixed graphic families, finger identity swaps, and overlap repair. A wheel
build verified inclusion of the 972-byte binary graphic-title template. The
template is a hand-built visual reference derived from source pixels, not a DL
model. A matching title and fullscreen field are required; blue colour alone is
insufficient, and both opening and closing wipes plus preceding live play are
needed for automatic replay exclusion.

Native benchmark extraction now includes one second of lookbehind so a contact
at the start of a reviewed section has measured pre-contact context. Evaluation
still counts only the labelled window. Feature cache version 35 and result policy
40 invalidate obsolete observations and clip selection for new analyses.

Tracked evidence:

- `benchmarks/classic_library_more_20261010.json`: first source-only labels.
- `benchmarks/classic_library_more_20261010_adjudicated.json`: explicit additive timing corrections.
- `benchmarks/classic_more_tests_20261010_results.json`: measured results, limitations and rule hashes.
- `tests/fixtures/classic_library/`: bounded original-source regressions.
- `snooker_ai/scene_detection/assets/championship_title.png`: packaged binary title template.

Local raw audit evidence, the unchanged first assessment, and the rendered sample
are under `data/editor_audit/original_2026_10_10/more-tests`. The video is
`prefix-export/highlights.mp4`. The latest code is available in a separate local
preview at `http://127.0.0.1:8003/`; the already-running 8002 process was preserved.
Restarting a normal server loads the updated rules. These results cover reviewed
sections and a known-failure prefix; the remaining full match library and future
footage have not reached verified perfect accuracy.
