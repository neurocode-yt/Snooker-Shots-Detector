# Highlight status labels

The library, shot editor and Shorts setup use automatic completion labels.
There are no required manual checking prompts, uncertainty badges on individual
shots or uncertainty filters in the editor.

- **Ready** means the finished video is available.
- **Shots ready** means detection finished and the saved shots can be exported.
- **Scores unavailable** means the scoreboard did not support automatic scoring.
- **Edits saved** means a new export is needed to include edited boundaries.
- **Needs attention** covers stopped work and saved edits awaiting export.

Unconfirmed score counts and source intervals remain factual information under
**Scoring details**. Detector confidence, timing evidence, selection rules and
saved outputs are preserved. These labels do not claim perfect detection.
Messages from older saved jobs are displayed using the same terminology.

Validation: 52 browser-side tests and 58 API, routing, export and scoring tests
passed. The live interface was checked with 270 saved shot records and the
269-shot Shorts setup; neither presents a manual checking step.
