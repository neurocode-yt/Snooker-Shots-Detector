# Pots + fouls action feedback — 15 October 2026

The reported Holt–Burns button successfully submitted a separate score-selection
job on the running server. Its original all-shots row continued to show Ready,
while the new job appeared elsewhere in the library. Reusing an existing version
only added a small message. Neither case offered the requested CapCut prompt.

The library and editor now share a dialog that immediately acknowledges a click,
follows the returned job through queuing, score checks and export, and displays
its progress. Repeated submissions share the initial request; existing variants
remain managed by the server's reuse/resume logic. Closing the dialog stops its
polling while processing continues on the server. A later click follows the
existing variant.

Completed reuse displays “Pots + fouls highlights already made.” New completion
displays “Pots + fouls highlights are ready.” Both ask “Open in CapCut?” and
provide download and editor links for the returned variant ID. CapCut opens only
after that button is clicked. Handoff failures remain visible and retryable;
closing and duplicate handoffs are blocked during project preparation. The
editor keeps its current match instead of redirecting to the library.

Network/server failures report their reason. No scoring shots or inconclusive
scores stop progress tracking and offer the variant's editor. These outcomes
cannot wait indefinitely for a nonexistent export or offer an empty CapCut handoff.

Validation:

- JavaScript suites: 25 passed, including nine creation/reuse/progress regressions.
- Python conversion, selection, CapCut and queue suites: 119 passed.
- JavaScript syntax and Git whitespace checks passed.
- Browser checks against the actual library followed Holt–Burns processing and
  reused the completed Lilley variant, showing the requested prompt. The editor
  showed the same dialog and retained its URL. Links pointed to the selected
  variant rather than the original all-shots job.
- An isolated browser fixture submitted a new job, followed it to the ready
  prompt, and created a one-shot CapCut draft through that prompt. The draft
  belonged to the selected variant; the parent received no CapCut export.
  The fixture used a temporary project directory and a launcher hook, leaving
  the user's CapCut project index untouched.

The updated server on port 8008 detects installed CapCut 9.5.0.4050. The older
server on port 8007 continues its existing Holt–Burns work; its attempted restart
was blocked by automatic approval review, so the updated server was started
separately without interrupting processing.
