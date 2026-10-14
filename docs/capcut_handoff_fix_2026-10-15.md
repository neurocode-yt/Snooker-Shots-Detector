# CapCut handoff fix — 15 October 2026

The running server on port 8007 returned `available: false` with “CapCut Desktop
was not found.” This disabled the action in both interfaces. The browser also
cached the first capability result indefinitely, so a failed check could not
recover in that page. A fresh process located Windows CapCut 9.5.0.4050 under
`D:\CapCutStorage\CapCut`, reached through the user's local app-data junction.
The precise environment difference in the existing server was not established.

Installation discovery now also reads the Windows registered Local AppData
folder and checks the user's conventional folder, resolves junctions, and can
use a version-directory executable when the root launcher is absent. An
explicit version executable retains its selected version. The integration
continues to reject unverified private draft versions.

The CapCut action stays clickable for idle, eligible jobs and selected shots.
Clicks refresh discovery and report any failure visibly. Brief capability
caching expires, duplicate handoffs are blocked, and editor changes stay locked
while preparing a project. The route reads the saved compact selection timeline
instead of loading all dense detection features. Original shot detection is
unchanged.

Validation:

- Python suite: 1,326 passed, 16 skipped, including 21 CapCut checks.
- Browser JavaScript suites: 16 passed, including four retry/cache/duplicate checks.
- Python lint, JavaScript syntax, and Git whitespace checks passed.
- In an isolated server and draft directory, library and editor buttons remained
  enabled after a simulated discovery failure and displayed its reason. After
  enabling discovery, clicking again without reloading created a draft containing
  the two included shots, excluding the third shot, and restored the controls.
  The launcher hook recorded the real detected executable and version; this test
  did not launch the desktop app or modify the user's CapCut project index.

Restart an existing Snooker AI server and refresh its page to load the update.
