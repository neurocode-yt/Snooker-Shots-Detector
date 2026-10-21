# CapCut discovery recovery — 21 October 2026

The running server on port 8007 reported CapCut unavailable while the same
repository's fresh process could locate the accessible installation on D:.
The exact cause inside the old process was not established. Its response was
reproduced before an authorized idle restart activated this change.

Discovery now tries automatic sources even if a configured path has become
stale, reads Windows App Paths registration, and saves the last verified
executable locally. Each request revalidates the file. Saved roots are scanned
for updates before falling back to the saved version. Actual version executables
take priority over the shared launcher and incomplete update directories.
The path spelling recorded by CapCut's project index remains preserved.

The browser's opening action goes directly to the server, whose fresh discovery
decides whether it can proceed. An older or in-flight failed status check cannot
veto that action or overwrite a newer successful opening. Errors remain real
and retryable. Both discovery endpoints run file and registry work outside the
API event loop. The capability response exposes the verified version and paths.

Validation:

- Full Python suite: 1,664 passed, 17 skipped.
- Frontend suite: 54 passed. Targeted Python CapCut suites: 43 passed, 1 skipped.
- Ruff passed for the changed Python source and tests.
- The restarted port 8007 server returned `available: true`, version
  `9.6.0.4097`, and `D:\CapCutStorage\CapCut\Apps\9.6.0.4097\CapCut.exe`.
- The actual editor button created a separate 176-clip Pots + fouls project,
  ending in `bdc9562a`, from saved analysis without reprocessing the match.
  CapCut opened that named project with the source video visible and separate
  clips on its editable timeline (18:28 at 25 fps). Existing CapCut projects,
  including the previously edited draft ending in `9daa8d58`, were retained.
  The new draft uses saved analysis; it does not import edits made only inside
  an older CapCut project.

Availability is never forced true: regression coverage removes the saved
executables and confirms discovery fails, and covers stale configured paths,
missing registration, incomplete updates, direct version paths, in-flight browser
checks, duplicate clicks, and slow discovery while other API requests complete.
