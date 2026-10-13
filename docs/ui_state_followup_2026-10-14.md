# Live editor state and export fixes

Editors opened before detection finished previously stayed at zero shots until
reload. A single status watcher now follows the active worker and loads the
completed timeline automatically. A saved-timeline revision distinguishes shot
changes from progress messages, so ordinary progress updates do not reload either
video. Reloading a shot list also preserves its current source playhead and view.

All review mutations now share busy/error handling. Replay edits invalidate the
visible download immediately, failed requests retain saved shots, and successful
structural changes refresh the list without restarting media. Library counts
reflect explicit selections and manually added/deleted shots.

Individual-clip export preserves the readiness of an existing combined video.
It still leaves an outdated combined export unavailable when review changes need
rendering. Review writes use the same process lock and OS lease as analysis and
export, preventing another tab or server instance from modifying their snapshot.

If an acknowledged structural edit cannot refresh its shot list because the
connection drops, the UI keeps the saved change, pauses further edits and retries
the refresh. It does not report the saved operation as a failed deletion or
silently operate on obsolete shot IDs.

Validation covers live completion in the browser, clip-only export, replay
invalidation and preserved playhead; API tests cover all review mutation routes
under a worker lease, combined-download freshness and saved-timeline revisions.
The classical detector and score classification policy are unchanged; DL remains
hidden. Restart the app server and refresh the page to load these changes.

The full suite passed with 1,291 Python tests and 16 skips; all nine Node UI
checks passed. The browser fixture transitioned from 78% processing with no saved
shots to a completed, downloadable timeline without reload. Clip-only export
created two real H.264 clips while retaining the combined download. A replay
edit then hid the outdated download and kept the source playhead at one second.
