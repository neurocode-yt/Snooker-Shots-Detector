# Close controls for activity and popups

The library and shot editor have a close button for each Current work entry and
for the entire Current work panel. Closing an entry only hides it; it does not
pause, cancel, delete, or change the task or its saved resources. Hidden entries
stay hidden during polling, after refresh, and when moving between the library
and editor on the same server address. **Show hidden work** restores them.
New task IDs still appear normally. If browser storage is unavailable, dismissal
works for the current page session.

Messages, warnings, CapCut status notices, and editor toasts have close buttons.
An identical dismissed message stays hidden on that page; a different message
can appear. **Show hidden notices** restores dismissed messages. Popup menus
have a close button, and project and pots + fouls dialogs have a corner close
button as well as their existing footer controls. Escape and close controls
remain usable during requests; closing does not cancel a request already sent.
Make Shorts and source previews retain their existing close controls.

Validation: all 65 Node frontend tests pass, including dismissal across reloads, polling updates,
cross-page synchronization, restoration, new tasks, and unavailable/corrupt
storage. Browser checks on port 8007 cover the interrupted Thepchaiya vs Hill
entry, individual and whole-panel dismissal, reload persistence, restoration,
library notices, project dialogs, and the editor menus and Shorts dialog.
