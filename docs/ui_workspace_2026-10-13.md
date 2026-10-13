# Workspace and editor cleanup

The upload page previously mixed job IDs, internal progress messages and several
equally prominent actions. Every completed job displayed Continue, and the
classic editor opened two players while placing all export and analysis tools
in its header.

The workspace now provides four explicit highlight choices, a single file
picker with drag/drop, optional collapsed source trimming and a searchable
video library. Video titles lead each row; modes, status, date and selected-shot
counts distinguish versions. Status filters search the entire library, including
older pages. Active jobs show progress; interrupted jobs offer Continue; ready
jobs offer download and editing. Conversion remains one click. CapCut, restart
and technical details are available under More. Polling preserves open menus,
pending actions and keyboard focus.

The editor uses tabs for highlights preview and source trimming, pausing the
other player when the view changes. Shot filters and persisted checkboxes make
selection accessible without opening trim controls. Selecting a shot opens its
source view. Boundary inputs, save/include controls and less-used shot tools are
grouped separately. Filenames replace job IDs in the header; export tools and
recovery remain available in More. Long-match clocks include hours.

Review changes now mark the existing export as out of date. The old file remains
on disk, but its download is withheld until a fresh combined export succeeds.
Rendering disables edit controls to prevent changing the selection mid-export.
The detector, scoring policy and existing workflows are preserved; DL remains
hidden. Refresh the page for the new interface and restart the server after any
running analysis finishes to load the export-status change.

Validation: 1,282 Python tests passed with 16 skips, plus seven Node presentation
checks. Browser checks cover desktop and phone layout without horizontal
overflow, multi-word search, status filters, source selection and optional
trimming, editor filters, persisted inclusion, one visible player, a real H.264
export after edits and disabled editing during export. Responsive controls keep
download, editing, mute and fullscreen available on narrow screens. JavaScript
syntax checks and Python lint passed.

Run the presentation checks with `node --test tests/web_ui.test.cjs`.
