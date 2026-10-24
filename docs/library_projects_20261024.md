# Library project names and resource deletion

A project is one source family in Your videos, with all its saved selections
and earlier runs. Family identity follows the library's source path, algorithm,
fingerprint and legacy-revision rules. Names live separately from progress and
analysis, so a processing update cannot overwrite a chosen name. The editor and
future CapCut handoffs use the saved name. Original filenames remain intact.

Deletion requires a resource preview and confirmation token. The server expands
the selected cards into their complete family membership and checks that the
membership, saved edits and CapCut indexes still match that preview. It rejects
active analysis, queued work, Shorts and CapCut handoffs using their OS leases.
Deletion markers prevent another server from starting a worker during removal.
CapCut must be closed when linked drafts are removed, so it cannot rewrite an
old project list over the deletion.

Each job directory is owned by its project. Linked CapCut folders are found by
their Snooker ownership manifests, including repeated exports and folders
renamed in CapCut. Removal updates CapCut's project index and preserves other
drafts. Only unshared uploads inside the configured uploads folder are eligible
for source cleanup. Other jobs and CapCut drafts retain shared source files;
external originals are preserved. Preprocessing receipts track uploaded inputs
for future cleanup; legacy input ownership is not inferred from filenames.

Resources are staged with a durable journal before the library entries and
CapCut indexes are committed. A staging failure restores the previous paths
and indexes. Interrupted staging can be recovered using resource cleanup;
committed removals with Windows sharing errors remain visible as pending
cleanup and can be retried. Staged folders are hidden from the job listing.

Validation uses temporary jobs, uploads, clips, Shorts and CapCut indexes.
Coverage includes shared media, external originals, full-family deletion,
renamed/repeated CapCut projects, active worker rejection, stale confirmation,
path and ownership rejection, rollback, recovery and cleanup retries. Browser
checks cover persisted naming, selection, resource preview and cancellation.
No production project was deleted during validation.

Final checks: 77 Python tests passed with 1 skipped; 34 web tests passed. The
exact staged changes were also tested in a separate copy of the repository.
CapCut index writes use an OS lease across server processes; rollback restores
only this deletion's entries and preserves newly registered unrelated drafts.
