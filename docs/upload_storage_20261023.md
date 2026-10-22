# Upload failure on a full drive — 23 October 2026

The server started correctly. `POST /api/upload` then failed in `out.write`
with `OSError: [Errno 28] No space left on device`. G: had only 516,096 bytes
available. The old handler left a 679,477,248-byte truncated copy of the
1,566,624,618-byte Kyren Wilson–Pang Junxu source and returned an HTTP 500.

The partial copy had no saved job references and matched the original's first
64 KiB. It was moved to the user's `C:\Users\husky\Snooker-Recovery\20261023`
folder with an `.incomplete` extension. The original source, previous uploads,
saved jobs and CapCut projects were left in place.

The upload handler now:

- Checks the parsed file's size against the upload limit and available storage
  before creating a destination. The default free-space reserve is 256 MiB.
- Rechecks available storage as it copies, handling another task filling the
  same drive. POSIX disk-full/quota errors and Windows disk-full codes return
  HTTP 507 with a readable reason that the existing UI can display.
- Writes to a private `.part` file, closes it before publication, and removes
  that partial after failures, including write, flush, publication and read
  failures. Only complete uploads receive a returned video path.
- Copies outside the API event loop, allowing progress and health requests to
  continue while a large upload is being saved.

The preflight is after multipart reception; it checks the destination drive.
The upload reserve does not promise enough capacity for subsequent proxies,
analysis caches, or exports. An abrupt process shutdown can leave an unpublished
`.part` file; startup does not delete unknown files automatically.

Validation: 39 upload, export API, queue and configuration tests passed, including
16 upload cases for exact bytes, preflight rejection, working-space reserve,
concurrent storage consumption, write/close/publication disk-full failures,
Windows errors, incomplete/oversized uploads, cleanup, retry, preservation of
previous files and responsiveness during a slow copy. Ruff passed.

After storage became available, a new upload of the same match completed and
its queued job entered proxy generation. The active server was left running;
the safeguard can be activated at its next normal restart.
