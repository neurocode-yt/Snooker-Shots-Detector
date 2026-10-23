# Snooker AI for Android

The Android application packages the repository's **default detector, API,
review editor and FFmpeg exports** inside an APK. Import, proxy generation,
visual detection, score reading, review, highlight export and vertical Shorts
run on the phone. A PC, account, cloud API and internet connection are not
required after installing the APK.

## Install and use

Copy `dist/SnookerAI-Android.apk` to a **64-bit Android 10 or newer** phone and
open it. Allow installation from your chosen Files/browser application if
Android requests it. This is a development-signed APK, suitable for direct
installation; it is not a Play Store release.

1. Open **Snooker AI**. Its **Background & checkpoints** setup links to Android's
   battery exemption, notification and app settings. Enable unrestricted
   background use and notifications. The same setup remains available below the editor.
2. Choose **All shots**, **Pots only**, **Pots + fouls**, or **Edit shots**.
3. Choose a match using Android's document picker and start processing.
4. Review or adjust the saved shot timeline if needed. Use **Download** to pick
   a location for the MP4, ZIP or labels. After saving, **Share** opens Android's
   share sheet, including compatible installed video editors.

The interface, detection rules, thresholds, trims, score selection, mix
transitions and Shorts layouts come from the existing desktop source. Android
uses one match worker, one native contact worker, two OpenCV threads and a
smaller decode prefetch queue. These resource limits preserve the detector's
sampling settings. The default desktop detector is rule based and does not use
trained weights. Its source is shared with Android; this port does not replace
it with a smaller model. **Android uses the default detector only.** DL Algo's
source, models and PyTorch training runtime are excluded from the APK, and no DL
option is exposed. Computer GPU acceleration is not bundled. Windows CapCut's private desktop draft
format has no Android equivalent; use MP4 save/share on the phone.

Imported videos are copied into private app storage, keeping the selected
original untouched. Keep room for that copy, an analysis proxy and exports.
During import, temporary upload storage can require another source-sized copy.
Keep Android System WebView updated so the editor's modern browser controls
are available, including on Android 10.
App data persists across normal closes and updates, but uninstalling or clearing
app storage removes the private library. Save finished exports to shared storage.

Processing runs in a foreground service and displays a notification. It
continues when leaving the editor or turning off the screen while Android
permits the service to run. It requests a sticky service restart after an OS
process kill and automatically continues interrupted jobs through the existing
Continue routes. Manually paused work stays paused. After a reboot, a notification
opens the app to resume work; Android 15 prohibits launching this service from boot.
Force-stop requires reopening the app. Android 15 and later impose a
[six-hour background media-processing limit](https://developer.android.com/develop/background-work/services/fgs/timeout).
Long matches can take longer than their duration and warm the phone;
full-match speed and accuracy must be measured on the user's actual handset and
broadcasts. The Android dependency versions differ from desktop. Matching a
tested clip does not establish identical accuracy on every broadcast; use the
comparison check below before relying on a new device or input format.

## Processing checkpoints

Both the coarse scan and native confirmation windows save full streaming
detector state every **20 analysed samples or 15 seconds**, whichever is reached
first at a frame boundary. This includes tracking, flow, table masks, sampling
clock, context, feature prefix and the next frame. Completed stages and native
windows are also cached. Continue reuses completed work and seeks within the
unfinished window; it does not rerun the match from the beginning. Native Android
resume uses the original presentation timestamp, including for VFR footage.
A completed early-stop snapshot never decodes extra frames when restored.

Saves use atomic replacement and fsync. Before publishing progress, checkpoint
or edit files, Android keeps three checksum-verified local recovery generations
in a synchronous SQLite WAL index with immutable data objects. Startup repairs
damaged primary files from a valid generation. A recovery-save failure propagates
instead of silently skipping a detector window. Superseded generations are pruned.
The default detector's rules, thresholds and sampling rates are unchanged.

Only progress and edits receive these extra recovery copies; imported videos
are not duplicated by checkpoint protection. Keep original videos separately.
These copies protect against process termination and damaged individual saves,
and remain inside app storage: uninstalling or clearing data removes them.
Work after the last durable checkpoint can repeat. An unfinished proxy or MP4
render can restart that stage, while completed detection remains reusable.

## Build on Windows

Prerequisites: JDK 17, Python 3.10+, Android SDK 36, NDK `27.1.12297006`, and
`ANDROID_HOME`. The first build downloads its dependencies; the APK embeds them.

```powershell
.\scripts\build_android.ps1
# With an emulator/device already connected:
.\scripts\build_android.ps1 -Test
# Optionally install the resulting APK on a connected device:
.\scripts\build_android.ps1 -Install
# Select a particular 64-bit emulator when several devices are connected:
.\scripts\build_android.ps1 -Test -Install -Device emulator-5556
```

`prepare.py` copies only `snooker_ai` (excluding `dl`), `apps/api`, `apps/web` and `configs` into
ignored generated sources. It does not package `data`, `.venv`, model weights or
desktop uploads. Rerun preparation after changing shared source. Builds contain
ARM64 phone and x86-64 emulator native libraries. Build outputs and downloaded
dependencies are ignored by Git. Keep the development signing key to install
future updates over the same APK; production distribution needs a dedicated
release signing key.

## Implementation and verification

- Chaquopy embeds Python 3.10. The API binds a randomly assigned **127.0.0.1**
  port and requires a random, HttpOnly session cookie, including for media and
  downloads. No listener binds to Wi-Fi or other network interfaces. WebView
  restricts resources and navigation to the local app. External links open in
  the user's browser.
- Android's OpenCV wheel has no video backend. `mobile_capture.py` decodes with
  the packaged FFmpeg executable and streams BGR frames with bounded buffering.
  It explicitly uses desktop OpenCV's BT.601 conversion matrix, including for
  HD broadcasts tagged BT.709. FFmpeg's default metadata-aware conversion
  previously changed cue-ball identity and missed a real contact. The decoder
  derives presentation times from integer PTS and the stream time base rather
  than rounded log timestamps, preserving dense tracker refresh decisions.
  Colour parity, timestamp precision, VFR, seek and decoder failure behavior
  have dedicated host tests. Decoder revisions invalidate analysis checkpoints.
- Native executable wrappers reside in Android's read-only `nativeLibraryDir`,
  preserving normal subprocess pipes, exit statuses, timeouts and cancellation.
  The shared pipeline uses `SNOOKER_FFMPEG` and `SNOOKER_FFPROBE` when set.
- Pydantic 1's pure-Python implementation is packaged because Android wheels for
  Pydantic-core are unavailable. `snooker_ai/utils/models.py` adapts the small
  model API used here; desktop continues using Pydantic 2. Validation, saved
  models, end trims and Shorts tests are also run against this fallback.
- `OnDeviceSmokeTest` runs the real local server, rejects unauthenticated access,
  creates and seeks a video, runs the actual classic analyzer, exports a mixed
  highlight with audio, and renders a 1080×1920 Short with audio. Its synthetic
  fixture verifies runtime functionality, not broadcast detection accuracy.
  It also checks twelve held-out broadcast score readings at two resolutions
  using the Android OpenCV build. The test APK contains these fixtures; the
  normal APK does not.

For an independent desktop/Android comparison on a real clip, install the
application and test APK, then run:

```powershell
.venv\Scripts\python.exe scripts/check_android_parity.py path/to/clip.mp4 --device emulator-5558
```

Both environments analyse the same SHA256-verified source. The check requires
equal shot counts and contact/start/end times within 1 ms. Android uses a fresh
analysis directory, and the check fails if any shot or boundary differs. Reports
are saved under `apps/android/vendor/parity/`. The source fixture is transferred
only for testing and is not embedded in the distributed APK. This check covers
the default detector; it does not test DL Algo.

Runtime pins: Chaquopy 17.0.0, NumPy 1.23.3, OpenCV 4.5.1.48, SciPy 1.8.1,
Pillow 11.0.0, Pydantic 1.10.26, FastAPI 0.115.14 and Starlette 0.46.2.
These are selected for Android wheel compatibility. See
[third-party notices](../apps/android/THIRD_PARTY.md) for bundled media sources.
See the [validation record](android-validation.md) for the broadcast comparison
and its limits.

The manifest enables Android's page-size compatibility mode for older native
Python dependencies on devices with 16 KB memory pages. These wheels are not
rebuilt for native 16 KB alignment, so this APK is intended for direct install,
not a claim of Play Store native-library compliance. See
[Android's page-size documentation](https://developer.android.com/guide/practices/page-sizes).
