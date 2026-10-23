# Android validation — 23 October 2026

The APK runs the **default detector only**, using the shared desktop analyzer,
rules, thresholds and segmentation code. DL source and weights are excluded.

## Broadcast regression

A 27-second, 1280×720, 30 fps excerpt from the existing Murphy–Day upload
(source interval 1280–1307 seconds) was analysed independently on desktop and
an Android 15 x86-64 emulator. Both analyses used CPU processing and one native
contact worker. Android ran in airplane mode.

Fixture SHA256:
`d54788a0bae85a394df067622f1d5648800137922d1283de9f93655decd33e17`

| Result | Desktop | Android |
| --- | --- | --- |
| Confirmed shots | 2 | 2 |
| First contact | 6.833333333333333 s | 6.833333333333333 s |
| First clip | 4.833333333333333–15.433333333333334 s | 4.833333333333333–15.433333333333334 s |
| Second contact | 17.433333333333334 s | 17.433333333333334 s |
| Second clip | 15.433333333333334–19.913333333333334 s | 15.433333333333334–19.913333333333334 s |

The original Android decoder found only the first contact. The HD source was
tagged BT.709; Android FFmpeg respected that tag while desktop OpenCV converted
with BT.601. Matching the desktop conversion produced byte-identical BGR frames
at the missed contact. Integer PTS parsing also removed rounded timestamps that
changed dense flow refresh timing. No detector thresholds were changed.

`OnDeviceSmokeTest#testBroadcastParity` passed in 340.367 seconds. It verified
the fixture hash, used fresh Android analysis storage, required equal counts,
and checked every contact/start/end against the desktop within 1 ms. The saved
values above were subsequently compared and were exactly equal.

The original emulator library job was then restarted through the normal app
interface. It independently produced the same two shots and identical saved
contact/start/end values, and completed its automatic 14.833333-second MP4
export with audio. The library displays two detected shots.

Reproduce a comparison on any clip using
`scripts/check_android_parity.py`; see [build and testing instructions](android.md).
The test video is not included in the APK or committed to the repository.

## Full original-video comparison (0.3.3)

The second case uses the entire original `match_test_1.mp4` upload: 538.656508
seconds, 1280×720 at 30 fps, 551,049,996 bytes. The existing desktop job
`20260719-003332-e9e7d385` contains 30 automatically detected contacts and no
manual edits. Source SHA256:
`4c1d95bdd50ed125d015093fc0c4b8e8f79f54bd57646e18ee07e741bc86077a`.

The original was imported through the Android document picker and processed by
the normal foreground service in airplane mode on Android 15 x86-64. Desktop
analysis was run independently using the current default detector, CPU feature
processing and the default Windows D3D11 video decode. Neither desktop
observations nor desktop detections were transferred to Android. A resumed
Android run reused only its own on-device proxy and checkpoints.

The completed normal Android job and corrected current PC result agree exactly:

| Check | Result |
| --- | --- |
| Current PC / Android detections | 38 / 38 |
| One-to-one matched detections | 38 |
| Missing / extra Android detections | 0 / 0 |
| Maximum contact timestamp difference | 0 seconds |
| Maximum clip start / end difference | 0 / 0 seconds |
| Older saved PC contacts represented on Android | 30 / 30 |

Every saved timestamp was compared, not only the totals. See the complete
[machine-readable comparison](validation/android-match-test-1.json). Inclusion
and replay flags also match. At 218.5 seconds, PC requests manual ending review
and Android does not; their stop confidence differs (0.87 versus 0.95), while
their actual saved cut is identical. Intermediate proposals and numerical
confidence metadata are not claimed to be identical across native libraries.

Both independent exports retain all 38 detection records and decode fully
without errors. Both contain 1280×720 H.264 video and AAC audio, approximately
203.798 seconds after pause removal and overlap merging. The Android automatic
export finished while the app remained in the background.

This larger case exposed three additional input differences:

- Windows' D3D11 decoder transfers NV12 surfaces. Direct planar YUV420-to-BGR
  conversion on Android uses a different swscale kernel, even with BT.601.
  Software NV12 conversion matched the PC's native BGR pixels at source times
  0, 20, 100, 200 and 500 seconds.
- Automatic x264 threading resolved to 12 frame/four lookahead threads on the
  reference PC and six/two on the emulator, changing the proxy's decoded pixels.
  Matching that setting produced identical SHA256 hashes for **all 16,159
  decoded 960×540 YUV frames** of the independently generated proxies:
  `03f7c6aa154d573123a42266754715dbef8ffb489fc6830b0388a8af8c3aa2aa`.
- Multiplying integer PTS by a rational before converting to float differs from
  OpenCV's multiplication by a floating-point time base. Matching OpenCV's
  operation removed one-ULP timestamp differences; 300 consecutive native
  frame timestamps were checked for exact equality.

The first current desktop run itself missed a contact at **285.566666667 s**
which the older desktop job had detected. Source frames show the white striking
a nearby red and reversing. Net displacement hid a strong measured initial
launch. The shared default detector now accepts this collision only with quiet
pre-contact observations, cue-address/contact evidence, a consistent initial
leg, sustained motion and independent object motion. Existing identity and
quiet/run gates remain required. A recorded numerical fixture exercises this
contact and rejects versions with missing contact, missing object motion or an
already-moving white. Result-policy revision 66 invalidates final results while
retaining compatible raw observations.

The corrected desktop result contains all 30 older contacts, with eight
additional detections. The shared detector flags the event at 43.033333333 s
for review; visual inspection shows white-ball placement by hand before the
break. Agreement between the platforms must not be interpreted as proving
that every returned event is a true shot.

The installed APK's SHA256 matches the tested build. All 75 packaged shared
Python source files match the repository byte for byte; no DL module is
packaged. The numerical dependencies still differ by platform, so matching
source code alone is not sufficient evidence of matching outputs.

The same full-video job also continued with the launcher in the foreground:
over a 58.7-second check it advanced from proposal 157 to 161 and wrote a newer
checkpoint. Updating to the shared collision fix resumed the existing job from
its source-293.5-second checkpoint and reused its completed local native windows.

The corrected code passed 342 selected classic detector, recall and negative
regressions (38.43 seconds), followed by 230 processing, checkpoint, pause/resume,
native-source, decoder, proxy and comparison regressions (40.38 seconds).
The 0.3.3 APK and instrumentation APK build successfully; Android lint reports
no errors (12 warnings).
The final APK's offline instrumentation passed in 21.496 seconds, including
real decode/seek, analysis, highlight and vertical export, native checkpoint
interruption/corruption repair and all twelve held-out scoreboard readings.

## Runtime and regression checks

- Android debug APK, instrumentation APK and Android lint passed.
- The final default-only APK's offline instrumentation test passed in 17.180
  seconds. It checked local server authentication, absence of DL code, default
  detector capability, real video decoding and seeking, actual analysis,
  a 7.8-second mixed highlight with audio, and a 1080×1920 Short with audio.
- Twelve held-out broadcast scoreboard readings passed on Android.
- The same offline test also passed on the Android 17 x86-64 emulator with
  16 KB pages (27.646 seconds), using the manifest's page compatibility mode.
- Background-service recovery was checked on Android 15: the service was
  stopped while the Activity remained in memory, then the existing Activity
  was brought forward. It started a new local server and restored the editor.
- Native document save preserved the exported MP4 byte for byte (SHA256
  checked). The Android share sheet received `video/mp4`, read permission and
  the correct saved filename. No file was sent to another person or service.
- Six host decoder tests passed, including frame-exact HD colour matching,
  timestamp precision, VFR, nonzero source start, seeking and missing input.
- Desktop model/API/export, saved trim, upload, queue, pot/score selection and
  editor JavaScript regression suites passed. The pure-Python Pydantic fallback
  was also tested separately against validation, review and export contracts.

## CapCut Android handoff (0.4.0)

Installed CapCut 13.6.0 from the APK linked by its official Android download
page on the Android 15 emulator. Completed CapCut's first-run setup and video
permission prompt. Used **Open in CapCut** on the normal 38-detection library
job from the full-video comparison above. No CapCut private project files were
written by Snooker AI; the app publishes completed clips under Movies and uses
CapCut's advertised local multiple-video import activity.

The actual editor opened with the source footage. Read-only inspection of the
project created by CapCut confirmed 38 distinct video materials and 38 separate
shot segments in source order, all with audio at volume 1. Each clip starts at
its own beginning and retains the rendered shot duration. Maximum difference
from the requested source interval was 0.026667 seconds, within one 30 fps frame.
CapCut appended its default two-second ending as a 39th segment; the project
duration is 214.257 seconds. Selection handles and clip editing controls were
inspected in the actual CapCut editor. See the per-clip
[validation data](validation/android-capcut.json).

Private content-provider URIs were rejected silently by this CapCut build, so
the production handoff publishes ordinary MediaStore videos. Saved media is
checksum-checked and reused on retry. It remains in Movies for CapCut projects
which reference it. The Android import has separate rendered clips; extending
a cut beyond its rendered footage requires changing the trim in Snooker AI and
opening another import. It does not provide desktop draft source handles.

Killed the Snooker AI process with SIGKILL after 24 of 38 clips were committed.
The sticky service restarted with a new PID, automatically resumed preparation
and finished all 38. All 24 committed clip hashes and modification times were
unchanged, proving they were reused rather than rendered again. Each rendered
clip and manifest are published atomically; a changed trim creates a new
generation instead of changing media used by an existing CapCut project.

Selected CapCut, on-device media, pause and recovery tests passed: 81 host tests
(one Windows directory-alias fixture skipped), eight JavaScript tests, and Android
build/lint. Host coverage includes real accurate seeking, first-frame comparison,
audio, duration, interruption, corrupt clips, saved trims and path validation.
Final Android instrumentation passed both the handoff and offline pipeline
tests, including read-only URI grants, traversal/write rejection, idempotent
MediaStore publication, decoding, checkpoint repair, exports and scoreboards.

## Checkpoint recovery update (0.2.0)

- Started a fresh normal queue job on Android 15 using the same broadcast clip.
  After a partial native window saved 20 observations with `next_idx=50`, sent
  SIGKILL to the app's UID-owned process while the app was in the background.
  This bypassed graceful pause/shutdown. Android restarted the sticky service
  automatically with a new PID. The same job resumed without user input,
  reused its proxy, 54 coarse observations and completed native windows,
  finished with the same two shots and exactly the contact/start/end values
  listed above, and completed its automatic MP4 export.
- The final Android offline instrumentation passed in 24.958 seconds. In
  addition to decode, analysis, highlight, Shorts and scoreboard checks, it
  interrupted a real native scan, deliberately damaged its primary NPZ,
  repaired it from local recovery storage, restored at frame 34 and required
  every resumed feature to equal uninterrupted extraction. This exercised the
  packaged Pydantic implementation, SQLite, fsync and Android FFmpeg seeking.
- Host interruption tests verify exact feature equality and a nonzero decoder
  seek for coarse and native windows. A terminal early-stop snapshot returns
  without decoding additional frames. Real FFmpeg tests verify exact resumed
  pixels and presentation timestamps for both CFR and VFR video.
- Recovery tests cover a damaged primary, damaged latest recovery generation,
  bounded three-generation retention, storage-full failure without replacing
  the previous save, explicit project deletion, and automatic Continue while
  preserving manual pauses and original export intent.
- Background setup, notification/battery settings links and checkpoint status
  were inspected in the emulator. Build and Android lint pass.
- All 208 selected host detector, pause/resume, checkpoint, native-window,
  score-selection and Android decoder regression tests passed (52.25 seconds).

This is evidence for the tested runtime and broadcast excerpt. It is not a
measurement of full-match recall on physical phones or a guarantee that every
shot in every broadcast will be detected. Native OpenCV and NumPy versions
still differ between the desktop environment and available Android wheels.
The APK targets 64-bit Android 10 onward; the device checks above used Android
15 and Android 17's page compatibility mode. Native 16 KB page compliance and
Play Store distribution are not certified.
