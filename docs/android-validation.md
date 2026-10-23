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
