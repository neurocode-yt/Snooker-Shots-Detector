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
- Six host decoder tests passed, including frame-exact HD colour matching,
  timestamp precision, VFR, nonzero source start, seeking and missing input.
- Desktop model/API/export, saved trim, upload, queue, pot/score selection and
  editor JavaScript regression suites passed. The pure-Python Pydantic fallback
  was also tested separately against validation, review and export contracts.

This is evidence for the tested runtime and broadcast excerpt. It is not a
measurement of full-match recall on physical phones or a guarantee that every
shot in every broadcast will be detected. Native OpenCV and NumPy versions
still differ between the desktop environment and available Android wheels.
The APK targets 64-bit Android 10 onward; the device checks above used Android
15. Native 16 KB page compliance and Play Store distribution are not certified.
