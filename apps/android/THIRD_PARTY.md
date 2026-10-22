# Android runtime sources and notices

The Android build embeds third-party runtime components. Their license notices
remain in the Python package metadata. Build dependencies are fetched at build
time and included in the installed APK; the app does not fetch executable code.

| Component | Version | Source / license |
| --- | --- | --- |
| FFmpeg-Kit Extended video GPL bundle | v0.11.2-android | [Exact release and assets](https://github.com/akashskypatel/ffmpeg-kit-builders/releases/tag/v0.11.2-android), [source and build scripts](https://github.com/akashskypatel/ffmpeg-kit-builders/tree/v0.11.2-android); includes GPL codecs such as x264 |
| Chaquopy | 17.0.0 | [Source and licenses](https://github.com/chaquo/chaquopy/tree/17.0.0) |
| CPython | 3.10 runtime | [Source and PSF license](https://www.python.org/downloads/source/) |
| Android Python wheels | pinned in app/build.gradle | [Chaquopy package recipes](https://github.com/chaquo/chaquopy/tree/master/server/pypi/packages) |
| Pydantic | 1.10.26, pure Python | [Source and MIT license](https://github.com/pydantic/pydantic/tree/v1.10.26) |
| FastAPI / Starlette / Uvicorn | pinned in app/build.gradle | [FastAPI](https://github.com/fastapi/fastapi), [Starlette](https://github.com/encode/starlette), [Uvicorn](https://github.com/encode/uvicorn), MIT/BSD licenses |

`runtime-lock.json` records SHA256 digests for the media archive and the official
portable build-Python distribution. The Pydantic source archive is checked
against its PyPI SHA256 digest. The native CLI wrapper source is
`native/media_cli.c`. The media library itself is copied unchanged from the
listed release. Preserve the component notices and provide corresponding
sources when redistributing GPL-covered binaries.
