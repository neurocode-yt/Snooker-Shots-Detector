"""Prepare reproducible Android inputs without copying user videos or models.

Usage: python apps/android/prepare.py [--sdk PATH] [--ndk PATH]
Downloads are build-time only. The installed app never downloads its runtime.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = HERE / "vendor"
MEDIA_URL = ("https://github.com/akashskypatel/ffmpeg-kit-builders/releases/download/"
             "v0.11.2-android/bundle-video-shared-small-gpl-release.aar")


def download(url, name, expected=None):
    path = VENDOR / name
    if not path.exists():
        print(f"Downloading {name}", flush=True)
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"SHA256 mismatch for {path}")
    print(f"{name}: sha256 {digest}", flush=True)
    return path


def prepare(sdk, ndk):
    VENDOR.mkdir(parents=True, exist_ok=True)
    lock = json.loads((HERE / "runtime-lock.json").read_text())
    media = download(MEDIA_URL, "bundle-video-shared-small-gpl-release.aar", lock["media"])
    native = HERE / "app/src/main/jniLibs"
    clang = ndk / "toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
    with zipfile.ZipFile(media) as archive:
        for abi, triple in [("arm64-v8a", "aarch64-linux-android26"),
                            ("x86_64", "x86_64-linux-android26")]:
            folder = native / abi
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "libffmpegkit.so").write_bytes(archive.read(f"jni/{abi}/libffmpegkit.so"))
            subprocess.run([str(clang), f"--target={triple}", "-O2", "-fPIE", "-pie",
                            "-Wl,-z,max-page-size=16384", str(HERE / "native/media_cli.c"),
                            "-ldl", "-o", str(folder / "libffmpeg.so")], check=True)
            shutil.copyfile(folder / "libffmpeg.so", folder / "libffprobe.so")

    python_zip = download("https://api.nuget.org/v3-flatcontainer/python/3.10.11/python.3.10.11.nupkg",
                          "python.3.10.11.nupkg", lock["python"])
    build_python = VENDOR / "python"
    build_python.mkdir(exist_ok=True)
    with zipfile.ZipFile(python_zip) as archive:
        for name in archive.namelist():
            if name.startswith("tools/") and not name.endswith("/"):
                target = build_python / name[len("tools/"):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    (build_python / "python310._pth").unlink(missing_ok=True)

    # Pydantic 1 supports pure Python; do not ship Windows extensions.
    info = json.load(urllib.request.urlopen("https://pypi.org/pypi/pydantic/1.10.26/json"))
    source = next(x for x in info["urls"] if x["packagetype"] == "sdist")
    tarball = download(source["url"], source["filename"], source["digests"]["sha256"])
    wheel = VENDOR / "pydantic-1.10.26-py3-none-any.whl"
    wheel_bytes = io.BytesIO()
    with tarfile.open(tarball) as archive, zipfile.ZipFile(wheel_bytes, "w", zipfile.ZIP_DEFLATED) as out:
        def put(name, content):
            out.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), content,
                         compress_type=zipfile.ZIP_DEFLATED)
        for member in archive.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) == 2 and parts[1].startswith("pydantic/") and member.isfile():
                put(parts[1], archive.extractfile(member).read())
        prefix = "pydantic-1.10.26.dist-info/"
        put(prefix + "METADATA", "Metadata-Version: 2.1\nName: pydantic\nVersion: 1.10.26\nRequires-Dist: typing-extensions>=4.2.0\n")
        put(prefix + "WHEEL", "Wheel-Version: 1.0\nGenerator: snooker-android\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        license_file = next(member for member in archive.getmembers() if member.name.endswith("/LICENSE"))
        put(prefix + "licenses/LICENSE", archive.extractfile(license_file).read())
        record = io.StringIO()
        writer = csv.writer(record, lineterminator="\n")
        for name in out.namelist():
            content = out.read(name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            writer.writerow([name, "sha256=" + digest, len(content)])
        writer.writerow([prefix + "RECORD", "", ""])
        put(prefix + "RECORD", record.getvalue())
    if not wheel.exists() or wheel.read_bytes() != wheel_bytes.getvalue():
        wheel.write_bytes(wheel_bytes.getvalue())

    generated = HERE / "app/src/main/python/generated"
    generated.mkdir(parents=True, exist_ok=True)
    for name in ["snooker_ai", "apps", "configs"]:
        target = generated / name
        # Only remove these fixed generated paths, never repository sources.
        if target.exists():
            assert target.resolve().parent == generated.resolve()
            shutil.rmtree(target)
        if name == "apps":
            target.mkdir()
            (target / "__init__.py").write_text("")
            for child in ["api", "web"]:
                shutil.copytree(ROOT / name / child, target / child,
                                ignore=shutil.ignore_patterns("__pycache__"))
            # Brand only the packaged mobile pages; retain the shared desktop UI.
            for page in (target / "web").glob("*.html"):
                page.write_text(page.read_text(encoding="utf-8").replace("Snooker AI", "Snooker App"),
                                encoding="utf-8")
        else:
            ignored = ["__pycache__"]
            if name == "snooker_ai":
                ignored.append("dl")  # Mobile deliberately runs the default detector only.
            shutil.copytree(ROOT / name, target, ignore=shutil.ignore_patterns(*ignored))
        if name == "configs":
            (target / "__init__.py").write_text("")
    (HERE / "local.properties").write_text("sdk.dir=" + sdk.as_posix().replace(":", "\\:") + "\n")
    print("Ready. Run gradlew.bat assembleDebug in apps/android.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=os.environ.get("ANDROID_HOME"))
    parser.add_argument("--ndk", type=Path)
    args = parser.parse_args()
    if args.sdk is None:
        parser.error("Set ANDROID_HOME or pass --sdk")
    sdk_path = Path(args.sdk)
    prepare(sdk_path, args.ndk or sdk_path / "ndk/27.1.12297006")
