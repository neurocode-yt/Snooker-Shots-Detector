# Windows setup

## Prerequisites

1. **Python 3.10+** — [python.org](https://www.python.org/downloads/)  
   Enable “Add python.exe to PATH”.
2. **FFmpeg** with `ffmpeg.exe` and `ffprobe.exe` on PATH.  
   Options:
   - `winget install FFmpeg`
   - Or download a static build and add its `bin` folder to PATH.
3. **Git** (optional).
4. **NVIDIA drivers** only if you later enable GPU models (Phase 2).

Verify:

```powershell
python --version
ffmpeg -version
ffprobe -version
```

## Install

```powershell
cd path\to\snooker
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Quick test

```powershell
pytest -q
snooker-ai version
snooker-ai analyze path\to\match.mp4 --mode natural
snooker-ai serve --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Long videos

- Ensure enough free disk for proxy (~10–20% of source size) plus clips.
- Jobs resume from `data/jobs/<id>/analysis.json` if interrupted after analysis.
- Proxy generation is the usual bottleneck on CPU.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ffmpeg not found` | Install FFmpeg; restart terminal; check PATH |
| OpenCV import error | `pip install opencv-python-headless` |
| librosa / soundfile issues | `pip install soundfile`; install [VC++ redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) |
| Export fails on no-audio | Pipeline retries without audio automatically |
| CUDA not used | Phase 1 is CPU CV; GPU reserved for optional torch models |
