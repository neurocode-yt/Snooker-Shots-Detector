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
snooker-ai analyze path\to\match.mp4
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

## Open in CapCut

The editable-project integration supports Windows CapCut 9.5 and 9.6. Discovery
checks Local AppData, the registered Windows user folder, and CapCut's Windows
installation registration for both per-user/machine and 32/64-bit registry views.
Custom drive installations are supported without an AppData directory junction.
Registry uninstall commands are read only as paths and are never executed.
If the shared launcher is missing, discovery checks actual executables in version
folders, so an incomplete newer update directory cannot hide a usable version.
The project-directory spelling saved by CapCut is preserved across junctions.

If an already installed CapCut is reported missing, check
`http://127.0.0.1:8007/api/capcut` (use your server's port). After a code update,
the running Python process must be restarted to load the updated detector. Pause
and save active work first. A completed video does not need another analysis or
export: restart the server, refresh the page, and retry **Open in CapCut**.

For an unusual installation, the existing configuration overrides are:

```yaml
capcut:
  executable_path: 'D:\Your CapCut Folder\Apps\CapCut.exe'
  # Optional: use only when CapCut's own project index does not record this path.
  # projects_dir: 'D:\Your Projects\com.lveditor.draft'
```

Set these to the actual local paths and restart the server. The integration
creates a fresh named project, keeps the source media at its existing path,
backs up the project index and preserves previous drafts. It does not require
reinstalling CapCut just because a server cannot discover its executable.
