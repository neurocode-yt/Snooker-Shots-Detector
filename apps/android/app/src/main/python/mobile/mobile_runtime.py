"""Start the existing application inside the phone, bound only to loopback."""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import threading

_server = None
_thread = None
_config = None
_app = None
_url = None
_token = None


def start(storage, native_dir):
    global _server, _thread, _config, _app, _url, _token
    if _thread is not None and _thread.is_alive():
        if not _server.should_exit:
            return json.dumps({"url": _url, "token": _token})
        _thread.join(timeout=15)
        if _thread.is_alive():
            raise RuntimeError("Previous processing is still pausing. Reopen the app shortly.")
    root = Path(storage)
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    os.environ["SNOOKER_FFMPEG"] = str(Path(native_dir) / "libffmpeg.so")
    os.environ["SNOOKER_FFPROBE"] = str(Path(native_dir) / "libffprobe.so")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["SNOOKER_EMBEDDED"] = "1"
    import cv2
    from mobile_capture import FFmpegCapture
    cv2.VideoCapture = FFmpegCapture
    cv2.setNumThreads(2)

    from snooker_ai.config import load_config
    # extractPackages makes packaged assets accessible by ordinary Path reads.
    import configs
    cfg_path = Path(configs.__file__).parent / "default.yaml"
    _config = load_config(cfg_path, overrides={
        "device": "cpu",
        "paths": {"data_dir": str(root), "uploads_dir": str(root / "uploads"),
                  "jobs_dir": str(root / "jobs"), "outputs_dir": str(root / "outputs"),
                  "models_dir": str(root / "models")},
        "jobs": {"max_workers": 1},
        "analysis": {"native_contact_workers": 1, "native_contact_backend": "thread",
                     "hwaccel_decode": False, "decode_prefetch_frames": 2,
                     # Included in the shared checkpoint fingerprint so decoder
                     # corrections never reuse observations from older APKs.
                     "decoder_profile": "ffmpeg-bgr601-integer-pts-v2"},
        "api": {"cors_origins": []},
    })
    from apps.api.main import create_app
    from fastapi.responses import JSONResponse, Response
    from starlette.staticfiles import StaticFiles
    import uvicorn

    _app = create_app(_config)
    _token = secrets.token_urlsafe(32)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    _url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    assets = Path(__file__).parent / "mobile_web"
    _app.mount("/mobile", StaticFiles(directory=assets), name="mobile")

    @_app.middleware("http")
    async def private_phone_app(request, call_next):
        if (request.headers.get("host") != _url.split("//", 1)[1]
                or not hmac.compare_digest(request.cookies.get("snooker_mobile", ""), _token)):
            return JSONResponse({"detail": "Open Snooker AI on this phone"}, status_code=403)
        if request.url.path == "/api/algorithms":
            return JSONResponse({"algorithms": [{"id": "classic", "label": "Default detector",
                                                 "available": True, "reason": "Ready"}]})
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/review/"):
            content = b"".join([chunk async for chunk in response.body_iterator])
            content = content.replace(b"</head>", b'<link rel="stylesheet" href="/mobile/mobile.css">'
                                      b'<script defer src="/mobile/mobile.js"></script></head>')
            headers = dict(response.headers)
            headers.pop("content-length", None)
            return Response(content, status_code=response.status_code, headers=headers)
        return response

    @_app.get("/api/mobile")
    async def mobile_info():
        import shutil
        return {"on_device": True, "free_bytes": shutil.disk_usage(root).free,
                "desktop_capcut": False, "detector": "classic"}

    _server = uvicorn.Server(uvicorn.Config(_app, host="127.0.0.1", loop="asyncio", http="h11",
                                          ws="none", log_config=None, access_log=False))
    _thread = threading.Thread(target=_server.run, kwargs={"sockets": [listener]},
                               name="snooker-local", daemon=True)
    _thread.start()
    return json.dumps({"url": _url, "token": _token})


def ready():
    return bool(_server and _server.started and not _server.should_exit
                and _thread and _thread.is_alive())


def pause_all():
    if _config is None:
        return
    from snooker_ai.jobs.store import JobStore
    from snooker_ai.utils.work_control import WorkControl
    from apps.api.main import _ACTIVE_STATUSES
    store = JobStore(_config)
    for meta in store.list_jobs():
        directory = store.job_dir(meta["job_id"])
        if meta.get("status") in _ACTIVE_STATUSES:
            WorkControl(directory).request()
        for batch in directory.glob("export/shorts/*"):
            manifest = batch / "manifest.json"
            if manifest.is_file():
                try:
                    if json.loads(manifest.read_text())["status"] in {"queued", "rendering"}:
                        WorkControl(batch).request()
                except (OSError, ValueError, KeyError):
                    pass


def stop():
    pause_all()
    if _server:
        _server.should_exit = True
