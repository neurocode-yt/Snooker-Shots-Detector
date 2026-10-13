"""A finished download must reflect the user's saved review changes."""

from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.rendering.exporter import ExportResult
from tests.test_pot_selection import api_setup


def test_review_edit_preserves_old_file_but_requires_export_before_download(
    config, tmp_path, monkeypatch
):
    store, job, _ = api_setup(config, tmp_path)
    path = store.job_dir(job) / "export" / "highlights.mp4"
    path.parent.mkdir()
    path.write_bytes(b"previous video")
    store.update_progress(job, 1, "completed", "Ready")

    def export(self, analysis, output_dir, request):
        assert analysis.shots[0].clip_end == 14
        output = Path(output_dir) / "highlights.mp4"
        output.write_bytes(b"updated highlights")
        return ExportResult(joined_path=output)

    monkeypatch.setattr("apps.api.main.Exporter.export", export)
    with TestClient(create_app(config)) as client:
        assert client.get(f"/api/jobs/{job}/progress").json()["download_url"]
        assert client.patch(f"/api/jobs/{job}/shots/1", json={"clip_end": 14}).status_code == 200
        progress = client.get(f"/api/jobs/{job}/progress").json()
        assert progress["export_needs_update"]
        assert progress["download_url"] is None
        assert client.get(f"/api/jobs/{job}/download/highlights").status_code == 409
        assert path.read_bytes() == b"previous video"
        response = client.post(f"/api/jobs/{job}/export", json={"export_clips": False})
        assert response.status_code == 200
        progress = client.get(f"/api/jobs/{job}/progress").json()
        assert not progress["export_needs_update"]
        assert client.get(progress["download_url"]).content == b"updated highlights"
