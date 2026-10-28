"""Each saved highlight selection has independent exports without new detection."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from snooker_ai.pot_detection.selector import PotSelector
from snooker_ai.rendering.exporter import ExportResult
from snooker_ai.types import ShotSelection
from tests.test_pot_selection import api_setup, readings, shot


def prepare_versions(config, tmp_path, monkeypatch):
    store, parent, result = api_setup(config, tmp_path)
    result.original_duration = result.metadata.duration = 80
    result.shots = [shot(10, 15), shot(30, 35).model_copy(update={"shot_id": 2}),
                    shot(50, 55).model_copy(update={"shot_id": 3})]
    store.save_analysis(result)
    store.update_progress(parent, 1, "completed", "All shots ready", shots_detected=3)
    meta=store.get_meta(parent)
    meta['scoring_gaps_unresolved']=[{'start':20,'end':26,'points':1}]
    store._write_meta(parent,meta)
    output = store.job_dir(parent) / "export" / "highlights.mp4"
    output.parent.mkdir()
    output.write_bytes(b"original all shots")
    monkeypatch.setattr("apps.api.main.create_analyzer", lambda *a, **k: pytest.fail("Detection reran"))
    scans, exports = [], []

    def scan(self, analysis, progress=None):
        scans.append(analysis.job_id)
        return (readings((20, 12), start=0, end=9.5)
                + readings((21, 12), start=11, end=29.5)
                + readings((21, 16), start=31, end=79, active=1))

    def export(self, analysis, output_dir, request):
        included = [s.shot_id for s in analysis.shots if s.included]
        exports.append((analysis.job_id, analysis.selection, included))
        path = Path(output_dir) / "highlights.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{analysis.selection.value}:{included}".encode())
        return ExportResult(joined_path=path)

    monkeypatch.setattr(PotSelector, "scan", scan)
    monkeypatch.setattr("apps.api.main.Exporter.export", export)
    return store, parent, scans, exports


def test_all_three_versions_export_their_own_shots_and_keep_original(config, tmp_path, monkeypatch):
    store, parent, scans, exports = prepare_versions(config, tmp_path, monkeypatch)
    before = {p.relative_to(store.job_dir(parent)): p.read_bytes()
              for p in store.job_dir(parent).rglob("*") if p.is_file()}
    with TestClient(create_app(config)) as client:
        assert client.get(f"/api/jobs/{parent}/progress").json()["can_create_variants"]
        points = client.post(f"/api/jobs/{parent}/versions/pots_and_fouls").json()["job_id"]
        assert client.app.state.match_queue.wait_idle()
        pots = client.post(f"/api/jobs/{parent}/versions/pots_only").json()["job_id"]
        assert client.app.state.match_queue.wait_idle()
        # The existing editor shortcut and the library must resolve the same version.
        assert client.post(f"/api/jobs/{parent}/points-highlights").json()["job_id"] == points
        assert client.post(f"/api/jobs/{parent}/versions/pots_only").json()["job_id"] == pots
        assert client.post(f"/api/jobs/{parent}/versions/all").json()["job_id"] == parent
        for child, selection, included in [(points, "pots_and_fouls", [1, 2]), (pots, "pots_only", [1])]:
            meta = client.get(f"/api/jobs/{child}/progress").json()
            assert meta["status"] == "completed" and meta["selection"] == selection
            assert meta['scoring_gaps_unresolved']==[{'start':20,'end':26,'points':1}]
            assert client.get(meta["download_url"]).content == f"{selection}:{included}".encode()
        # Restoring All shots from a filtered version includes the miss and foul.
        restored = client.post(f"/api/jobs/{pots}/versions/all").json()["job_id"]
        assert client.app.state.match_queue.wait_idle()
        assert client.get(f"/api/jobs/{restored}/download/highlights").content == b"all:[1, 2, 3]"
        assert client.post(f"/api/jobs/{pots}/versions/all").json()["job_id"] == restored
    assert len(set([parent, points, pots, restored])) == 4
    assert scans == [points, pots]  # All shots requires no score pass.
    assert exports == [(points, ShotSelection.POTS_AND_FOULS, [1, 2]),
                       (pots, ShotSelection.POTS_ONLY, [1]), (restored, ShotSelection.ALL, [1, 2, 3])]
    assert before == {p.relative_to(store.job_dir(parent)): p.read_bytes()
                      for p in store.job_dir(parent).rglob("*") if p.is_file()}


def test_old_selection_policy_creates_an_updated_version_without_overwriting_export(config, tmp_path, monkeypatch):
    store, parent, _, _ = prepare_versions(config, tmp_path, monkeypatch)
    with TestClient(create_app(config)) as client:
        child = client.post(f'/api/jobs/{parent}/versions/pots_and_fouls').json()['job_id']
        assert client.app.state.match_queue.wait_idle()
        old_export = (store.job_dir(child)/'export/highlights.mp4').read_bytes()
        meta = store.get_meta(child)
        meta['pot_selection_summary']['policy'] = 5
        store._write_meta(child, meta)
        updated = client.post(f'/api/jobs/{child}/versions/pots_and_fouls').json()['job_id']
        assert updated != child
        assert client.app.state.match_queue.wait_idle()
        from snooker_ai.pot_detection.selector import SELECTION_POLICY
        assert store.get_meta(updated)['pot_selection_summary']['policy'] == SELECTION_POLICY
        assert (store.job_dir(child)/'export/highlights.mp4').read_bytes() == old_export
        assert client.post(f'/api/jobs/{updated}/versions/pots_and_fouls').json()['job_id'] == updated


@pytest.mark.parametrize("selection", ["all", "pots_only", "pots_and_fouls"])
def test_variants_reject_changed_sources_and_unfinished_parents(config, tmp_path, monkeypatch, selection):
    store, parent, scans, exports = prepare_versions(config, tmp_path, monkeypatch)
    with TestClient(create_app(config)) as client:
        store.update_progress(parent, .5, "paused", "Paused")
        assert client.post(f"/api/jobs/{parent}/versions/{selection}").status_code == 409
        store.update_progress(parent, 1, "completed", "Ready")
        Path(store.get_meta(parent)["source_path"]).write_bytes(b"changed source")
        response = client.post(f"/api/jobs/{parent}/versions/{selection}")
        assert response.status_code == 409 and "changed" in response.json()["detail"]
        assert client.post(f"/api/jobs/{parent}/versions/not-a-selection").status_code == 422
    assert not scans and not exports
    assert len(store.list_jobs()) == 1


def test_score_recheck_preserves_previous_export_and_uses_new_reader(config, tmp_path, monkeypatch):
    store, parent, scans, exports = prepare_versions(config, tmp_path, monkeypatch)
    with TestClient(create_app(config)) as client:
        points=client.post(f'/api/jobs/{parent}/versions/pots_and_fouls').json()['job_id']
        assert client.app.state.match_queue.wait_idle()
        original=(store.job_dir(points)/'export'/'highlights.mp4').read_bytes()
        response=client.post(f'/api/jobs/{points}/recheck-scores')
        assert response.status_code==200
        refreshed=response.json()['job_id']
        assert refreshed!=points
        assert client.app.state.match_queue.wait_idle()
        assert client.get(f'/api/jobs/{refreshed}/progress').json()['status']=='completed'
        assert client.post(f'/api/jobs/{points}/recheck-scores').json()['job_id']==refreshed
        assert client.post(f'/api/jobs/{parent}/recheck-scores').status_code==400
    assert scans==[points,refreshed]
    assert (store.job_dir(points)/'export'/'highlights.mp4').read_bytes()==original
