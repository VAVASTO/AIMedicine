from pathlib import Path
import json
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_saved_cases_render_without_api_calls(monkeypatch):
    pytest.importorskip("streamlit")
    if not (ROOT / "reports/offline/summary.json").exists():
        pytest.skip("Generate offline demo reports before UI smoke test")
    import httpx
    from streamlit.testing.v1 import AppTest

    def forbidden_api(*args, **kwargs):
        raise AssertionError("Replay must not call an LLM provider")

    monkeypatch.setattr(httpx.Client, "post", forbidden_api)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    app.selectbox(key="run_directory").set_value(ROOT / "reports/offline").run(timeout=30)
    assert not app.exception
    for index in range(3):
        app.selectbox(key="patient_selection").set_value(index).run(timeout=30)
        assert not app.exception
        assert len(app.dataframe) >= 3
        assert not app.get("doc_string"), "UI must not render Streamlit object documentation"
        assert all("DeltaGenerator" not in item.value for item in app.markdown)
    if list((ROOT / "reports/evaluation").glob("fault_*.json")):
        app.selectbox(key="run_directory").set_value(ROOT / "reports/evaluation").run(timeout=30)
        assert not app.exception
        assert any("внесённым дефектом" in warning.value for warning in app.warning)
    live = [button for button in app.button if button.label == "Запустить live"]
    assert len(live) == 1 and live[0].disabled


def test_failed_repeat_does_not_present_stale_live_summary_as_success(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    source = ROOT / "reports/offline"
    if not (source / "summary.json").exists():
        pytest.skip("Generate offline demo reports before UI smoke test")
    import httpx
    from streamlit.testing.v1 import AppTest

    def forbidden_api(*args, **kwargs):
        raise AssertionError("Replay must not call an LLM provider")

    monkeypatch.setattr(httpx.Client, "post", forbidden_api)
    shutil.copy2(ROOT / "app.py", tmp_path / "app.py")
    (tmp_path / "data").mkdir()
    for path in (ROOT / "data").glob("*.json"):
        shutil.copy2(path, tmp_path / "data" / path.name)
    offline = tmp_path / "reports/offline"
    live = tmp_path / "reports/live"
    shutil.copytree(source, offline)
    live.mkdir()
    summary = json.loads((source / "summary.json").read_text())
    summary["mode"] = "live"
    for row in summary["cases"]:
        trace = json.loads((source / row["path"]).read_text())
        trace["mode"] = "live"
        (live / row["path"]).write_text(json.dumps(trace))
    (live / "summary.json").write_text(json.dumps(summary))
    (live / "run_status.json").write_text(json.dumps({"status": "failed"}))
    app = AppTest.from_file(str(tmp_path / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="run_directory").value == offline
    app.selectbox(key="run_directory").set_value(live).run(timeout=30)
    assert not app.exception
    assert any("не подтверждён полный завершённый прогон" in warning.value for warning in app.warning)


def test_public_demo_replays_without_loading_credentials_or_start_controls(monkeypatch):
    pytest.importorskip("streamlit")
    if not (ROOT / "reports/offline/summary.json").exists():
        pytest.skip("Generate offline demo reports before UI smoke test")
    import httpx
    import dotenv
    from streamlit.testing.v1 import AppTest

    def forbidden(*args, **kwargs):
        raise AssertionError("Public replay must not load credentials or call the provider")

    monkeypatch.setenv("AIMEDICINE_PUBLIC_DEMO", "1")
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(httpx.Client, "post", forbidden)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert not any(button.label in ("Запустить live", "Запустить offline") for button in app.button)
    assert not app.checkbox
    assert all("development" not in str(option) for option in app.selectbox(key="run_directory").options)
    app.selectbox(key="run_directory").set_value(ROOT / "reports/offline").run(timeout=30)
    for index in range(3):
        app.selectbox(key="patient_selection").set_value(index).run(timeout=30)
        assert not app.exception
    if list((ROOT / "reports/evaluation_live").glob("fault_*.json")):
        app.selectbox(key="run_directory").set_value(ROOT / "reports/evaluation_live").run(timeout=30)
        assert not app.exception
        assert any("внесённым дефектом" in warning.value for warning in app.warning)
