"""Headless smoke test of the Streamlit UI (skipped when processed data is absent)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/oof_cst_geo_visits.csv").exists(), reason="processed data missing")


def test_app_renders_and_signs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
    assert at.subheader[0].value.startswith(("⚑", "▲", "●", "▼"))
    for src in ("cnn", "trial"):
        at.sidebar.radio[0].set_value(src).run()
        assert not at.exception


def test_demo_replay_matches_engine_semantics():
    from visentia_te import demo

    vv = demo.run_eye(demo.DEFAULT_EYE)
    assert [v.packet.action for v in vv[:3]] == ["LOADING"] * 3
    assert all(v.packet.next_interval_wk in range(4, 17) for v in vv)
    assert any(v.packet.flagged for v in vv)
