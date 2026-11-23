"""Smoke tests: every page renders without errors. Run: pytest -q"""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import config

pytestmark = pytest.mark.skipif(not config.MODEL_FILE.exists(), reason="model not trained yet")
ENTRY = str(config.ROOT / "app" / "streamlit_app.py")


def run_page(page=None):
    at = AppTest.from_file(ENTRY, default_timeout=180).run()
    if page:
        at.switch_page(str(config.ROOT / "app" / page)).run()
    return at


def test_home_is_check_a_customer_and_predicts():
    at = run_page()
    assert not at.exception
    assert at.title[0].value.startswith("🔍 Check a customer")
    at.button[0].click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "risk</span>" in text and "Pushing risk up" in text


@pytest.mark.parametrize("page,title", [
    ("views/batch.py", "Score a customer list"),
    ("views/insights.py", "Customer insights"),
    ("views/about.py", "About the model"),
])
def test_pages_render(page, title):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert title in at.title[0].value


def test_business_pages_have_no_ml_jargon():
    jargon = ["ROC", "AUC", "SHAP", "XGBoost", "Optuna", "calibrat", "threshold", "Brier"]
    for page in [None, "views/insights.py"]:
        at = run_page(page)
        if page is None:
            at.button[0].click().run()
        visible = " ".join(e.value for e in list(at.markdown) + list(at.caption) + list(at.title) + list(at.info)
                           + list(at.subheader) if isinstance(e.value, str))
        found = [j for j in jargon if j.lower() in visible.lower()]
        assert not found, f"{page or 'home'} shows ML jargon: {found}"


def test_score_list_adds_reason_and_action():
    from app.common import get_artifacts, score_list
    df = pd.read_csv(config.TEST_FILE).head(50)
    out = score_list(df, get_artifacts())
    assert {"churn_probability", "risk_level", "main_reason", "suggested_action"} <= set(out.columns)
    assert out["main_reason"].str.len().gt(0).all()
    assert "Gender" not in " ".join(out["suggested_action"])


def test_ratio_phrase():
    from app.common import ratio_phrase
    assert ratio_phrase(0.738) == "about 3 in 4"
    assert ratio_phrase(0.541) == "about half"
