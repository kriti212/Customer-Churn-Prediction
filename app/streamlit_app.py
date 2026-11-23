"""Customer Retention Assistant: entry point with multipage navigation.

Run from the project root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common import require_artifacts  # noqa: E402

st.set_page_config(page_title="Customer Retention Assistant", page_icon="🏦", layout="wide")

VIEWS = Path(__file__).parent / "views"
nav = st.navigation(
    {
        "Predict": [
            st.Page(VIEWS / "predict.py", title="Check a customer", icon="🔍", default=True),
            st.Page(VIEWS / "batch.py", title="Score a customer list", icon="📋"),
        ],
        "Understand": [
            st.Page(VIEWS / "insights.py", title="Customer insights", icon="📊"),
            st.Page(VIEWS / "about.py", title="About the model", icon="ℹ️"),
        ],
    }
)

art = require_artifacts()
trained = datetime.fromisoformat(art.metadata["trained_at"]).strftime("%d %b %Y")
st.sidebar.caption(f"Model last updated: {trained}")

nav.run()
