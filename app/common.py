"""Shared loaders, wording and display helpers for all app pages.

Business-facing pages use plain language only. Technical detail lives on the
"About the model" page.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Artifacts, explain, load_artifacts, score, suggest_actions  # noqa: E402
from src.preprocess import add_age_band, load_raw  # noqa: E402

RED, BLUE, ORANGE, GREEN = "#D62728", "#1F77B4", "#FF7F0E", "#2CA02C"
RISK_COLORS = {"High": RED, "Medium": ORANGE, "Low": GREEN}
RISK_ORDER = ["High", "Medium", "Low"]
RISK_GUIDANCE = {
    "High": "Contact this customer soon with a retention offer.",
    "Medium": "Keep an eye on this customer and include them in the next engagement campaign.",
    "Low": "No action needed right now.",
}


# ------------------------------------------------------------------ loaders ---
@st.cache_resource(show_spinner="Loading the prediction model...")
def get_artifacts() -> Artifacts:
    return load_artifacts()


def require_artifacts() -> Artifacts:
    try:
        return get_artifacts()
    except FileNotFoundError:
        st.error("The prediction model is not available yet. Ask your data team to train it "
                 "(`python -m src.train`).")
        st.stop()


@st.cache_data
def get_metrics() -> dict | None:
    return json.loads(config.METRICS_FILE.read_text()) if config.METRICS_FILE.exists() else None


@st.cache_data
def get_history() -> pd.DataFrame:
    """Historical customer data with readable labels, for the insights page."""
    df = load_raw()
    df["Age group"] = add_age_band(df).astype(str)
    df["Activity"] = df["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    df["Products held"] = df["NumOfProducts"].map(lambda n: f"{n} product" + ("s" if n > 1 else ""))
    df["Balance status"] = (df["Balance"] == 0).map({True: "Zero balance", False: "Has a balance"})
    df["Has credit card"] = df["HasCrCard"].map({1: "Yes", 0: "No"})
    return df


@st.cache_data
def template_csv() -> bytes:
    sample = load_raw(config.TEST_FILE).sample(25, random_state=1)
    return sample[["CustomerId", "Surname"] + config.FEATURE_COLS].to_csv(index=False).encode()


# ------------------------------------------------------------- plain wording ---
def _num(v) -> str:
    return f"{v:,.0f}"


REASON_TEXT = {
    "NumOfProducts": lambda v: f"Holds {int(v)} product" + ("s" if int(v) != 1 else ""),
    "Age": lambda v: f"Age {int(v)}",
    "Gender": lambda v: f"Gender: {v}",
    "IsActiveMember": lambda v: "Active member" if int(v) == 1 else "Not an active member",
    "Geography": lambda v: f"Based in {v}",
    "Balance": lambda v: "Zero account balance" if v == 0 else f"Account balance {_num(v)}",
    "CreditScore": lambda v: f"Credit score {int(v)}",
    "Tenure": lambda v: f"{int(v)} year" + ("s" if int(v) != 1 else "") + " with the bank",
    "EstimatedSalary": lambda v: f"Estimated salary {_num(v)}",
    "HasCrCard": lambda v: "Has a credit card" if int(v) == 1 else "No credit card",
    "Satisfaction Score": lambda v: f"Satisfaction {int(v)}/5",
    "Card Type": lambda v: f"{str(v).title()} card",
    "Point Earned": lambda v: f"{int(v)} loyalty points",
}
DERIVED_TEXT = {
    "Balance-to-salary ratio": "Balance compared with salary",
    "Age × active member": "Age combined with account activity",
}


def reason_text(r: dict) -> str:
    if r["feature"] in DERIVED_TEXT:
        return DERIVED_TEXT[r["feature"]]
    fn = REASON_TEXT.get(r["feature"])
    return fn(r["value"]) if fn and r["value"] is not None else r["feature"]


def strength(impact: float) -> str:
    a = abs(impact)
    return "strong" if a >= 0.5 else "moderate" if a >= 0.2 else "slight"


def ratio_phrase(x: float) -> str:
    """0.738 -> 'about 3 in 4', 0.54 -> 'about half'."""
    if abs(x - 0.5) < 0.05:
        return "about half"
    best = min(((n, d) for d in range(2, 6) for n in range(1, d)), key=lambda nd: abs(x - nd[0] / nd[1]))
    return f"about {best[0]} in {best[1]}"


def trust_statement(metrics: dict | None) -> str:
    if not metrics:
        return "Use these predictions to prioritise outreach, not as a certainty."
    m = metrics["metrics"]
    return (f"Of customers who actually leave, this tool flags {ratio_phrase(m['recall'])} as High risk. "
            f"Of customers flagged High risk, {ratio_phrase(m['precision'])} go on to leave. "
            "Use it to prioritise outreach, not as a certainty.")


def risk_badge(level: str, size: str = "1.4rem") -> str:
    return (f"<span style='background:{RISK_COLORS[level]};color:white;padding:0.2em 0.7em;"
            f"border-radius:0.5em;font-weight:600;font-size:{size}'>{level} risk</span>")


# --------------------------------------------------------------- rendering ---
def render_reasons(reasons: list[dict]) -> None:
    up = [r for r in reasons if r["impact"] > 0]
    down = [r for r in reasons if r["impact"] <= 0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔺 Pushing risk up**")
        for r in up or []:
            st.markdown(f"- {reason_text(r)} · *{strength(r['impact'])}*")
        if not up:
            st.caption("Nothing significant.")
    with c2:
        st.markdown("**🔻 Keeping risk down**")
        for r in down:
            st.markdown(f"- {reason_text(r)} · *{strength(r['impact'])}*")
        if not down:
            st.caption("Nothing significant.")


def render_actions(reasons: list[dict]) -> None:
    st.markdown("**Recommended actions**")
    for a in suggest_actions(reasons):
        st.markdown(f"- {a}")


# ------------------------------------------------------------ batch scoring ---
def score_list(df: pd.DataFrame, art: Artifacts) -> pd.DataFrame:
    """Score a list and add the main reason and first suggested action for each customer."""
    scored = score(df, art)
    all_reasons = explain(df, art, top_n=5)
    scored["main_reason"] = [
        next((reason_text(r) for r in rs if r["impact"] > 0), "No strong risk factor") for rs in all_reasons
    ]
    scored["suggested_action"] = [
        suggest_actions(rs, max_actions=1)[0] if lvl != "Low" else RISK_GUIDANCE["Low"]
        for rs, lvl in zip(all_reasons, scored["risk_level"])
    ]
    return scored
