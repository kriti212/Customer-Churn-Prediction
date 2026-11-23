"""Check a single customer."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import (
    RISK_COLORS, RISK_GUIDANCE, get_metrics, render_actions, render_reasons, require_artifacts, risk_badge,
    trust_statement,
)
from src import config
from src.predict import explain, score

art = require_artifacts()

st.title("🔍 Check a customer")
st.caption("Enter a customer's details to see how likely they are to leave, why, and what to do about it.")

with st.form("customer"):
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**About the customer**")
        geography = st.selectbox("Country", config.CATEGORY_VALUES["Geography"])
        gender = st.selectbox("Gender", config.CATEGORY_VALUES["Gender"])
        age = st.slider("Age", 18, 92, 40)
        salary = st.number_input("Estimated yearly salary", 0.0, 250_000.0, 100_000.0, step=1_000.0)
    with c2.container(border=True):
        st.markdown("**Relationship with the bank**")
        tenure = st.slider("Years with the bank", 0, 10, 5)
        active = st.radio("Uses the account actively?", ["Yes", "No"], horizontal=True,
                          help="Has the customer logged in or transacted recently?")
        satisfaction = st.slider("Satisfaction score", 1, 5, 3, help="1 = very unhappy, 5 = very happy")
        points = st.number_input("Loyalty points earned", 0, 1_500, 600)
    with c3.container(border=True):
        st.markdown("**Products and finances**")
        products = st.selectbox("Number of products held", [1, 2, 3, 4], index=1)
        balance = st.number_input("Account balance", 0.0, 300_000.0, 75_000.0, step=1_000.0)
        credit = st.number_input("Credit score", 300, 900, 650)
        card = st.radio("Has a credit card?", ["Yes", "No"], horizontal=True)
        card_type = st.selectbox("Card tier", config.CATEGORY_VALUES["Card Type"])
    submitted = st.form_submit_button("Check churn risk", type="primary", width="stretch")

if submitted:
    row = pd.DataFrame([{
        "CreditScore": credit, "Geography": geography, "Gender": gender, "Age": age, "Tenure": tenure,
        "Balance": balance, "NumOfProducts": products, "HasCrCard": int(card == "Yes"),
        "IsActiveMember": int(active == "Yes"), "EstimatedSalary": salary,
        "Satisfaction Score": satisfaction, "Card Type": card_type, "Point Earned": points,
    }])
    result = score(row, art).iloc[0]
    reasons = explain(row, art)[0]
    prob, level = float(result.churn_probability), result.risk_level

    left, right = st.columns([1, 1.6])
    with left.container(border=True):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number={"suffix": "%", "valueformat": ".0f", "font": {"color": RISK_COLORS[level]}},
            title={"text": "Likelihood of leaving"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": RISK_COLORS[level]},
                "steps": [
                    {"range": [0, art.threshold * 50], "color": "#E3F4E3"},
                    {"range": [art.threshold * 50, art.threshold * 100], "color": "#FFEBD1"},
                    {"range": [art.threshold * 100, 100], "color": "#FADADA"},
                ],
            },
        ))
        fig.update_layout(height=250, margin=dict(l=25, r=25, t=50, b=0))
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"<div style='text-align:center'>{risk_badge(level)}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;margin-top:0.8em'>{RISK_GUIDANCE[level]}</p>",
                    unsafe_allow_html=True)

    with right.container(border=True):
        st.markdown("#### Why this customer got this score")
        render_reasons(reasons)
        st.divider()
        if level == "Low":
            st.markdown("**Recommended actions**")
            st.markdown(f"- {RISK_GUIDANCE['Low']}")
        else:
            render_actions(reasons)

    st.info(trust_statement(get_metrics()), icon="💡")
