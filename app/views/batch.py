"""Score a list of customers from a CSV file."""
import pandas as pd
import streamlit as st

from app.common import (
    RISK_GUIDANCE, RISK_ORDER, get_metrics, render_actions, render_reasons, require_artifacts, risk_badge,
    score_list, template_csv, trust_statement,
)
from src import config
from src.predict import explain
from src.preprocess import InputValidationError

art = require_artifacts()

st.title("📋 Score a customer list")
st.caption("Upload a list of customers to find out who is most likely to leave and who to contact first.")

COLUMN_HELP = {
    "CreditScore": "Credit score (300-900)",
    "Geography": "Country: France, Germany or Spain",
    "Gender": "Female or Male",
    "Age": "Age in years",
    "Tenure": "Years with the bank (0-10)",
    "Balance": "Account balance",
    "NumOfProducts": "Number of products held (1-4)",
    "HasCrCard": "Has a credit card: 1 = yes, 0 = no",
    "IsActiveMember": "Uses the account actively: 1 = yes, 0 = no",
    "EstimatedSalary": "Estimated yearly salary",
    "Satisfaction Score": "Satisfaction (1-5)",
    "Card Type": "Card tier: DIAMOND, GOLD, PLATINUM or SILVER",
    "Point Earned": "Loyalty points earned",
}

c1, c2 = st.columns([1, 2])
with c1.container(border=True):
    st.markdown("**Step 1: get the file format**")
    st.download_button("Download example file", template_csv(), "example_customers.csv", "text/csv",
                       width="stretch")
    with st.expander("Which columns does the file need?"):
        st.table(pd.DataFrame({"Column": COLUMN_HELP.keys(), "Meaning": COLUMN_HELP.values()}).set_index("Column"))
        st.caption("Extra columns such as `CustomerId` or `Surname` are kept in the results.")
with c2.container(border=True):
    st.markdown("**Step 2: upload your list**")
    upload = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")

if upload is None:
    st.stop()

try:
    data = pd.read_csv(upload)
except Exception:
    st.error("This file could not be read. Please upload a CSV file (comma-separated).")
    st.stop()

try:
    with st.spinner(f"Scoring {len(data):,} customers..."):
        scored = score_list(data, art)
except InputValidationError as e:
    st.error("Some details in the file need fixing before it can be scored:")
    for err in e.errors:
        st.markdown(f"- {err}")
    st.stop()

scored["risk_level"] = pd.Categorical(scored["risk_level"], RISK_ORDER, ordered=True)
scored = scored.sort_values(["risk_level", "churn_probability"], ascending=[True, False])
counts = scored["risk_level"].value_counts()

k = st.columns(4)
k[0].metric("Customers scored", f"{len(scored):,}")
k[1].metric("🔴 High risk: contact soon", f"{counts.get('High', 0):,}")
k[2].metric("🟠 Medium risk: monitor", f"{counts.get('Medium', 0):,}")
k[3].metric("Expected to leave", f"{scored['churn_probability'].sum():,.0f}",
            help="Estimated number of customers in this list who will leave, if nothing is done.")

show = st.multiselect("Show risk levels", RISK_ORDER, default=RISK_ORDER)
view = scored[scored["risk_level"].isin(show)]

id_cols = [c for c in ["CustomerId", "Surname"] if c in view.columns]
display = view[id_cols + ["churn_probability", "risk_level", "main_reason", "suggested_action",
                          "Geography", "Age", "NumOfProducts", "IsActiveMember", "Balance"]]
st.dataframe(
    display,
    hide_index=True,
    width="stretch",
    height=420,
    column_config={
        "CustomerId": st.column_config.TextColumn("Customer ID"),
        "churn_probability": st.column_config.ProgressColumn(
            "Likelihood of leaving", min_value=0, max_value=1, format="percent"),
        "risk_level": st.column_config.TextColumn("Risk"),
        "main_reason": st.column_config.TextColumn("Main reason", width="medium"),
        "suggested_action": st.column_config.TextColumn("Suggested action", width="large"),
        "Geography": "Country",
        "NumOfProducts": st.column_config.NumberColumn("Products"),
        "IsActiveMember": st.column_config.CheckboxColumn("Active"),
        "Balance": st.column_config.NumberColumn("Balance", format="%,.0f"),
    },
)

export = scored.rename(columns={
    "churn_probability": "likelihood_of_leaving", "risk_level": "risk", "predicted_churn": "flag_for_contact",
})
st.download_button("Download results", export.to_csv(index=False).encode(), "customer_churn_risk.csv",
                   "text/csv", type="primary")

st.divider()
st.subheader("Look at one customer in detail")
options = view.index[:500]
if len(options) == 0:
    st.caption("No customers match the selected risk levels.")
    st.stop()
idx = st.selectbox(
    "Customer",
    options,
    format_func=lambda i: (f"{scored.at[i, 'CustomerId']} · " if "CustomerId" in scored else f"Row {i + 1} · ")
    + (f"{scored.at[i, 'Surname']} · " if "Surname" in scored else "")
    + f"{scored.at[i, 'churn_probability']:.0%} ({scored.at[i, 'risk_level']})",
)
level = scored.at[idx, "risk_level"]
reasons = explain(data.loc[[idx]], art)[0]
with st.container(border=True):
    st.markdown(f"{risk_badge(level, '1.1rem')} &nbsp; Likelihood of leaving: "
                f"**{scored.at[idx, 'churn_probability']:.0%}**. {RISK_GUIDANCE[level]}", unsafe_allow_html=True)
    render_reasons(reasons)
    if level != "Low":
        render_actions(reasons)

st.info(trust_statement(get_metrics()), icon="💡")
